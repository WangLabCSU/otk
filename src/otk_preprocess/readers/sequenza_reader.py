"""Reader and feature adapter for Sequenza copy-number result files."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import pandas as pd


FileCollection = List[Path]

SEGMENTS_SUFFIX = "_segments.txt"
ALTERNATIVE_SOLUTIONS_SUFFIX = "_alternative_solutions.txt"
SEGMENT_COLUMNS = ("chromosome", "start.pos", "end.pos", "CNt", "B")


@dataclass
class SequenzaResults:
	"""Raw tables read from one or more paired Sequenza result files."""

	segments: pd.DataFrame
	sample_parameters: pd.DataFrame


@dataclass
class SequenzaFeatureTables:
	"""Standardized region-level and sample-level tables for annotating."""

	region_features: pd.DataFrame
	sample_features: pd.DataFrame


def _sample_id_from_segment_path(segment_path: Path) -> str:
	"""Derive the sample identifier using the Sequenza output file convention."""
	return segment_path.name[: -len(SEGMENTS_SUFFIX)]


def _alternative_solutions_path(segment_path: Path) -> Path:
	"""Return the paired alternative-solutions file for one segment file."""
	sample_id = _sample_id_from_segment_path(segment_path)
	return segment_path.with_name(f"{sample_id}{ALTERNATIVE_SOLUTIONS_SUFFIX}")


def _require_columns(data: pd.DataFrame, required_columns: tuple, file_path: Path) -> None:
	"""Raise a useful error when a Sequenza result table is incomplete."""
	missing_columns = [column for column in required_columns if column not in data.columns]
	if missing_columns:
		formatted_columns = ", ".join(missing_columns)
		raise ValueError(f"Sequenza file is missing required columns ({formatted_columns}): {file_path}")


def read_sequenza_results(file_paths: Union[FileCollection, List[str]]) -> SequenzaResults:
	"""Read paired Sequenza segment and alternative-solution result files.

	Each ``*_segments.txt`` file must have a same-directory
	``*_alternative_solutions.txt`` partner. The segment table contributes
	copy-number regions; the first row and first two columns of the partner
	provide the Sequenza-selected purity and ploidy values, matching GCAP's
	``read_copynumber_seqz_cn`` and ``read_copynumber_seqz_pp`` behavior.
	"""
	if not file_paths:
		raise ValueError("At least one Sequenza segment file is required")

	segment_tables = []
	parameter_tables = []
	seen_samples = set()

	for file_path in file_paths:
		segment_path = Path(file_path)
		if not segment_path.is_file():
			raise FileNotFoundError(f"Sequenza segment file does not exist: {segment_path}")

		sample_id = _sample_id_from_segment_path(segment_path)
		if sample_id in seen_samples:
			raise ValueError(f"Duplicate Sequenza sample identifier: {sample_id}")
		seen_samples.add(sample_id)

		alternative_path = _alternative_solutions_path(segment_path)
		if not alternative_path.is_file():
			raise FileNotFoundError(
				"Missing Sequenza alternative-solutions file for "
				f"sample '{sample_id}': {alternative_path}"
			)

		segments = pd.read_csv(segment_path, sep="\t")
		_require_columns(segments, SEGMENT_COLUMNS, segment_path)
		segments = segments.loc[:, SEGMENT_COLUMNS].copy()
		segments["sample"] = sample_id
		segment_tables.append(segments)

		alternatives = pd.read_csv(alternative_path, sep="\t")
		if alternatives.empty or alternatives.shape[1] < 2:
			raise ValueError(
				"Sequenza alternative-solutions file must contain at least one row "
				f"and two columns: {alternative_path}"
			)
		parameter_tables.append(
			pd.DataFrame(
				{
					"sample": [sample_id],
					"purity": [alternatives.iloc[0, 0]],
					"ploidy": [alternatives.iloc[0, 1]],
				}
			)
		)

	return SequenzaResults(
		segments=pd.concat(segment_tables, ignore_index=True),
		sample_parameters=pd.concat(parameter_tables, ignore_index=True),
	)


def extract_features_from_sequenza(results: SequenzaResults) -> SequenzaFeatureTables:
	"""Convert raw Sequenza results into OTK's region and sample feature tables.

	``CNt`` and ``B`` are mapped to OTK's ``segVal`` and ``minor_cn``. 
	"""
	regions = results.segments.rename(
		columns={
			"start.pos": "start",
			"end.pos": "end",
			"CNt": "segVal",
			"B": "minor_cn",
		}
	).loc[:, ["sample", "chromosome", "start", "end", "segVal", "minor_cn"]]
	samples = results.sample_parameters.loc[:, ["sample", "purity", "ploidy"]].copy()

	for column in ("start", "end", "segVal", "minor_cn"):
		regions[column] = pd.to_numeric(regions[column], errors="raise")
	for column in ("purity", "ploidy"):
		samples[column] = pd.to_numeric(samples[column], errors="raise")

	return SequenzaFeatureTables(region_features=regions, sample_features=samples)
