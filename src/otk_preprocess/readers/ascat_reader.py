"""Reader and feature adapter for ASCAT CSV result files."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import pandas as pd


FileCollection = List[Path]

SEGMENTS_SUFFIX = "_ascat_segments.csv"
PURITY_PLOIDY_SUFFIX = "_ascat_pp.csv"
SEGMENT_COLUMNS = ("sample", "chr", "startpos", "endpos", "nMajor", "nMinor")
PURITY_PLOIDY_COLUMNS = ("sample", "purity", "ploidy")


@dataclass
class AscatResults:
	"""Raw region and sample tables read from paired ASCAT CSV files."""

	segments: pd.DataFrame
	sample_parameters: pd.DataFrame


@dataclass
class AscatFeatureTables:
	"""Standardized region- and sample-level tables for OTK preprocessing."""

	region_features: pd.DataFrame
	sample_features: pd.DataFrame


def _batch_id_from_segment_path(segment_path: Path) -> str:
	"""Derive the batch prefix used to pair ASCAT output files."""
	return segment_path.name[: -len(SEGMENTS_SUFFIX)]


def _purity_ploidy_path(segment_path: Path) -> Path:
	"""Return the paired sample purity/ploidy CSV path for one ASCAT batch."""
	batch_id = _batch_id_from_segment_path(segment_path)
	return segment_path.with_name(f"{batch_id}{PURITY_PLOIDY_SUFFIX}")


def _require_columns(data: pd.DataFrame, required_columns: tuple, file_path: Path) -> None:
	"""Raise a clear error when an ASCAT result file is missing required fields."""
	missing_columns = [column for column in required_columns if column not in data.columns]
	if missing_columns:
		formatted_columns = ", ".join(missing_columns)
		raise ValueError(f"ASCAT file is missing required columns ({formatted_columns}): {file_path}")


def read_ascat_results(file_paths: Union[FileCollection, List[str]]) -> AscatResults:
	"""Read and merge paired ASCAT segment and sample-parameter CSV batches.

	The batch filename prefix is not a sample identifier: both files contain a
	``sample`` column and may hold multiple samples. Each segments file requires
	a same-directory ``*_ascat_pp.csv`` partner whose samples exactly match the
	corresponding segments table.
	"""
	if not file_paths:
		raise ValueError("At least one ASCAT segment file is required")

	segment_tables = []
	parameter_tables = []
	seen_samples = set()

	for file_path in file_paths:
		segment_path = Path(file_path)
		if not segment_path.is_file():
			raise FileNotFoundError(f"ASCAT segment file does not exist: {segment_path}")

		purity_ploidy_path = _purity_ploidy_path(segment_path)
		if not purity_ploidy_path.is_file():
			raise FileNotFoundError(
				f"Missing ASCAT purity/ploidy file for batch '{_batch_id_from_segment_path(segment_path)}': "
				f"{purity_ploidy_path}"
			)

		segments = pd.read_csv(segment_path)
		_require_columns(segments, SEGMENT_COLUMNS, segment_path)
		segments = segments.loc[:, SEGMENT_COLUMNS].copy()

		sample_parameters = pd.read_csv(purity_ploidy_path)
		_require_columns(sample_parameters, PURITY_PLOIDY_COLUMNS, purity_ploidy_path)
		sample_parameters = sample_parameters.loc[:, PURITY_PLOIDY_COLUMNS].copy()

		segment_samples = set(segments["sample"])
		parameter_samples = set(sample_parameters["sample"])
		if segment_samples != parameter_samples:
			raise ValueError(
				"ASCAT segments and purity/ploidy files must contain the same samples: "
				f"{segment_path}, {purity_ploidy_path}"
			)
		if sample_parameters["sample"].duplicated().any():
			raise ValueError(f"ASCAT purity/ploidy file has duplicate samples: {purity_ploidy_path}")
		duplicate_samples = seen_samples.intersection(segment_samples)
		if duplicate_samples:
			raise ValueError(
				"ASCAT sample identifiers occur in more than one batch: "
				+ ", ".join(sorted(duplicate_samples))
			)
		seen_samples.update(segment_samples)

		segment_tables.append(segments)
		parameter_tables.append(sample_parameters)

	return AscatResults(
		segments=pd.concat(segment_tables, ignore_index=True),
		sample_parameters=pd.concat(parameter_tables, ignore_index=True),
	)


def extract_features_from_ascat(results: AscatResults) -> AscatFeatureTables:
	"""Convert ASCAT copy-number results into standardized OTK feature tables.

	ASCAT major and minor allele copy numbers are summed to form ``segVal``,
	matching GCAP's ``total_cn = major_cn + minor_cn`` conversion. 
	"""
	regions = results.segments.rename(
		columns={
			"chr": "chromosome",
			"startpos": "start",
			"endpos": "end",
			"nMinor": "minor_cn",
		}
	).loc[:, ["sample", "chromosome", "start", "end", "nMajor", "minor_cn"]]
	samples = results.sample_parameters.loc[:, ["sample", "purity", "ploidy"]].copy()

	for column in ("start", "end", "nMajor", "minor_cn"):
		regions[column] = pd.to_numeric(regions[column], errors="raise")
	for column in ("purity", "ploidy"):
		samples[column] = pd.to_numeric(samples[column], errors="raise")
	regions["segVal"] = regions["nMajor"] + regions["minor_cn"]
	regions = regions.loc[:, ["sample", "chromosome", "start", "end", "segVal", "minor_cn"]]

	return AscatFeatureTables(region_features=regions, sample_features=samples)
