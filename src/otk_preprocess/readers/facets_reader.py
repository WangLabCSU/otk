"""Reader and feature adapter for FACETS CSV result files."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import warnings

import pandas as pd


FileCollection = List[Path]

SEGMENTS_SUFFIX = "_facets_segments.csv"
PURITY_PLOIDY_SUFFIX = "_facets_pp.csv"
SEGMENT_COLUMNS = ("chrom", "start", "end", "tcn.em", "lcn.em")
PURITY_PLOIDY_COLUMNS = ("purity", "ploidy")


@dataclass
class FacetsResults:
	"""Raw region and sample tables read from paired FACETS CSV files."""

	segments: pd.DataFrame
	sample_parameters: pd.DataFrame


@dataclass
class FacetsFeatureTables:
	"""Standardized region- and sample-level tables for OTK preprocessing."""

	region_features: pd.DataFrame
	sample_features: pd.DataFrame


def _sample_id_from_segment_path(segment_path: Path) -> str:
	"""Derive the sample identifier from a FACETS segment filename."""
	return segment_path.name[: -len(SEGMENTS_SUFFIX)]


def _purity_ploidy_path(segment_path: Path) -> Path:
	"""Return the paired purity/ploidy file for a FACETS segment file."""
	sample_id = _sample_id_from_segment_path(segment_path)
	return segment_path.with_name(f"{sample_id}{PURITY_PLOIDY_SUFFIX}")


def _require_columns(data: pd.DataFrame, required_columns: tuple, file_path: Path) -> None:
	"""Raise a clear error when a FACETS result file is missing required fields."""
	missing_columns = [column for column in required_columns if column not in data.columns]
	if missing_columns:
		formatted_columns = ", ".join(missing_columns)
		raise ValueError(f"FACETS file is missing required columns ({formatted_columns}): {file_path}")


def read_facets_results(file_paths: Union[FileCollection, List[str]]) -> FacetsResults:
	"""Read paired FACETS segment and purity/ploidy CSV files for all samples.

	Every ``*_facets_segments.csv`` file requires a same-directory
	``*_facets_pp.csv`` file. The segment file provides genomic copy-number
	regions, while the paired file provides one purity/ploidy row per sample.
	"""
	if not file_paths:
		raise ValueError("At least one FACETS segment file is required")

	segment_tables = []
	parameter_tables = []
	seen_samples = set()

	for file_path in file_paths:
		segment_path = Path(file_path)
		if not segment_path.is_file():
			raise FileNotFoundError(f"FACETS segment file does not exist: {segment_path}")

		sample_id = _sample_id_from_segment_path(segment_path)
		if sample_id in seen_samples:
			raise ValueError(f"Duplicate FACETS sample identifier: {sample_id}")
		seen_samples.add(sample_id)

		purity_ploidy_path = _purity_ploidy_path(segment_path)
		if not purity_ploidy_path.is_file():
			raise FileNotFoundError(
				f"Missing FACETS purity/ploidy file for sample '{sample_id}': "
				f"{purity_ploidy_path}"
			)

		segments = pd.read_csv(segment_path)
		_require_columns(segments, SEGMENT_COLUMNS, segment_path)
		segments = segments.loc[:, SEGMENT_COLUMNS].copy()
		segments["sample"] = sample_id
		segment_tables.append(segments)

		sample_parameters = pd.read_csv(purity_ploidy_path)
		_require_columns(sample_parameters, PURITY_PLOIDY_COLUMNS, purity_ploidy_path)
		if len(sample_parameters) != 1:
			raise ValueError(
				"FACETS purity/ploidy file must contain exactly one row per sample: "
				f"{purity_ploidy_path}"
			)
		sample_parameters = sample_parameters.loc[:, PURITY_PLOIDY_COLUMNS].copy()
		sample_parameters.insert(0, "sample", sample_id)
		parameter_tables.append(sample_parameters)

	return FacetsResults(
		segments=pd.concat(segment_tables, ignore_index=True),
		sample_parameters=pd.concat(parameter_tables, ignore_index=True),
	)


def extract_features_from_facets(results: FacetsResults) -> FacetsFeatureTables:
	"""Convert raw FACETS CSV data into standardized OTK feature tables.

	FACETS ``tcn.em`` and ``lcn.em`` become the OTK-compatible ``segVal`` and
	``minor_cn`` fields. 
	"""
	regions = results.segments.rename(
		columns={"chrom": "chromosome", "tcn.em": "segVal", "lcn.em": "minor_cn"}
	).loc[:, ["sample", "chromosome", "start", "end", "segVal", "minor_cn"]]
	samples = results.sample_parameters.loc[:, ["sample", "purity", "ploidy"]].copy()

	for column in ("start", "end", "segVal", "minor_cn"):
		regions[column] = pd.to_numeric(regions[column], errors="raise")
	for column in ("purity", "ploidy"):
		samples[column] = pd.to_numeric(samples[column], errors="raise")

	valid_intervals = regions["start"] < regions["end"]
	invalid_intervals = regions[~valid_intervals]          # invalid rows: start >= end
	if not invalid_intervals.empty:
		warnings.warn(
			"FACETS segments must have start < end; invalid rows found for samples: "
			+ ", ".join(sorted(invalid_intervals["sample"].unique()))
		)
	regions = regions[valid_intervals].copy()              # valid rows

	return FacetsFeatureTables(region_features=regions, sample_features=samples)
