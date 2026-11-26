"""Collapse copy-number segments to genes and assemble OTK input features."""

from pathlib import Path
from typing import Any, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd


GENOME_BUILDS = ("hg19", "hg38", "mm10")
FREQUENCY_COLUMNS = ("freq_Linear", "freq_BFB", "freq_Circular", "freq_HR")
REGION_COLUMNS = ("sample", "chromosome", "start", "end", "segVal", "minor_cn")

ANNOTATION_DATA_DIR = Path(__file__).resolve().parent


def _feature_tables_parts(feature_tables: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Get region- and sample-level tables from a reader result object or mapping."""
	if isinstance(feature_tables, dict):
		regions = feature_tables.get("region_features")
		samples = feature_tables.get("sample_features")
	else:
		regions = getattr(feature_tables, "region_features", None)
		samples = getattr(feature_tables, "sample_features", None)

	return regions.copy(), samples.copy()


def _normalize_chromosomes(chromosomes: pd.Series, source: str) -> pd.Series:
	"""Use the same chr-prefixed chromosome convention for inputs and references."""
	if chromosomes.isna().any():
		raise ValueError(f"{source} contains missing chromosome values")

	values = chromosomes.astype(str).str.strip()
	if values.eq("").any():
		raise ValueError(f"{source} contains empty chromosome values")
	values = values.str.replace(r"\.0$", "", regex=True)
	return values.map(lambda value: value if value.lower().startswith("chr") else f"chr{value}")


def _load_reference_tables(gene_build: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Load the gene-coordinate and prior-frequency tables for a genome build."""
	if gene_build not in GENOME_BUILDS:
		supported = ", ".join(GENOME_BUILDS)
		raise ValueError(f"Unsupported gene_build '{gene_build}'. Expected one of: {supported}")

	gene_path = ANNOTATION_DATA_DIR / f"{gene_build}_target_genes.csv"
	frequency_name = "amplicon_freq_mm10.csv" if gene_build == "mm10" else "amplicon_freq.csv"
	frequency_path = ANNOTATION_DATA_DIR / frequency_name

	if not gene_path.is_file():
		raise FileNotFoundError(f"Gene-coordinate reference file not found: {gene_path}")
	if not frequency_path.is_file():
		raise FileNotFoundError(f"Amplicon-frequency reference file not found: {frequency_path}")

	genes = pd.read_csv(gene_path)
	required_gene_columns = {"chrom", "start", "end", "gene_id"}
	missing_gene_columns = required_gene_columns.difference(genes.columns)
	if missing_gene_columns:
		raise ValueError(
			f"Gene reference is missing columns: {', '.join(sorted(missing_gene_columns))}"
		)
	genes = genes.loc[:, ["chrom", "start", "end", "gene_id"]].copy()
	genes["chrom"] = _normalize_chromosomes(genes["chrom"], "Gene reference")
	for column in ("start", "end"):
		genes[column] = pd.to_numeric(genes[column], errors="raise")
	if genes["gene_id"].isna().any() or (genes["start"] > genes["end"]).any():
		raise ValueError("Gene reference contains missing gene IDs or invalid coordinates")

	frequencies = pd.read_csv(frequency_path)
	required_frequency_columns = {"gene_id", *FREQUENCY_COLUMNS}
	missing_frequency_columns = required_frequency_columns.difference(frequencies.columns)
	if missing_frequency_columns:
		raise ValueError(
			"Amplicon-frequency reference is missing columns: "
			+ ", ".join(sorted(missing_frequency_columns))
		)
	frequencies = frequencies.loc[:, ["gene_id", *FREQUENCY_COLUMNS]].copy()
	if frequencies["gene_id"].duplicated().any():
		raise ValueError(f"Amplicon-frequency reference has duplicate gene IDs: {frequency_path}")

	return genes, frequencies


def _collapse_regions_to_genes(
	region_features: pd.DataFrame,
	genes: pd.DataFrame,
	overlap: float,
) -> pd.DataFrame:
	"""Return overlapping sample/gene records with gene coverage ratios.

	Genomic intervals use closed coordinates, so the shared length includes both
	endpoints. As in GCAP, the overlap threshold and ``intersect_ratio`` use the
	reference gene length as their denominator.
	"""
	missing_columns = set(REGION_COLUMNS).difference(region_features.columns)
	if missing_columns:
		raise ValueError(
			"Region features are missing columns: " + ", ".join(sorted(missing_columns))
		)
	if not 0 <= overlap <= 1:
		raise ValueError("overlap must be between 0 and 1")

	regions = region_features.loc[:, REGION_COLUMNS].copy()
	if regions["sample"].isna().any():
		raise ValueError("Region features contain missing sample identifiers")
	regions["sample"] = regions["sample"].astype(str)
	regions["chromosome"] = _normalize_chromosomes(regions["chromosome"], "Region features")
	for column in ("start", "end", "segVal", "minor_cn"):
		regions[column] = pd.to_numeric(regions[column], errors="raise")
	if (regions["start"] > regions["end"]).any():
		raise ValueError("Region features contain intervals where start is greater than end")

	genes_by_chromosome = {}
	for chromosome, chromosome_genes in genes.groupby("chrom", sort=False):
		chromosome_genes = chromosome_genes.sort_values("start").reset_index(drop=True)
		prefix_max_end = np.maximum.accumulate(chromosome_genes["end"].to_numpy())
		genes_by_chromosome[chromosome] = (chromosome_genes, prefix_max_end)

	records = []
	for region in regions.itertuples(index=False):
		chromosome_index = genes_by_chromosome.get(region.chromosome)
		if chromosome_index is None:
			continue
		chromosome_genes, prefix_max_end = chromosome_index

		# Binary searches skip genes that cannot overlap this segment.
		candidate_start = np.searchsorted(prefix_max_end, region.start, side="left")
		candidate_limit = chromosome_genes["start"].searchsorted(region.end, side="right")
		candidates = chromosome_genes.iloc[candidate_start:candidate_limit]
		candidates = candidates.loc[candidates["end"] >= region.start]
		for gene in candidates.itertuples(index=False):
			intersect_size = min(region.end, gene.end) - max(region.start, gene.start) + 1
			gene_length = gene.end - gene.start + 1
			intersect_ratio = intersect_size / gene_length
			if intersect_ratio >= overlap:
				records.append(
					{
						"sample": region.sample,
						"gene_id": gene.gene_id,
						"segVal": region.segVal,
						"minor_cn": region.minor_cn,
						"intersect_ratio": intersect_ratio,
					}
				)

	return pd.DataFrame(
		records,
		columns=["sample", "gene_id", "segVal", "minor_cn", "intersect_ratio"],
	)


def _prepare_extra_info(extra_info: Optional[Union[pd.DataFrame, str, Path]]) -> Optional[pd.DataFrame]:
	"""Keep useful optional metadata and normalize age/gender to OTK-friendly values."""
	if extra_info is None:
		return None
	if isinstance(extra_info, (str, Path)):
		if not Path(extra_info).exists():
			raise FileNotFoundError(f"Extra info file path does not exist: {extra_info}")
		extra_info = pd.read_csv(extra_info, dtype={"sample": str})
	if not isinstance(extra_info, pd.DataFrame):
		raise TypeError("extra_info must be a pandas DataFrame, CSV path, or None")

	optional_columns = [column for column in ("age", "gender", "type") if column in extra_info]
	optional_columns.extend(column for column in extra_info.columns if column.startswith("type_"))
	if not optional_columns:
		return None
	if "sample" not in extra_info:
		raise ValueError("extra_info must contain a 'sample' column when extra_info is provided")

	prepared = extra_info.loc[:, ["sample", *optional_columns]].copy()
	if prepared["sample"].isna().any():
		raise ValueError("extra_info contains missing sample identifiers")
	prepared["sample"] = prepared["sample"].astype(str)
	if prepared["sample"].duplicated().any():
		raise ValueError("extra_info must contain at most one row per sample")

	if "age" in prepared:
		prepared["age"] = pd.to_numeric(prepared["age"], errors="coerce")
	if "gender" in prepared:
		gender_values = prepared["gender"]
		numeric_gender = pd.to_numeric(gender_values, errors="coerce")
		text_gender = gender_values.astype("string").str.strip().str.upper()
		text_mapping = {
			"XY": 1,
			"MALE": 1,
			"M": 1,
			"XX": 0,
			"FEMALE": 0,
			"F": 0,
		}
		mapped_gender = text_gender.map(text_mapping)
		prepared["gender"] = numeric_gender.where(numeric_gender.notna(), mapped_gender).fillna(0)
	if "type" in prepared:
		prepared["type"] = prepared["type"].astype("string")

	for c in [col for col in extra_info.columns if col.startswith('type_')]:
		prepared[c] = extra_info[c].astype("int")

	return prepared


def merge_sample_and_region_features(
	annotated_region_features: pd.DataFrame,
	sample_features: pd.DataFrame,
	extra_info: Optional[Union[pd.DataFrame, str, Path]] = None,
) -> pd.DataFrame:
	"""Join per-sample metrics and optional metadata onto gene-level records."""
	if "sample" not in annotated_region_features or "sample" not in sample_features:
		raise ValueError("Both region-level and sample-level features must contain 'sample'")

	regions = annotated_region_features.copy()
	samples = sample_features.copy()
	regions["sample"] = regions["sample"].astype(str)
	samples["sample"] = samples["sample"].astype(str)

	overlapping_columns = (set(regions.columns) & set(samples.columns)) - {"sample"}
	if overlapping_columns:
		raise ValueError(
			"Region and sample features have conflicting columns: "
			+ ", ".join(sorted(overlapping_columns))
		)
	result = regions.merge(samples, on="sample", how="left", validate="many_to_one")

	prepared_extra = _prepare_extra_info(extra_info)
	if prepared_extra is not None:
		duplicate_columns = (set(result.columns) & set(prepared_extra.columns)) - {"sample"}
		if duplicate_columns:
			raise ValueError(
				"Extra metadata conflicts with existing feature columns: "
				+ ", ".join(sorted(duplicate_columns))
			)
		result = result.merge(prepared_extra, on="sample", how="left", validate="many_to_one")

	return result


def annotate_genes(
	feature_tables: Any,
	gene_build: str,
	extra_info: Optional[Union[pd.DataFrame, str, Path]] = None,
	overlap: float = 1.0,
) -> pd.DataFrame:
	"""Annotate copy-number regions and assemble a gene-level OTK input table.

	``feature_tables`` contains ``region_features`` and ``sample_features`` from
	a reader. ``gene_build`` selects hg19, hg38, or mm10 reference coordinates.
	Optional ``extra_info`` may provide sample, age, gender, and/or type columns.
	"""
	region_features, sample_features = _feature_tables_parts(feature_tables)
	genes, frequencies = _load_reference_tables(gene_build)

	annotated = _collapse_regions_to_genes(region_features, genes, overlap)
	if annotated.empty:
		warnings.warn("No copy-number regions met the gene-overlap threshold")

	annotated = annotated.merge(
		frequencies,
		on="gene_id",
		how="left",
		validate="many_to_one",
	)
	annotated.loc[:, FREQUENCY_COLUMNS] = annotated.loc[:, FREQUENCY_COLUMNS].fillna(0)

	return merge_sample_and_region_features(annotated, sample_features, extra_info)
