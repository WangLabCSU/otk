"""Input routing for the OTK preprocessing workflow.

This module accepts an already loaded ``pandas.DataFrame``, one CSV/TSV file, or
a directory containing one compatible family of ASCAT, Sequenza, or FACETS files.
Gene-level CSV/TSV inputs pass through to OTK; segment inputs are annotated.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from .annotation.gene_annotation import annotate_genes, merge_sample_and_region_features
from .readers.generic_reader import read_delimited_file
from .readers.ascat_reader import (
    SEGMENTS_SUFFIX as ASCAT_SEGMENTS_SUFFIX,
    extract_features_from_ascat,
    read_ascat_results,
)
from .readers.facets_reader import (
    SEGMENTS_SUFFIX as FACETS_SEGMENTS_SUFFIX,
    extract_features_from_facets,
    read_facets_results,
)
from .readers.sequenza_reader import (
    SEGMENTS_SUFFIX as SEQUENZA_SEGMENTS_SUFFIX,
    extract_features_from_sequenza,
    read_sequenza_results,
)


InputData = Union[pd.DataFrame, str, Path]
FileCollection = List[Path]

ASCAT_SUFFIX = ASCAT_SEGMENTS_SUFFIX
SEQUENZA_SUFFIX = SEQUENZA_SEGMENTS_SUFFIX
FACETS_SUFFIX = FACETS_SEGMENTS_SUFFIX


def discover_result_files(directory: Union[str, Path]) -> Tuple[str, FileCollection]:
    """Find one supported result-file family in a directory.

    A directory must contain files from exactly one family. This prevents silently
    combining incompatible copy-number formats in the same prediction input.
    """
    directory_path = Path(directory)

    # Ensure the directory exists and is a directory
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Expected an input directory: {directory_path}")
    
    files = [path for path in directory_path.iterdir() if path.is_file()]
    families = {
        "ascat": [path for path in files if path.name.lower().endswith(ASCAT_SUFFIX)],
        "sequenza": [path for path in files if path.name.lower().endswith(SEQUENZA_SUFFIX)],
        "facets": [path for path in files if path.name.lower().endswith(FACETS_SUFFIX)],
    }
    matches = [(name, matched_files) for name, matched_files in families.items() if matched_files]

    if not matches:
        raise FileNotFoundError(
            "No supported result files found. Expected files ending in "
            f"'{ASCAT_SUFFIX}', '{SEQUENZA_SUFFIX}', or '{FACETS_SUFFIX}'."
        )
    if len(matches) > 1:
        found = ", ".join(f"{name} ({len(paths)})" for name, paths in matches)
        raise ValueError(
            "Input directory contains multiple incompatible result types: "
            f"{found}. Provide a directory containing only one type."
        )

    result_type, result_files = matches[0]

    return result_type, sorted(result_files)


def load_input_data(input_data: InputData) -> Tuple[str, Union[pd.DataFrame, FileCollection]]:
    """Classify input and load a generic file, or collect specialized result files.

    Returns:
        A pair of ``(input_type, data)``. For ``dataframe`` and ``generic`` inputs,
        ``data`` is a DataFrame. For directory inputs, it is a sorted file list.
    """
    if isinstance(input_data, pd.DataFrame):
        return "dataframe", input_data.copy()

    if not isinstance(input_data, (str, Path)):
        raise TypeError("input_data must be a pandas DataFrame, file path, or directory path")

    input_path = Path(input_data)
    if input_path.is_file():
        return "generic", read_delimited_file(input_path)
    if input_path.is_dir():
        return discover_result_files(input_path)

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_preprocessing_workflow(
    input_data: InputData,
    gene_build: str = "hg38",
    extra_info: Optional[Union[pd.DataFrame, str, Path]] = None,
    overlap: float = 1.0,
) -> pd.DataFrame:
    """Prepare gene-level OTK input from a DataFrame, csv_file, or results directory."""
    input_type, loaded_data = load_input_data(input_data)

    if input_type in ("dataframe", "generic"):
        if extra_info is None:
            return loaded_data
        if isinstance(extra_info, (str, Path)):
            if not Path(extra_info).exists():
                raise FileNotFoundError(f"Extra info file path does not exist: {extra_info}")
            extra_info = pd.read_csv(extra_info, dtype={"sample": str})
        if not isinstance(extra_info, pd.DataFrame):
            raise TypeError("extra_info must be a pandas DataFrame, CSV path, or None")
        if any(column == "type" or column.startswith("type_") for column in extra_info.columns):
            raise ValueError(
                "extra_info cannot contain type or type_* columns when input is a DataFrame or file"
            )
        sample_features = loaded_data.loc[:, ["sample"]].drop_duplicates()
        return merge_sample_and_region_features(loaded_data, sample_features, extra_info)
    elif input_type == "ascat":
        copy_number_data = read_ascat_results(loaded_data)
        feature_tables = extract_features_from_ascat(copy_number_data)
    elif input_type == "sequenza":
        copy_number_data = read_sequenza_results(loaded_data)
        feature_tables = extract_features_from_sequenza(copy_number_data)
    elif input_type == "facets":
        copy_number_data = read_facets_results(loaded_data)
        feature_tables = extract_features_from_facets(copy_number_data)
    else:
        raise RuntimeError(f"Unsupported routed input type: {input_type}")

    # Gene annotation and sample-feature merging can be enabled after feature-table testing.
    return annotate_genes(
        feature_tables,
        gene_build=gene_build,
        extra_info=extra_info,
        overlap=overlap,
    )


