"""Reader for a single laboratory CSV or TSV file."""

from pathlib import Path
from typing import Union

import pandas as pd


DELIMITED_SUFFIXES = {".csv": ",", ".tsv": "\t"}


def read_delimited_file(file_path: Union[str, Path]) -> pd.DataFrame:
	"""Read exactly one CSV or TSV file into a DataFrame.

	The workflow deliberately does not accept a directory here. Directory input is
	reserved for ASCAT, Sequenza, and FACETS result collections.
	"""
	path = Path(file_path)

	"""
	if not path.is_file():
		raise FileNotFoundError(f"CSV/TSV input must be an existing file: {path}")
	"""
	
	separator = DELIMITED_SUFFIXES.get(path.suffix.lower())
	if separator is None:
		supported = ", ".join(DELIMITED_SUFFIXES)
		raise ValueError(f"Unsupported delimited file type '{path.suffix}'. Expected: {supported}")

	return pd.read_csv(path, sep=separator)