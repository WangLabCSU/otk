"""Preprocessing-aware OTK prediction entry point."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from otk_preprocess.workflow import run_preprocessing_workflow


def predict_input(
    input_data: Union[pd.DataFrame, str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model: str = "xgb_new",
    gene_build: str = "hg38",
    extra_info: Optional[Union[pd.DataFrame, str, Path]] = None,
    overlap: float = 1.0,
    gpu: int = -1,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Preprocess an input DataFrame, CSV, or result directory and predict.

    ``extra_info`` can be a DataFrame or CSV path. ``output_path`` is optional;
    a directory receives a file named ``predictions.csv``.
    """
    from otk.cli import get_models_dir
    from otk.predict.predictor import UnifiedPredictor

    model_dir = get_models_dir() / model
    if not model_dir.exists() and output_path is not None:
        model_dir = Path(output_path).parent / "models" / model
    model_file = next(
        (model_dir / name for name in ("best_model.pkl", "best_model.pth")
         if (model_dir / name).is_file()),
        None,
    )
    if model_file is None:
        raise FileNotFoundError(f"Model '{model}' not found in {model_dir}")

    data = run_preprocessing_workflow(input_data, gene_build, extra_info, overlap)
    if data.empty:
        raise ValueError("Preprocessing produced no gene-level records for prediction")
    predictor = UnifiedPredictor(str(model_file), gpu=gpu)
    if threshold is not None:
        predictor.optimal_threshold = threshold

    output_file = Path(output_path) if output_path is not None else None
    if output_file is not None and not output_file.suffix:
        output_file = output_file / "predictions.csv"
    return predictor.run(data, output_file)