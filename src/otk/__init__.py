"""
OTK - ecDNA Analysis Toolkit

A comprehensive toolkit for extrachromosomal DNA (ecDNA) prediction
using deep learning and machine learning models.

Paper: Wang, S., et al. (2024). Machine learning-based extrachromosomal DNA
identification in large-scale cohorts reveals its clinical implications in cancer.
Nature Communications.

Example usage:
    >>> from otk.models import XGBNewModel, create_neural_model
    >>> from otk.data import load_split
    >>> from otk.predict import UnifiedPredictor

    >>> # Load pre-split data
    >>> train_df, val_df, test_df = load_split()

    >>> # Train model
    >>> model = XGBNewModel()
    >>> model.fit(train_df.drop('y', axis=1), train_df['y'],
    ...           val_df.drop('y', axis=1), val_df['y'])

    >>> # Predict
    >>> predictor = UnifiedPredictor('otk_api/models/xgb_new/best_model.pkl')
    >>> results = predictor.run(test_data)

CLI usage:
    $ otk train --model xgb_new --gpu 0
    $ otk predict --input data.csv --output results.csv --model xgb_new
    $ otk models  # List available models
    $ otk api --port 8000  # Start prediction API server
"""

__version__ = "1.0.2"
__author__ = "Shixiang Wang"
__email__ = "wangshx@csu.edu.cn"

import importlib

from .data import get_data_splits, load_split
from .models import BaseEcDNAModel, ModelTrainer
from .predict import UnifiedPredictor, predict
from .utils import RANDOM_SEED, set_random_seed

# Model classes are resolved lazily (PEP 562). On macOS both torch and xgboost
# bundle their own OpenMP (libomp) runtime, and loading two OpenMP runtimes into
# one process corrupts memory and segfaults torch model construction. Deferring
# these re-exports means importing ``otk`` never loads a modelling backend, and
# each workflow only loads the backend it actually uses.
_LAZY_MODEL_ATTRS = {
    "XGB11Model",
    "XGBNewModel",
    "BaselineMLPModel",
    "TransformerEcDNAModel",
    "DeepResidualModel",
    "OptimizedResidualModel",
    "DGITSuperModel",
    "TabPFNModel",
    "create_neural_model",
}


def __getattr__(name):
    if name in _LAZY_MODEL_ATTRS:
        value = getattr(importlib.import_module(".models", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Data
    "load_split",
    "get_data_splits",
    # Models
    "BaseEcDNAModel",
    "ModelTrainer",
    "XGB11Model",
    "XGBNewModel",
    "BaselineMLPModel",
    "TransformerEcDNAModel",
    "DeepResidualModel",
    "OptimizedResidualModel",
    "DGITSuperModel",
    "TabPFNModel",
    "create_neural_model",
    # Predict
    "UnifiedPredictor",
    "predict",
    # Utils
    "set_random_seed",
    "RANDOM_SEED",
]
