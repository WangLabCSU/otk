"""
OTK Models Module

All models inherit from BaseEcDNAModel for unified interface.
"""

import importlib

from .base_model import BaseEcDNAModel, ModelTrainer

# Model implementations are imported lazily (PEP 562). On macOS, torch and
# xgboost each bundle their own OpenMP (libomp) runtime; loading two OpenMP
# runtimes into one process and running heavy compute in both corrupts memory
# and segfaults. Deferring these imports means a given run only loads the
# backend it needs: the torch-backed neural models never load xgboost, and the
# xgboost-backed models never load torch.
_LAZY_MODULES = {
    "XGB11Model": ".xgb11_model",
    "XGBNewModel": ".xgb11_model",
    "BaselineMLPModel": ".neural_models",
    "TransformerEcDNAModel": ".neural_models",
    "DeepResidualModel": ".neural_models",
    "OptimizedResidualModel": ".neural_models",
    "DGITSuperModel": ".neural_models",
    "create_neural_model": ".neural_models",
    "TabPFNModel": ".tabpfn_model",
}


def __getattr__(name):
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    'BaseEcDNAModel',
    'ModelTrainer',
    'XGB11Model',
    'XGBNewModel',
    'BaselineMLPModel',
    'TransformerEcDNAModel',
    'DeepResidualModel',
    'OptimizedResidualModel',
    'DGITSuperModel',
    'TabPFNModel',
    'create_neural_model',
]
