"""
OTK Models Module

All models inherit from BaseEcDNAModel for unified interface.
"""

from .base_model import BaseEcDNAModel, ModelTrainer
from .xgb11_model import XGB11Model, XGBNewModel
from .neural_models import (
    BaselineMLPModel,
    TransformerEcDNAModel,
    DeepResidualModel,
    OptimizedResidualModel,
    DGITSuperModel,
    create_neural_model
)
from .tabpfn_model import TabPFNModel

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
