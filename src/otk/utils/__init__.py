"""
OTK Utility Module

Common utilities for ecDNA analysis toolkit.
"""

import random
import numpy as np

RANDOM_SEED = 2026

def set_random_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

__all__ = ['set_random_seed', 'RANDOM_SEED']