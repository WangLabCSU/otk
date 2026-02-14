"""
OTK Data Module

Unified data splitting with seed=2026 for reproducibility.
"""

from .data_split import load_split, get_data_splits

__all__ = ['load_split', 'get_data_splits']
