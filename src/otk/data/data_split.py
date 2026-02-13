#!/usr/bin/env python
"""
Unified Data Split Module

Provides consistent train/val/test splits across all models.

Split strategy:
- Ratio: 80% train, 10% val, 10% test
- Method: Sort samples by y_sum, then equidistant sampling
- Random seed: 2026 (fixed)
- Sample-level split to prevent data leakage

Usage:
    from otk.data.data_split import get_data_splits, load_split
    
    # Get sample IDs for each split
    train_samples, val_samples, test_samples = get_data_splits()
    
    # Or load directly
    train_df, val_df, test_df = load_split(data_path)
"""

import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Set, Dict
import logging

logger = logging.getLogger(__name__)

# Fixed parameters
RANDOM_SEED = 2026
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Default paths
DATA_DIR = Path(__file__).parent
SPLIT_FILE = DATA_DIR / 'split_2026.json'


def create_data_splits(
    data_path: Path = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
    save_path: Path = SPLIT_FILE
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Create train/val/test splits and save to file.
    
    Strategy:
    1. Sort samples by y_sum (number of positive genes per sample)
    2. Use equidistant sampling to ensure balanced distribution
    3. Split into train/val/test
    
    Args:
        data_path: Path to modeling data (csv.gz)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        save_path: Where to save the split
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples) as sets
    """
    if data_path is None:
        data_path = DATA_DIR / 'sorted_modeling_data.csv.gz'
    
    logger.info(f"Creating data splits from {data_path}")
    logger.info(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Random seed: {random_seed}")
    
    # Load data
    with gzip.open(data_path, 'rt') as f:
        df = pd.read_csv(f)
    
    # Get unique samples sorted by y_sum
    sample_y_sum = df.groupby('sample')['y'].sum().sort_values()
    sorted_samples = sample_y_sum.index.tolist()
    
    n_samples = len(sorted_samples)
    logger.info(f"Total samples: {n_samples}")
    
    # Calculate split sizes
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    test_size = n_samples - train_size - val_size
    
    logger.info(f"Split sizes: train={train_size}, val={val_size}, test={test_size}")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Equidistant sampling for train
    train_indices = np.linspace(0, n_samples - 1, train_size, dtype=int)
    train_samples = set([sorted_samples[i] for i in train_indices])
    
    # Remaining samples
    remaining_indices = list(set(range(n_samples)) - set(train_indices))
    remaining_indices.sort()
    
    # Equidistant sampling for val from remaining
    val_indices = np.linspace(0, len(remaining_indices) - 1, val_size, dtype=int)
    val_samples = set([sorted_samples[remaining_indices[i]] for i in val_indices])
    
    # Rest go to test
    test_indices = list(set(remaining_indices) - set([remaining_indices[i] for i in val_indices]))
    test_samples = set([sorted_samples[i] for i in test_indices])
    
    # Verify no overlap
    assert len(train_samples & val_samples) == 0, "Train and val overlap!"
    assert len(train_samples & test_samples) == 0, "Train and test overlap!"
    assert len(val_samples & test_samples) == 0, "Val and test overlap!"
    
    # Log statistics
    train_pos = df[df['sample'].isin(train_samples)]['y'].sum()
    val_pos = df[df['sample'].isin(val_samples)]['y'].sum()
    test_pos = df[df['sample'].isin(test_samples)]['y'].sum()
    
    logger.info(f"Train: {len(train_samples)} samples, {train_pos} positive genes")
    logger.info(f"Val: {len(val_samples)} samples, {val_pos} positive genes")
    logger.info(f"Test: {len(test_samples)} samples, {test_pos} positive genes")
    
    # Save to file
    split_data = {
        'random_seed': random_seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'train_samples': sorted(list(train_samples)),
        'val_samples': sorted(list(val_samples)),
        'test_samples': sorted(list(test_samples)),
        'statistics': {
            'train': {'n_samples': len(train_samples), 'n_positive_genes': int(train_pos)},
            'val': {'n_samples': len(val_samples), 'n_positive_genes': int(val_pos)},
            'test': {'n_samples': len(test_samples), 'n_positive_genes': int(test_pos)}
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    logger.info(f"Splits saved to {save_path}")
    
    return train_samples, val_samples, test_samples


def get_data_splits(split_file: Path = SPLIT_FILE) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Load train/val/test splits from file.
    If file doesn't exist, create it first.
    
    Args:
        split_file: Path to split JSON file
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples) as sets
    """
    if not split_file.exists():
        logger.info(f"Split file not found, creating...")
        return create_data_splits(save_path=split_file)
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    logger.info(f"Loaded splits from {split_file}")
    logger.info(f"Random seed used: {split_data.get('random_seed', 'unknown')}")
    
    train_samples = set(split_data['train_samples'])
    val_samples = set(split_data['val_samples'])
    test_samples = set(split_data['test_samples'])
    
    return train_samples, val_samples, test_samples


def load_split(
    data_path: Path = None,
    split_file: Path = SPLIT_FILE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data and split into train/val/test DataFrames.
    
    Args:
        data_path: Path to modeling data
        split_file: Path to split JSON file
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if data_path is None:
        data_path = DATA_DIR / 'sorted_modeling_data.csv.gz'
    
    # Get splits
    train_samples, val_samples, test_samples = get_data_splits(split_file)
    
    # Load data
    with gzip.open(data_path, 'rt') as f:
        df = pd.read_csv(f)
    
    # Split
    train_df = df[df['sample'].isin(train_samples)].copy()
    val_df = df[df['sample'].isin(val_samples)].copy()
    test_df = df[df['sample'].isin(test_samples)].copy()
    
    logger.info(f"Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def get_split_statistics(split_file: Path = SPLIT_FILE) -> Dict:
    """Get statistics about the split"""
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    return split_data.get('statistics', {})


if __name__ == "__main__":
    # Create splits when run directly
    logging.basicConfig(level=logging.INFO)
    
    print("Creating unified data splits...")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    
    train, val, test = create_data_splits()
    
    print(f"\nSplit created successfully!")
    print(f"Train: {len(train)} samples")
    print(f"Val: {len(val)} samples")
    print(f"Test: {len(test)} samples")
    print(f"\nSaved to: {SPLIT_FILE}")
