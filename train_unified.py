#!/usr/bin/env python
"""
Unified Training Script for All ecDNA Models

Usage:
    python train_unified.py --model xgb_new --gpu 0
    python train_unified.py --model transformer --gpu 0
    python train_unified.py --all --gpu 0
    python train_unified.py --all --gpu -1  # CPU only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import logging
import torch
import yaml
from pathlib import Path

from otk.data.data_split import load_split
from otk.models.base_model import ModelTrainer
from otk.models.xgb11_model import XGB11Model, XGBNewModel
from otk.models.neural_models import create_neural_model
from otk.models.tabpfn_model import TabPFNModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 2026


def get_device(gpu: int) -> str:
    """Get device based on GPU parameter"""
    if gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{gpu}'
        logger.info(f"Using GPU: {gpu} ({torch.cuda.get_device_name(gpu)})")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    return device


def train_xgb_model(model_type: str, output_dir: str, device: str = 'cuda'):
    """Train XGB model"""
    logger.info(f"Training XGB model: {model_type} on {device}")
    
    train_df, val_df, test_df = load_split()
    
    if model_type == 'paper':
        model = XGB11Model()
    else:
        model = XGBNewModel()
    
    trainer = ModelTrainer(model, output_dir, f'xgb_{model_type}')
    results = trainer.train(train_df, val_df, test_df)
    
    return results


def train_neural_model(model_name: str, device: str = 'cuda'):
    """Train Neural Network model"""
    logger.info(f"Training Neural model: {model_name} on {device}")
    
    train_df, val_df, test_df = load_split()
    
    model = create_neural_model(model_name, device=device)
    
    output_dir = f'otk_api/models/{model_name}'
    trainer = ModelTrainer(model, output_dir, model_name)
    
    results = trainer.train(train_df, val_df, test_df)
    
    return results


def train_tabpfn_model(device: str = 'cuda'):
    """Train TabPFN model"""
    logger.info(f"Training TabPFN model on {device}")
    
    train_df, val_df, test_df = load_split()
    
    model = TabPFNModel(n_estimators=5, max_samples_per_estimator=5000, device=device)
    
    output_dir = 'otk_api/models/tabpfn'
    trainer = ModelTrainer(model, output_dir, 'tabpfn')
    
    results = trainer.train(train_df, val_df, test_df)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train ecDNA models')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID (default: 0). Use -1 for CPU.')
    args = parser.parse_args()
    
    device = get_device(args.gpu)
    
    all_models = [
        ('xgb', 'new'),
        ('xgb', 'paper'),
        ('nn', 'transformer'),
        ('nn', 'baseline_mlp'),
        ('nn', 'deep_residual'),
        ('nn', 'optimized_residual'),
        ('nn', 'dgit_super'),
        ('tabpfn', None),
    ]
    
    if args.all:
        for model_type, model_name in all_models:
            try:
                if model_type == 'xgb':
                    train_xgb_model(model_name, f'otk_api/models/xgb_{model_name}', device)
                elif model_type == 'tabpfn':
                    train_tabpfn_model(device)
                else:
                    train_neural_model(model_name, device)
            except Exception as e:
                logger.error(f"Failed to train {model_name or model_type}: {e}")
                import traceback
                traceback.print_exc()
    
    elif args.model:
        if args.model == 'tabpfn':
            train_tabpfn_model(device)
        elif args.model.startswith('xgb_'):
            model_type = args.model.split('_')[1]
            train_xgb_model(model_type, f'otk_api/models/{args.model}', device)
        else:
            train_neural_model(args.model, device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
