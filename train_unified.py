#!/usr/bin/env python
"""
Unified Training Script for All ecDNA Models

Usage:
    python train_unified.py --model xgb_new
    python train_unified.py --model transformer
    python train_unified.py --all
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import logging
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


def train_xgb_model(model_type: str, output_dir: str):
    """Train XGB model"""
    logger.info(f"Training XGB model: {model_type}")
    
    train_df, val_df, test_df = load_split()
    
    if model_type == 'paper':
        model = XGB11Model()
    else:
        model = XGBNewModel()
    
    trainer = ModelTrainer(model, output_dir, f'xgb_{model_type}')
    results = trainer.train(train_df, val_df, test_df)
    
    return results


def train_neural_model(model_name: str):
    """Train Neural Network model"""
    logger.info(f"Training Neural model: {model_name}")
    
    train_df, val_df, test_df = load_split()
    
    model = create_neural_model(model_name)
    
    output_dir = f'otk_api/models/{model_name}'
    trainer = ModelTrainer(model, output_dir, model_name)
    
    results = trainer.train(train_df, val_df, test_df)
    
    return results


def train_tabpfn_model():
    """Train TabPFN model"""
    logger.info("Training TabPFN model")
    
    train_df, val_df, test_df = load_split()
    
    model = TabPFNModel(n_estimators=5, max_samples_per_estimator=5000)
    
    output_dir = 'otk_api/models/tabpfn'
    trainer = ModelTrainer(model, output_dir, 'tabpfn')
    
    results = trainer.train(train_df, val_df, test_df)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train ecDNA models')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--all', action='store_true', help='Train all models')
    args = parser.parse_args()
    
    # All 8 models
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
                    train_xgb_model(model_name, f'otk_api/models/xgb_{model_name}')
                elif model_type == 'tabpfn':
                    train_tabpfn_model()
                else:
                    train_neural_model(model_name)
            except Exception as e:
                logger.error(f"Failed to train {model_name or model_type}: {e}")
                import traceback
                traceback.print_exc()
    
    elif args.model:
        if args.model == 'tabpfn':
            train_tabpfn_model()
        elif args.model.startswith('xgb_'):
            model_type = args.model.split('_')[1]
            train_xgb_model(model_type, f'otk_api/models/{args.model}')
        else:
            train_neural_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
