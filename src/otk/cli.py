#!/usr/bin/env python
"""
OTK CLI - Command Line Interface for ecDNA analysis

Usage:
    otk train --model xgb_new --gpu 0
    otk train --all --gpu 0
    otk predict --input data.csv --output predictions.csv --model xgb_new
    otk predict --input data.csv --output results/ --model transformer
"""

import click
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch

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


@click.group()
@click.version_option(version='0.1.0', prog_name='otk')
def cli():
    """OTK - ecDNA Analysis Toolkit
    
    A comprehensive toolkit for extrachromosomal DNA (ecDNA) prediction
    using deep learning and machine learning models.
    """
    pass


@cli.command()
@click.option('--model', '-m', type=str, default=None,
              help='Model name to train (e.g., xgb_new, transformer, baseline_mlp)')
@click.option('--all', '-a', 'train_all', is_flag=True,
              help='Train all available models')
@click.option('--parallel', '-p', is_flag=True,
              help='Enable parallel training (use with --all)')
@click.option('--gpu', '-g', type=int, default=0,
              help='GPU device ID (default: 0). Use -1 for CPU.')
@click.option('--gpus', type=str, default='0',
              help='Comma-separated GPU IDs for parallel training (e.g., "0,1,2,3")')
@click.option('--log-dir', '-l', type=str, default='logs/training',
              help='Directory for training logs')
@click.option('--output', '-o', type=str, default='otk_api/models',
              help='Output directory for trained models')
def train(model: Optional[str], train_all: bool, parallel: bool, 
          gpu: int, gpus: str, log_dir: str, output: str):
    """Train ecDNA prediction models.
    
    Examples:
        otk train --model xgb_new --gpu 0
        otk train --model transformer --gpu 0
        otk train --all --gpu 0
        otk train --all --parallel --gpus 0,1,2,3
        otk train --all --gpu -1  # CPU only
    """
    from otk.data.data_split import load_split
    from otk.models.base_model import ModelTrainer
    from otk.models.xgb11_model import XGB11Model, XGBNewModel
    from otk.models.neural_models import create_neural_model
    from otk.models.tabpfn_model import TabPFNModel
    import multiprocessing as mp
    import time
    
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
    
    def train_single_model(model_info: dict) -> dict:
        """Train a single model (used for parallel execution)"""
        model_type = model_info['type']
        model_name = model_info['name']
        device = model_info['device']
        output_dir = model_info['output_dir']
        
        logger.info(f"Starting training: {model_name} on {device}")
        start_time = time.time()
        
        try:
            train_df, val_df, test_df = load_split()
            
            if model_type == 'xgb':
                if model_name == 'xgb_paper':
                    mdl = XGB11Model()
                else:
                    mdl = XGBNewModel()
            elif model_type == 'tabpfn':
                mdl = TabPFNModel(n_estimators=5, max_samples_per_estimator=5000, device=device)
            else:
                mdl = create_neural_model(model_name, device=device)
            
            trainer = ModelTrainer(mdl, output_dir, model_name)
            results = trainer.train(train_df, val_df, test_df)
            
            elapsed = time.time() - start_time
            logger.info(f"Completed {model_name} in {elapsed:.1f}s")
            
            return {'model': model_name, 'status': 'success', 'results': results, 'time': elapsed}
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return {'model': model_name, 'status': 'failed', 'error': str(e), 'time': elapsed}
    
    def train_xgb_model(model_type: str, output_dir: str, device: str):
        """Train XGB model"""
        logger.info(f"Training XGB model: {model_type} on {device}")
        train_df, val_df, test_df = load_split()
        
        if model_type == 'paper':
            mdl = XGB11Model()
        else:
            mdl = XGBNewModel()
        
        trainer = ModelTrainer(mdl, output_dir, f'xgb_{model_type}')
        return trainer.train(train_df, val_df, test_df)
    
    def train_neural_model(model_name: str, device: str):
        """Train Neural Network model"""
        logger.info(f"Training Neural model: {model_name} on {device}")
        train_df, val_df, test_df = load_split()
        
        mdl = create_neural_model(model_name, device=device)
        trainer = ModelTrainer(mdl, f'{output}/{model_name}', model_name)
        return trainer.train(train_df, val_df, test_df)
    
    def train_tabpfn_model(device: str):
        """Train TabPFN model"""
        logger.info(f"Training TabPFN model on {device}")
        train_df, val_df, test_df = load_split()
        
        mdl = TabPFNModel(n_estimators=5, max_samples_per_estimator=5000, device=device)
        trainer = ModelTrainer(mdl, f'{output}/tabpfn', 'tabpfn')
        return trainer.train(train_df, val_df, test_df)
    
    def run_parallel_training(models, gpu_list, log_directory):
        """Run parallel training on multiple GPUs"""
        logger.info(f"Starting parallel training on GPUs: {gpu_list}")
        
        Path(log_directory).mkdir(parents=True, exist_ok=True)
        
        model_infos = []
        for i, (mtype, mname) in enumerate(models):
            gpu_id = gpu_list[i % len(gpu_list)]
            device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu'
            
            if mtype == 'xgb':
                out_dir = f'{output}/xgb_{mname}'
                name = f'xgb_{mname}'
            elif mtype == 'tabpfn':
                out_dir = f'{output}/tabpfn'
                name = 'tabpfn'
            else:
                out_dir = f'{output}/{mname}'
                name = mname
            
            model_infos.append({
                'type': mtype,
                'name': name,
                'device': device,
                'output_dir': out_dir,
            })
        
        n_processes = min(len(models), len(gpu_list) * 2)
        logger.info(f"Running {len(models)} models with {n_processes} parallel processes")
        
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(train_single_model, model_infos)
        
        logger.info("\n" + "="*60)
        logger.info("PARALLEL TRAINING SUMMARY")
        logger.info("="*60)
        
        for r in results:
            if r['status'] == 'success':
                logger.info(f"  {r['model']}: SUCCESS in {r['time']:.1f}s")
            else:
                logger.info(f"  {r['model']}: FAILED - {r.get('error', 'Unknown error')}")
        
        return results
    
    # Execute training
    if train_all and parallel:
        gpu_list = [int(g) for g in gpus.split(',')]
        run_parallel_training(all_models, gpu_list, log_dir)
    
    elif train_all:
        device = get_device(gpu)
        for mtype, mname in all_models:
            try:
                if mtype == 'xgb':
                    train_xgb_model(mname, f'{output}/xgb_{mname}', device)
                elif mtype == 'tabpfn':
                    train_tabpfn_model(device)
                else:
                    train_neural_model(mname, device)
            except Exception as e:
                logger.error(f"Failed to train {mname or mtype}: {e}")
                import traceback
                traceback.print_exc()
    
    elif model:
        device = get_device(gpu)
        if model == 'tabpfn':
            train_tabpfn_model(device)
        elif model.startswith('xgb_'):
            model_type = model.split('_')[1]
            train_xgb_model(model_type, f'{output}/{model}', device)
        else:
            train_neural_model(model, device)
    else:
        click.echo("Please specify --model or --all. Use --help for more information.")
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', 'input_path', type=str, required=True,
              help='Input CSV file path')
@click.option('--output', '-o', 'output_path', type=str, required=True,
              help='Output file or directory path')
@click.option('--model', '-m', type=str, default='xgb_new',
              help='Model name (default: xgb_new)')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU device ID (default: -1 for CPU)')
@click.option('--threshold', '-t', type=float, default=None,
              help='Prediction threshold (default: use model optimal threshold)')
def predict(input_path: str, output_path: str, model: str, 
            gpu: int, threshold: Optional[float]):
    """Run prediction using trained model.
    
    Examples:
        otk predict --input data.csv --output predictions.csv --model xgb_new
        otk predict -i data.csv -o results/ -m transformer --gpu 0
    """
    from otk.predict.predictor import UnifiedPredictor
    import pandas as pd
    
    input_file = Path(input_path)
    if not input_file.exists():
        click.echo(f"Error: Input file not found: {input_path}", err=True)
        sys.exit(1)
    
    # Find model path
    model_dir = Path(__file__).parent.parent.parent / 'otk_api' / 'models' / model
    if not model_dir.exists():
        model_dir = Path(output_path).parent / 'models' / model
    
    model_file = model_dir / 'best_model.pkl'
    if not model_file.exists():
        model_file = model_dir / 'best_model.pth'
    
    if not model_file.exists():
        click.echo(f"Error: Model not found: {model}", err=True)
        click.echo(f"Looked in: {model_dir}", err=True)
        sys.exit(1)
    
    logger.info(f"Loading model from: {model_file}")
    predictor = UnifiedPredictor(str(model_file), gpu=gpu)
    
    if threshold is not None:
        predictor.optimal_threshold = threshold
    
    # Run prediction
    output_file = Path(output_path)
    if output_file.suffix == '':
        output_file = output_file / 'predictions.csv'
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running prediction on: {input_file}")
    results = predictor.run(str(input_file), str(output_file))
    
    logger.info(f"Predictions saved to: {output_file}")
    
    # Print summary
    n_total = len(results)
    n_positive = results['prediction'].sum()
    click.echo(f"\nPrediction Summary:")
    click.echo(f"  Total samples: {n_total}")
    click.echo(f"  Positive predictions: {n_positive} ({100*n_positive/n_total:.1f}%)")
    click.echo(f"  Output: {output_file}")


@cli.command()
def models():
    """List available models."""
    model_dir = Path(__file__).parent.parent.parent / 'otk_api' / 'models'
    
    if not model_dir.exists():
        click.echo("No models directory found.")
        return
    
    click.echo("\nAvailable Models:")
    click.echo("="*60)
    
    for model_path in sorted(model_dir.iterdir()):
        if model_path.is_dir():
            config_file = model_path / 'config.yml'
            summary_file = model_path / 'training_summary.yml'
            model_file = model_path / 'best_model.pkl'
            
            if not model_file.exists():
                model_file = model_path / 'best_model.pth'
            
            status = "✓" if model_file.exists() else "✗"
            name = model_path.name
            
            if summary_file.exists():
                import yaml
                with open(summary_file) as f:
                    summary = yaml.safe_load(f)
                test_auprc = summary.get('gene_level', {}).get('test', {}).get('auPRC', None)
                if test_auprc is not None:
                    click.echo(f"  [{status}] {name:<25} Test auPRC: {test_auprc:.4f}")
                else:
                    click.echo(f"  [{status}] {name:<25} (training incomplete)")
            else:
                click.echo(f"  [{status}] {name:<25} (not trained)")


@cli.command()
@click.option('--model', '-m', type=str, required=True,
              help='Model name to analyze')
def analyze(model: str):
    """Analyze a trained model's performance."""
    import yaml
    
    model_dir = Path(__file__).parent.parent.parent / 'otk_api' / 'models' / model
    
    if not model_dir.exists():
        click.echo(f"Error: Model not found: {model}", err=True)
        sys.exit(1)
    
    summary_file = model_dir / 'training_summary.yml'
    if not summary_file.exists():
        click.echo(f"Error: No training summary found for model: {model}", err=True)
        sys.exit(1)
    
    with open(summary_file) as f:
        summary = yaml.safe_load(f)
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Model Analysis: {model}")
    click.echo(f"{'='*60}")
    
    # Gene-level metrics
    gene_test = summary.get('gene_level', {}).get('test', {})
    if gene_test:
        click.echo(f"\nGene-Level Test Metrics:")
        for metric in ['auPRC', 'AUC', 'Precision', 'Recall', 'F1']:
            val = gene_test.get(metric)
            if val is not None:
                click.echo(f"  {metric}: {val:.4f}")
    
    gene_val = summary.get('gene_level', {}).get('val', {})
    if gene_val:
        click.echo(f"\nGene-Level Validation Metrics:")
        for metric in ['auPRC', 'AUC', 'Precision', 'Recall', 'F1']:
            val = gene_val.get(metric)
            if val is not None:
                click.echo(f"  {metric}: {val:.4f}")
    
    # Sample-level metrics
    sample_test = summary.get('sample_level', {}).get('test', {})
    if sample_test:
        click.echo(f"\nSample-Level Test Metrics:")
        for metric in ['auPRC', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1']:
            val = sample_test.get(metric)
            if val is not None:
                click.echo(f"  {metric}: {val:.4f}")
        click.echo(f"  Total Samples: {sample_test.get('total_samples', 'N/A')}")
        click.echo(f"  Positive Samples: {sample_test.get('positive_samples', 'N/A')}")
        click.echo(f"  Predicted Positive: {sample_test.get('predicted_positive', 'N/A')}")


if __name__ == '__main__':
    cli()
