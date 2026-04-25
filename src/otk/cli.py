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


# Config command group
@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command('generate')
@click.option('--model', '-m', type=str, default=None,
              help='Model name to generate config for (e.g., xgb_new, transformer)')
@click.option('--all', '-a', 'generate_all', is_flag=True,
              help='Generate configs for all available models')
@click.option('--output', '-o', type=str, default='otk_api/models',
              help='Output directory for generated configs')
def config_generate(model: Optional[str], generate_all: bool, output: str):
    """Generate config.yml files for models.
    
    Examples:
        otk config generate --model xgb_new
        otk config generate --all
        otk config generate --model transformer --output custom/models
    """
    from otk.models.config_generator import save_config, MODEL_CONFIGS
    
    output_dir = Path(output)
    
    if generate_all:
        all_models = list(MODEL_CONFIGS.keys())
        click.echo(f"Generating configs for {len(all_models)} models...")
        click.echo("="*60)
        
        success_count = 0
        for model_name in all_models:
            try:
                model_output_dir = output_dir / model_name
                save_config(model_name, model_output_dir)
                success_count += 1
                click.echo(f"  ✓ {model_name}")
            except Exception as e:
                click.echo(f"  ✗ {model_name}: {e}", err=True)
        
        click.echo("="*60)
        click.echo(f"Generated {success_count}/{len(all_models)} configs successfully!")
        click.echo(f"Output directory: {output_dir.absolute()}")
    
    elif model:
        if model not in MODEL_CONFIGS:
            available = ', '.join(MODEL_CONFIGS.keys())
            click.echo(f"Error: Unknown model '{model}'", err=True)
            click.echo(f"Available models: {available}", err=True)
            sys.exit(1)
        
        try:
            model_output_dir = output_dir / model
            save_config(model, model_output_dir)
            click.echo(f"✓ Config generated for '{model}'")
            click.echo(f"  Location: {model_output_dir / 'config.yml'}")
        except Exception as e:
            click.echo(f"Error generating config: {e}", err=True)
            sys.exit(1)
    
    else:
        click.echo("Please specify --model or --all. Use --help for more information.")
        sys.exit(1)


@config.command('list')
def config_list():
    """List all available model configurations."""
    from otk.models.config_generator import MODEL_CONFIGS

    click.echo("\nAvailable Model Configurations:")
    click.echo("="*60)

    for model_name, config in MODEL_CONFIGS.items():
        model_type = config.get('model', {}).get('type', 'Unknown')
        variant = config.get('model', {}).get('variant', 'Unknown')
        click.echo(f"  {model_name:<20} [{model_type}] {variant}")

    click.echo("="*60)
    click.echo(f"Total: {len(MODEL_CONFIGS)} models")
    click.echo("\nUse 'otk config generate --model <name>' to generate a config file.")


@cli.command()
@click.option('--model', '-m', type=str, default=None,
              help='Model name to download (e.g., tabpfn)')
@click.option('--force', '-f', is_flag=True,
              help='Force re-download even if file exists')
@click.option('--list', '-l', 'list_models', is_flag=True,
              help='List all large models that can be downloaded')
@click.option('--info', '-i', is_flag=True,
              help='Show download info for a model')
def download(model: Optional[str], force: bool, list_models: bool, info: bool):
    """Download large model files from GitHub Release.

    Large models (~275MB) are hosted on GitHub Release.
    Chinese mirrors are supported for faster download.

    Examples:
        otk download --list
        otk download --model tabpfn
        otk download --model tabpfn --info
        otk download --model tabpfn --force
    """
    # Import downloader
    download_dir = Path(__file__).parent.parent.parent / 'otk_api' / 'download'
    sys.path.insert(0, str(download_dir))

    try:
        from model_downloader import (
            download_model, get_download_info, list_large_models,
            LARGE_MODELS, DOWNLOAD_MIRRORS, GITHUB_REPO, GITHUB_RELEASE_TAG
        )
    except ImportError:
        click.echo("Error: model_downloader module not found", err=True)
        click.echo(f"Expected at: {download_dir}", err=True)
        sys.exit(1)

    if list_models:
        click.echo("\nLarge Models (require download from GitHub Release):")
        click.echo("="*60)

        for name, config in LARGE_MODELS.items():
            size_mb = config['size'] // 1024 // 1024
            click.echo(f"  {name:<15} ~{size_mb}MB  {config['description']}")

        click.echo("="*60)
        click.echo(f"GitHub Release: {GITHUB_REPO}/{GITHUB_RELEASE_TAG}")
        click.echo("\nChinese mirrors available:")
        for mirror in DOWNLOAD_MIRRORS[1:]:
            click.echo(f"  - {mirror}")
        click.echo("\nUse: otk download --model <name>")
        return

    if info and model:
        if model not in LARGE_MODELS:
            click.echo(f"Model '{model}' is not a large model requiring download")
            return

        info_text = get_download_info(model)
        click.echo(f"\n{info_text}")
        return

    if model:
        if model not in LARGE_MODELS:
            click.echo(f"Error: '{model}' is not a large model requiring download", err=True)
            click.echo(f"Available large models: {', '.join(LARGE_MODELS.keys())}", err=True)
            sys.exit(1)

        config = LARGE_MODELS[model]
        size_mb = config['size'] // 1024 // 1024

        click.echo(f"\nDownloading {model} (~{size_mb}MB)")
        click.echo("="*60)

        try:
            model_dir = Path(__file__).parent.parent.parent / 'otk_api' / 'models'
            downloaded_path = download_model(model, base_dir=model_dir, force=force)
            click.echo(f"\n✓ Download completed!")
            click.echo(f"  Model: {model}")
            click.echo(f"  Path: {downloaded_path}")

        except RuntimeError as e:
            click.echo(f"\n✗ Download failed: {e}", err=True)
            click.echo("\nManual download options:", err=True)
            click.echo(f"  1. Visit: https://github.com/{GITHUB_REPO}/releases/tag/{GITHUB_RELEASE_TAG}", err=True)
            click.echo(f"  2. Place file at: otk_api/models/{model}/best_model.pkl", err=True)
            sys.exit(1)

        return

    # No options specified
    click.echo("Please specify an option:")
    click.echo("  --list     List all large models")
    click.echo("  --model    Download a specific model")
    click.echo("  --info     Show download info")
    click.echo("\nExample: otk download --model tabpfn")


if __name__ == '__main__':
    cli()
