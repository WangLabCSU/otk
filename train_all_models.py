#!/usr/bin/env python
"""
Parallel Training Script for All ecDNA Models

Trains all models using unified data split and interface.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    'xgb11_paper': {
        'script': 'train_xgb.py',
        'model_type': 'paper',
        'output_dir': 'otk_api/models/xgb11_paper',
        'priority': 2
    },
    'xgb_new': {
        'script': 'train_xgb.py',
        'model_type': 'new',
        'output_dir': 'otk_api/models/xgb_new',
        'priority': 1  # Highest priority
    },
    'baseline_mlp': {
        'script': 'train_nn.py',
        'model_name': 'baseline_mlp',
        'priority': 3
    },
    'deep_residual': {
        'script': 'train_nn.py',
        'model_name': 'deep_residual',
        'priority': 3
    },
    'transformer': {
        'script': 'train_nn.py',
        'model_name': 'transformer',
        'priority': 2
    },
    'optimized_residual': {
        'script': 'train_nn.py',
        'model_name': 'optimized_residual',
        'priority': 4
    },
    'dgit_super': {
        'script': 'train_nn.py',
        'model_name': 'dgit_super',
        'priority': 5
    }
}


def train_model(model_name, config):
    """Train a single model"""
    logger.info(f"Starting training for {model_name}")
    
    script = config['script']
    
    if script == 'train_xgb.py':
        cmd = [
            'python', script,
            '--model-type', config['model_type'],
            '--output-dir', config['output_dir']
        ]
    else:
        cmd = [
            'python', script,
            '--model-name', config['model_name']
        ]
    
    log_file = f"train_{model_name}.log"
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd='/data/home/wsx/Projects/otk/otk'
            )
            return process, log_file
    except Exception as e:
        logger.error(f"Failed to start {model_name}: {e}")
        return None, None


def check_model_status(model_name, output_dir):
    """Check if model training completed successfully"""
    summary_file = Path(output_dir) / 'training_summary.yml'
    
    if not summary_file.exists():
        return 'running' if is_process_running(model_name) else 'failed'
    
    try:
        import yaml
        with open(summary_file, 'r') as f:
            summary = yaml.safe_load(f)
        
        # Check if metrics are present
        if 'gene_level' in summary and 'sample_level' in summary:
            return 'completed'
        return 'incomplete'
    except:
        return 'corrupted'


def is_process_running(model_name):
    """Check if training process is still running"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', f'train.*{model_name}'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


def monitor_training(processes, max_wait_hours=24):
    """Monitor all training processes"""
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600
    
    while processes and (time.time() - start_time) < max_wait_seconds:
        completed = []
        
        for model_name, (process, log_file) in processes.items():
            if process.poll() is not None:
                # Process finished
                if process.returncode == 0:
                    logger.info(f"{model_name} completed successfully")
                else:
                    logger.error(f"{model_name} failed with code {process.returncode}")
                completed.append(model_name)
        
        for model_name in completed:
            del processes[model_name]
        
        if processes:
            time.sleep(60)  # Check every minute
    
    # Kill remaining processes if timeout
    if processes:
        logger.warning("Timeout reached, killing remaining processes")
        for model_name, (process, _) in processes.items():
            process.terminate()


def main():
    """Main training orchestrator"""
    logger.info("="*60)
    logger.info("Starting Parallel Model Training")
    logger.info("="*60)
    
    # Sort models by priority
    sorted_models = sorted(MODELS.items(), key=lambda x: x[1]['priority'])
    
    # Group by priority
    priority_groups = {}
    for model_name, config in sorted_models:
        p = config['priority']
        if p not in priority_groups:
            priority_groups[p] = []
        priority_groups[p].append((model_name, config))
    
    # Train in priority groups
    for priority in sorted(priority_groups.keys()):
        group = priority_groups[priority]
        logger.info(f"\nTraining priority {priority} group: {[m[0] for m in group]}")
        
        processes = {}
        for model_name, config in group:
            process, log_file = train_model(model_name, config)
            if process:
                processes[model_name] = (process, log_file)
        
        if processes:
            monitor_training(processes)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    
    for model_name, config in MODELS.items():
        output_dir = config.get('output_dir', f"otk_api/models/{model_name}")
        status = check_model_status(model_name, output_dir)
        logger.info(f"{model_name}: {status}")


if __name__ == "__main__":
    main()
