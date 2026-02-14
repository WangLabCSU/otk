#!/usr/bin/env python
"""
Unified Training Script for All ecDNA Models

This script is a wrapper around the otk CLI for backward compatibility.

Usage:
    python train_unified.py --model xgb_new --gpu 0
    python train_unified.py --model transformer --gpu 0
    python train_unified.py --all --gpu 0
    python train_unified.py --all --parallel --gpus 0,1,2,3  # Parallel training
    python train_unified.py --all --gpu -1  # CPU only
    
Or use the CLI directly:
    otk train --model xgb_new --gpu 0
    otk train --all --gpu 0
    otk predict --input data.csv --output predictions.csv --model xgb_new
"""

import sys
import os

if __name__ == "__main__":
    from otk.cli import cli
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append('--help')
    
    cli()
