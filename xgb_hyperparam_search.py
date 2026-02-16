#!/usr/bin/env python
"""
XGBoost Hyperparameter Search Script

Uses Optuna for hyperparameter optimization with auPRC as the objective.

Usage:
    python xgb_hyperparam_search.py --n-trials 100 --gpu 0
    python xgb_hyperparam_search.py --n-trials 50 --study-name xgb_search_v1
    python xgb_hyperparam_search.py --resume --study-name xgb_search_v1
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from otk.data.data_split import load_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 2026


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for XGBNewModel"""
    feature_df = pd.DataFrame()
    
    for f in ['segVal', 'minor_cn', 'purity', 'ploidy', 'pLOH', 'AScore', 'cna_burden']:
        feature_df[f] = df[f].fillna(0) if f in df.columns else 0
    
    for f in ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']:
        feature_df[f] = df[f].fillna(0) if f in df.columns else 0
    
    for i in range(1, 20):
        f = f'CN{i}'
        feature_df[f] = df[f].fillna(0) if f in df.columns else 0
    
    if 'age' in df.columns:
        feature_df['age'] = df['age'].fillna(df['age'].mean())
    if 'gender' in df.columns:
        feature_df['gender'] = df['gender'].fillna(0)
    
    for c in [col for col in df.columns if col.startswith('type_')]:
        feature_df[c] = df[c].fillna(0)
    
    if 'segVal' in df.columns and 'ploidy' in df.columns:
        feature_df['cn_imbalance'] = df['segVal'] / (df['ploidy'] + 1e-6)
    if 'minor_cn' in df.columns and 'segVal' in df.columns:
        feature_df['allele_imbalance'] = df['minor_cn'] / (df['segVal'] + 1e-6)
    if 'cna_burden' in df.columns and 'purity' in df.columns:
        feature_df['cna_burden_adj'] = df['cna_burden'] * df['purity']
    if 'AScore' in df.columns and 'purity' in df.columns:
        feature_df['ascore_adj'] = df['AScore'] * df['purity']
    for f in ['freq_Circular', 'freq_BFB', 'freq_HR']:
        if f in df.columns:
            feature_df[f'has_{f.split("_")[1].lower()}'] = (df[f] > 0).astype(int)
    
    freq_cols = ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']
    if all(c in df.columns for c in freq_cols):
        feature_df['amplicon_type_count'] = (df[freq_cols] > 0).sum(axis=1)
    
    cn_cols = [f'CN{i}' for i in range(1, 20)]
    existing_cn = [c for c in cn_cols if c in df.columns]
    if existing_cn:
        feature_df['cn_sig_diversity'] = (df[existing_cn] > 0).sum(axis=1)
        feature_df['max_cn_sig'] = df[existing_cn].max(axis=1)
    
    if 'purity' in df.columns and 'ploidy' in df.columns:
        feature_df['purity_x_ploidy'] = df['purity'] * df['ploidy']
    if 'pLOH' in df.columns:
        feature_df['has_loh'] = (df['pLOH'] > 0.1).astype(int)
    
    return feature_df


def calculate_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate auPRC"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def get_param_space(trial) -> Dict[str, Any]:
    """Define hyperparameter search space
    
    Optimized for highly imbalanced data (positive ratio ~0.35%, scale_pos_weight ~270)
    """
    params = {
        'eta': trial.suggest_float('eta', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.9),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'alpha': trial.suggest_float('alpha', 0, 5),
        'lambda': trial.suggest_float('lambda', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 50, 400),
        
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'random_state': RANDOM_SEED,
    }
    return params


def train_and_evaluate(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    device: str = 'cuda:0'
) -> tuple:
    """Train model and return validation auPRC"""
    params = params.copy()
    params['device'] = device
    params['tree_method'] = 'hist'
    params['nthread'] = 1
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    y_prob = model.predict(dval)
    auprc = calculate_auprc(y_val.values, y_prob)
    
    precision, recall, thresholds = precision_recall_curve(y_val.values, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return auprc, model, optimal_threshold, model.best_iteration


def objective(trial, X_train, y_train, X_val, y_val, device: str) -> float:
    """Optuna objective function with cross-validation to reduce overfitting"""
    from sklearn.model_selection import StratifiedKFold
    
    params = get_param_space(trial)
    
    try:
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        auprc_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
            X_tr, X_vl = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
            y_tr, y_vl = y_combined.iloc[train_idx], y_combined.iloc[val_idx]
            
            auprc, _, _, _ = train_and_evaluate(
                params, X_tr, y_tr, X_vl, y_vl, device
            )
            auprc_scores.append(auprc)
        
        return np.mean(auprc_scores)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_hyperparameter_search(
    n_trials: int = 100,
    study_name: str = "xgb_hyperparam_search",
    storage_path: Optional[str] = None,
    resume: bool = False,
    device: str = 'cuda:0',
    output_dir: str = 'otk_api/models/xgb_tuned',
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Run hyperparameter search using Optuna
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the study
        storage_path: Path to SQLite storage for persistence
        resume: Whether to resume existing study
        device: Device to use for training
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs (default: 1)
    """
    import optuna
    from optuna.samplers import TPESampler
    
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    logger.info("Loading data...")
    train_df, val_df, test_df = load_split()
    
    logger.info("Preparing features...")
    X_train = prepare_features(train_df)
    y_train = train_df['y']
    X_val = prepare_features(val_df)
    y_val = val_df['y']
    X_test = prepare_features(test_df)
    y_test = test_df['y']
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    if storage_path is None:
        storage_path = f"sqlite:///logs/{study_name}.db"
    
    Path("logs").mkdir(exist_ok=True)
    
    sampler = TPESampler(seed=RANDOM_SEED)
    
    if resume:
        study = optuna.load_study(study_name=study_name, storage=storage_path)
        logger.info(f"Resuming study '{study_name}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            direction='maximize',
            load_if_exists=True
        )
        logger.info(f"Created new study '{study_name}'")
    
    logger.info(f"Starting hyperparameter search with {n_trials} trials (n_jobs={n_jobs})...")
    start_time = time.time()
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, device),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Search completed in {elapsed/60:.1f} minutes")
    
    logger.info("\n" + "="*60)
    logger.info("BEST TRIAL")
    logger.info("="*60)
    logger.info(f"Best auPRC: {study.best_value:.4f}")
    logger.info(f"Best params:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nTraining final model with best parameters on full training data...")
    best_params = get_param_space(study.best_trial)
    best_params['device'] = device
    
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    
    dtrain_full = xgb.DMatrix(X_full, label=y_full)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    best_params_final = best_params.copy()
    best_params_final.pop('device', None)
    best_params_final['tree_method'] = 'hist'
    best_params_final['nthread'] = 1
    
    final_model = xgb.train(
        best_params_final,
        dtrain_full,
        num_boost_round=best_iteration if best_iteration > 0 else 500,
        evals=[(dtrain_full, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    test_probs = final_model.predict(dtest)
    test_auprc = calculate_auprc(y_test.values, test_probs)
    
    precision, recall, thresholds = precision_recall_curve(y_test.values, test_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    logger.info(f"Test auPRC: {test_auprc:.4f}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    import pickle
    model_file = output_path / 'best_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'params': best_params,
            'optimal_threshold': optimal_threshold,
            'feature_names': list(X_train.columns),
            'best_iteration': final_model.best_iteration
        }, f)
    logger.info(f"Model saved to {model_file}")
    
    results = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'best_value': float(study.best_value),
        'best_params': study.best_params,
        'test_auprc': float(test_auprc),
        'optimal_threshold': float(optimal_threshold),
        'best_iteration': int(final_model.best_iteration),
        'elapsed_seconds': elapsed
    }
    
    results_file = output_path / 'hyperparam_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    import yaml
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    test_pred = (test_probs >= optimal_threshold).astype(int)
    
    gene_test_metrics = {
        'auPRC': float(test_auprc),
        'AUC': float(roc_auc_score(y_test, test_probs)),
        'Precision': float(precision_score(y_test, test_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, test_pred, zero_division=0)),
        'F1': float(f1_score(y_test, test_pred, zero_division=0)),
        'threshold': float(optimal_threshold)
    }
    
    results_df = pd.DataFrame({
        'sample': test_df['sample'],
        'y': y_test,
        'prob': test_probs
    })
    
    sample_agg = results_df.groupby('sample').agg({
        'y': 'max',
        'prob': 'max'
    }).reset_index()
    
    sample_y_true = sample_agg['y'].values
    sample_y_prob = sample_agg['prob'].values
    sample_y_pred = (sample_y_prob >= optimal_threshold).astype(int)
    
    sample_precision, sample_recall, _ = precision_recall_curve(sample_y_true, sample_y_prob)
    sample_auprc = auc(sample_recall, sample_precision)
    
    tp = ((sample_y_pred == 1) & (sample_y_true == 1)).sum()
    fp = ((sample_y_pred == 1) & (sample_y_true == 0)).sum()
    fn = ((sample_y_pred == 0) & (sample_y_true == 1)).sum()
    
    sample_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sample_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sample_f1 = 2 * sample_prec * sample_rec / (sample_prec + sample_rec) if (sample_prec + sample_rec) > 0 else 0.0
    
    sample_test_metrics = {
        'auPRC': float(sample_auprc),
        'AUC': float(roc_auc_score(sample_y_true, sample_y_prob)),
        'Precision': float(sample_prec),
        'Recall': float(sample_rec),
        'F1': float(sample_f1),
        'total_samples': int(len(sample_y_true)),
        'positive_samples': int(sample_y_true.sum()),
        'predicted_positive': int(sample_y_pred.sum())
    }
    
    training_summary = {
        'model_name': 'xgb_tuned',
        'hyperparameter_search': {
            'n_trials': len(study.trials),
            'best_val_auPRC': float(study.best_value),
            'search_time_seconds': elapsed
        },
        'best_params': study.best_params,
        'gene_level': {'test': gene_test_metrics},
        'sample_level': {'test': sample_test_metrics}
    }
    
    summary_file = output_path / 'training_summary.yml'
    with open(summary_file, 'w') as f:
        yaml.dump(training_summary, f, default_flow_style=False)
    logger.info(f"Training summary saved to {summary_file}")
    
    importance = best_model.get_score(importance_type='gain')
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} for k, v in importance.items()
    ]).sort_values('importance', ascending=False)
    importance_file = output_path / 'feature_importance.csv'
    importance_df.to_csv(importance_file, index=False)
    logger.info(f"Feature importance saved to {importance_file}")
    
    trials_df = study.trials_dataframe()
    trials_file = output_path / 'all_trials.csv'
    trials_df.to_csv(trials_file, index=False)
    logger.info(f"All trials saved to {trials_file}")
    
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Best validation auPRC: {study.best_value:.4f}")
    logger.info(f"Test gene-level auPRC: {test_auprc:.4f}")
    logger.info(f"Test sample-level auPRC: {sample_auprc:.4f}")
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"Best iteration: {best_iteration}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Search')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of trials (default: 100)')
    parser.add_argument('--study-name', type=str, default='xgb_hyperparam_search',
                        help='Study name for Optuna (default: xgb_hyperparam_search)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Storage path for Optuna (default: sqlite:///logs/{study_name}.db)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume existing study')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0, use -1 for CPU)')
    parser.add_argument('--output', type=str, default='otk_api/models/xgb_tuned',
                        help='Output directory (default: otk_api/models/xgb_tuned)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    
    args = parser.parse_args()
    
    import torch
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        logger.info(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    
    results = run_hyperparameter_search(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage_path=args.storage,
        resume=args.resume,
        device=device,
        output_dir=args.output,
        n_jobs=args.n_jobs
    )
    
    return results


if __name__ == '__main__':
    main()
