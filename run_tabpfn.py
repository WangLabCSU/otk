#!/usr/bin/env python
"""
Train TabPFN model independently
Using TabPFN v2 (no authentication required)
"""
import sys
import os
import time
from datetime import datetime

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    
    print("=" * 60)
    print("Training TabPFN Model (v2)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load data
        import gzip
        data_path = 'src/otk/data/sorted_modeling_data.csv.gz'
        print(f"Loading data from {data_path}")
        
        with gzip.open(data_path, 'rt') as f:
            df = pd.read_csv(f)
        
        print(f"Loaded data shape: {df.shape}")
        
        feature_cols = [col for col in df.columns if col not in ['sample', 'gene_id', 'y']]
        X = df[feature_cols].values
        y = df['y'].values
        samples = df['sample'].values
        
        # Split by sample
        unique_samples = np.unique(samples)
        np.random.seed(42)
        np.random.shuffle(unique_samples)
        
        n_samples = len(unique_samples)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.82)
        
        train_samples = set(unique_samples[:train_end])
        val_samples = set(unique_samples[train_end:val_end])
        test_samples = set(unique_samples[val_end:])
        
        train_mask = np.isin(samples, list(train_samples))
        val_mask = np.isin(samples, list(val_samples))
        test_mask = np.isin(samples, list(test_samples))
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"Train: {X_train.shape[0]} samples, {y_train.sum()} positive")
        print(f"Val: {X_val.shape[0]} samples, {y_val.sum()} positive")
        print(f"Test: {X_test.shape[0]} samples, {y_test.sum()} positive")
        
        # Stratified sampling for TabPFN (max ~5000 samples recommended)
        def stratified_sample(X, y, n_samples, seed):
            np.random.seed(seed)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            
            pos_ratio = len(pos_idx) / len(y)
            n_pos = min(int(n_samples * pos_ratio * 2), len(pos_idx))
            n_neg = min(n_samples - n_pos, len(neg_idx))
            
            selected_pos = np.random.choice(pos_idx, size=n_pos, replace=False)
            selected_neg = np.random.choice(neg_idx, size=n_neg, replace=False)
            
            selected_idx = np.concatenate([selected_pos, selected_neg])
            np.random.shuffle(selected_idx)
            
            return X[selected_idx], y[selected_idx]
        
        # Train ensemble of TabPFN models
        n_estimators = 5
        max_samples = 5000
        
        models = []
        for i in range(n_estimators):
            X_sample, y_sample = stratified_sample(X_train, y_train, max_samples, seed=42 + i * 100)
            print(f"\nModel {i+1}/{n_estimators}: Training on {len(X_sample)} samples "
                  f"({(y_sample==1).sum()} positive, {(y_sample==0).sum()} negative)")
            
            clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cuda')
            clf.fit(X_sample, y_sample)
            models.append(clf)
        
        # Predict on validation and test sets
        def predict_ensemble(models, X, batch_size=10000):
            n_samples = len(X)
            all_probs = []
            
            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                batch_probs = []
                
                for model in models:
                    proba = model.predict_proba(batch)
                    batch_probs.append(proba[:, 1])
                
                avg_prob = np.mean(batch_probs, axis=0)
                all_probs.append(avg_prob)
                
                if (i // batch_size) % 10 == 0:
                    print(f"Predicted {min(i+batch_size, n_samples)}/{n_samples} samples")
            
            return np.concatenate(all_probs)
        
        print("\nPredicting on validation set...")
        val_probs = predict_ensemble(models, X_val)
        
        print("\nPredicting on test set...")
        test_probs = predict_ensemble(models, X_test)
        
        # Calculate metrics
        def calc_metrics(probs, y_true):
            preds = (probs >= 0.5).astype(int)
            
            precision, recall, _ = precision_recall_curve(y_true, probs)
            auprc = auc(recall, precision)
            auc_score = roc_auc_score(y_true, probs)
            
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            return {'auPRC': auprc, 'AUC': auc_score, 'Precision': prec, 'Recall': rec, 'F1': f1}
        
        val_metrics = calc_metrics(val_probs, y_val)
        test_metrics = calc_metrics(test_probs, y_test)
        
        elapsed = time.time() - start_time
        
        # Save results
        import yaml
        os.makedirs('otk_api/models/tabpfn', exist_ok=True)
        summary = {
            'model': 'TabPFN Ensemble (v2)',
            'n_estimators': n_estimators,
            'max_samples_per_estimator': max_samples,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        with open('otk_api/models/tabpfn/training_summary.yml', 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Validation auPRC: {val_metrics['auPRC']:.4f}")
        print(f"Validation Precision: {val_metrics['Precision']:.4f}")
        print(f"Validation Recall: {val_metrics['Recall']:.4f}")
        print(f"\nTest auPRC: {test_metrics['auPRC']:.4f}")
        print(f"Test Precision: {test_metrics['Precision']:.4f}")
        print(f"Test Recall: {test_metrics['Recall']:.4f}")
        print(f"Test F1: {test_metrics['F1']:.4f}")
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
