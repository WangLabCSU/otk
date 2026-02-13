#!/usr/bin/env python
"""
Configuration Generator for All ecDNA Models

Generates config.yml files for all models with unified structure.
"""

from pathlib import Path
import yaml

# Base configuration template
BASE_CONFIG = {
    'data': {
        'features': [
            'segVal', 'minor_cn', 'intersect_ratio', 'purity', 'ploidy',
            'AScore', 'pLOH', 'cna_burden',
            'CN1', 'CN2', 'CN3', 'CN4', 'CN5', 'CN6', 'CN7', 'CN8', 'CN9', 'CN10',
            'CN11', 'CN12', 'CN13', 'CN14', 'CN15', 'CN16', 'CN17', 'CN18', 'CN19',
            'age', 'gender',
            'type_BLCA', 'type_BRCA', 'type_CESC', 'type_COAD', 'type_DLBC',
            'type_ESCA', 'type_GBM', 'type_HNSC', 'type_KICH', 'type_KIRC',
            'type_KIRP', 'type_LGG', 'type_LIHC', 'type_LUAD', 'type_LUSC',
            'type_OV', 'type_PRAD', 'type_READ', 'type_SARC', 'type_SKCM',
            'type_STAD', 'type_THCA', 'type_UCEC', 'type_UVM',
            'freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR'
        ],
        'gene_id': 'gene_id',
        'sample_id': 'sample',
        'target': 'y',
        'missing_value_strategy': {'age': 'mean'}
    },
    'training': {
        'seed': 2026,
        'batch_size': 4096,
        'epochs': 150,
        'early_stopping': {
            'patience': 50,
            'min_delta': 0.001
        }
    },
    'prediction': {
        'threshold': 0.5
    }
}

# Model-specific configurations
MODEL_CONFIGS = {
    'xgb_new': {
        'model': {
            'type': 'XGBoost',
            'variant': 'XGBNew',
            'params': {
                'eta': 0.05,
                'max_depth': 6,
                'gamma': 3,
                'subsample': 0.8,
                'max_delta_step': 1,
                'min_child_weight': 2,
                'alpha': 0.1,
                'lambda': 2,
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'device': 'cuda'
            }
        },
        'features': {
            'engineering': [
                'cn_imbalance', 'allele_imbalance', 'cna_burden_adj',
                'ascore_adj', 'has_circular', 'has_bfb', 'has_hr',
                'amplicon_type_count', 'cn_sig_diversity', 'max_cn_sig',
                'purity_x_ploidy', 'has_loh'
            ]
        }
    },
    'xgb_paper': {
        'model': {
            'type': 'XGBoost',
            'variant': 'XGB11',
            'params': {
                'eta': 0.1,
                'max_depth': 4,
                'gamma': 10,
                'subsample': 0.6,
                'max_delta_step': 0,
                'min_child_weight': 1,
                'alpha': 0,
                'lambda': 1,
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'device': 'cuda'
            }
        },
        'features': {
            'core_11': [
                'total_cn', 'minor_cn', 'purity', 'ploidy', 'AScore',
                'pLOH', 'cna_burden', 'freq_Linear', 'freq_BFB',
                'freq_Circular', 'freq_HR'
            ]
        }
    },
    'baseline_mlp': {
        'model': {
            'type': 'NeuralNetwork',
            'variant': 'BaselineMLP',
            'architecture': {
                'layers': [
                    {'input_dim': 57, 'output_dim': 128, 'activation': 'ReLU', 'dropout': 0.3},
                    {'input_dim': 128, 'output_dim': 64, 'activation': 'ReLU', 'dropout': 0.3},
                    {'input_dim': 64, 'output_dim': 1, 'activation': None}
                ]
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 0.001,
                'weight_decay': 0.0001
            },
            'loss_function': {
                'type': 'BCEWithLogitsLoss',
                'pos_weight': 10.0
            }
        }
    },
    'transformer': {
        'model': {
            'type': 'NeuralNetwork',
            'variant': 'Transformer',
            'architecture': {
                'input_dim': 57,
                'hidden_dim': 128,
                'num_heads': 4,
                'num_layers': 3,
                'dropout': 0.3,
                'classifier': [
                    {'input_dim': 128, 'output_dim': 64, 'activation': 'GELU', 'dropout': 0.3},
                    {'input_dim': 64, 'output_dim': 1, 'activation': None}
                ]
            },
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.001,
                'weight_decay': 0.01
            },
            'loss_function': {
                'type': 'BCEWithLogitsLoss',
                'pos_weight': 10.0
            },
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'patience': 10
            }
        }
    },
    'deep_residual': {
        'model': {
            'type': 'NeuralNetwork',
            'variant': 'DeepResidual',
            'architecture': {
                'input_dim': 57,
                'hidden_dim': 128,
                'num_residual_blocks': 6,
                'dropout': 0.3,
                'classifier': [
                    {'input_dim': 128, 'output_dim': 64, 'activation': 'ReLU', 'dropout': 0.3},
                    {'input_dim': 64, 'output_dim': 1, 'activation': None}
                ]
            },
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.001,
                'weight_decay': 0.01
            },
            'loss_function': {
                'type': 'BCEWithLogitsLoss',
                'pos_weight': 10.0
            },
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'patience': 10
            }
        }
    },
    'optimized_residual': {
        'model': {
            'type': 'NeuralNetwork',
            'variant': 'OptimizedResidual',
            'architecture': {
                'input_dim': 57,
                'hidden_dim': 128,
                'num_residual_blocks': 8,
                'dropout': 0.3,
                'classifier': [
                    {'input_dim': 128, 'output_dim': 64, 'activation': 'ReLU', 'dropout': 0.3},
                    {'input_dim': 64, 'output_dim': 1, 'activation': None}
                ]
            },
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.001,
                'weight_decay': 0.01
            },
            'loss_function': {
                'type': 'BCEWithLogitsLoss',
                'pos_weight': 10.0
            },
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'patience': 10
            }
        }
    },
    'dgit_super': {
        'model': {
            'type': 'NeuralNetwork',
            'variant': 'DGITSuper',
            'architecture': {
                'input_dim': 57,
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.3,
                'classifier': [
                    {'input_dim': 256, 'output_dim': 128, 'activation': 'ReLU', 'dropout': 0.3},
                    {'input_dim': 128, 'output_dim': 64, 'activation': 'ReLU', 'dropout': 0.3},
                    {'input_dim': 64, 'output_dim': 1, 'activation': None}
                ]
            },
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.001,
                'weight_decay': 0.01
            },
            'loss_function': {
                'type': 'BCEWithLogitsLoss',
                'pos_weight': 10.0
            },
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'patience': 10
            }
        }
    }
}


def generate_config(model_name: str) -> dict:
    """Generate configuration for a model"""
    config = BASE_CONFIG.copy()
    
    if model_name in MODEL_CONFIGS:
        config.update(MODEL_CONFIGS[model_name])
    
    return config


def save_config(model_name: str, output_dir: Path):
    """Save configuration to file"""
    config = generate_config(model_name)
    
    config_path = output_dir / 'config.yml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config saved: {config_path}")


def main():
    """Generate configs for all models"""
    models_dir = Path('/data/home/wsx/Projects/otk/otk/otk_api/models')
    
    all_models = [
        'xgb_new', 'xgb_paper',
        'baseline_mlp', 'transformer',
        'deep_residual', 'optimized_residual', 'dgit_super'
    ]
    
    for model_name in all_models:
        model_dir = models_dir / model_name
        save_config(model_name, model_dir)
    
    print("\nAll configs generated successfully!")


if __name__ == "__main__":
    main()
