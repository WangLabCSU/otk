# OTK Model Refactoring Plan

## Goals
1. Unify all models to use consistent 80/10/10 data split (seed=2026)
2. Maximize gene-level auPRC (with Precision >= 0.8) and sample-level auROC
3. Standardize model interfaces (fit, predict, evaluate, save, load)
4. Clean up unused code
5. Parallel training with monitoring

## Current Status

### Data Split
- ✅ `data_split.py`: New unified split module (80/10/10, seed=2026)
- ⚠️ `data_processor.py`: Has old split logic - needs to use data_split

### Models in otk_api/models/
1. `baseline_mlp/` - Keep, retrain
2. `deep_residual/` - Keep, retrain
3. `dgit_super/` - Keep, retrain
4. `optimized_residual/` - Keep, retrain
5. `transformer/` - Keep, retrain
6. `xgb11_paper/` - Keep
7. `xgb11_full/` - Keep
8. `xgb_new/` - Keep (best performer)

### Code to Clean Up
- Old model implementations in src/otk/models/ (keep only working ones)
- Duplicate split logic

## Implementation Steps

### Phase 1: Base Interface
1. ✅ Create `base_model.py` with BaseEcDNAModel and ModelTrainer

### Phase 2: Update XGB Models
1. Update XGB11Model to inherit from BaseEcDNAModel
2. Update XGBNewModel to inherit from BaseEcDNAModel

### Phase 3: Update Neural Network Models
1. Create unified neural network trainer
2. Update all NN models to use data_split

### Phase 4: Data Processor
1. Refactor to use data_split module
2. Remove duplicate split logic

### Phase 5: Parallel Training
1. Create parallel training script
2. Add monitoring and auto-restart

### Phase 6: Update Analyzer
1. Update model_analyzer to use unified split
2. Generate comprehensive report

## Model Training Priority

1. **XGB New** (highest priority - already performs well)
2. **Transformer** (best NN model)
3. **Deep Residual** (balanced)
4. **Baseline MLP** (baseline)
5. **Optimized Residual**
6. **DGIT Super** (needs redesign)
7. **XGB11 Paper** (for comparison)
8. **XGB11 Full** (for comparison)

## Success Criteria

### Gene Level
- auPRC >= 0.85
- Precision >= 0.8

### Sample Level
- auPRC >= 0.99
- auROC >= 0.9

## Notes
- Dataset has ~500 samples, mostly ecDNA+
- Only ~0.35% genes are ecDNA cargo genes
- Sample-level prediction: any gene positive → sample positive
