use anyhow::{Context, Result};
use burn::backend::NdArray;
use otk::cli::{parse_args, setup_logging, Commands, TrainArgs, PredictArgs, EvaluateArgs};
use otk::data::{DataLoader, convert_to_features, Preprocessor, SplitConfig};
use otk::model::ModelConfig;
use otk::predict::predictor::{Predictor, utils as predict_utils};
use otk::training::{TrainingConfig, trainer::Trainer};
use tracing::{info, error};

fn main() {
    let cli = parse_args();
    
    setup_logging(cli.verbose);
    
    info!("{}", otk::info());
    
    let result = match cli.command {
        Commands::Train(args) => run_train(args),
        Commands::Predict(args) => run_predict(args),
        Commands::Evaluate(args) => run_evaluate(args),
        Commands::Convert(args) => run_convert(args),
    };
    
    if let Err(e) = result {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_train(args: TrainArgs) -> Result<()> {
    info!("Starting training...");
    info!("Input file: {:?}", args.input);
    info!("Output directory: {:?}", args.output);
    
    otk::utils::ensure_dir(&args.output)?;
    
    info!("Loading data...");
    let loader = DataLoader::new();
    let records = loader.load(&args.input)
        .with_context(|| format!("Failed to load data from {:?}", args.input))?;
    
    info!("Loaded {} records", records.len());
    
    let mut vectors = convert_to_features(&records);
    
    info!("Preprocessing data...");
    let mut preprocessor = Preprocessor::new();
    preprocessor.fit_transform(&mut vectors)
        .context("Failed to preprocess data")?;
    
    let split_config = SplitConfig {
        train_ratio: 1.0 - args.val_ratio - args.test_ratio,
        val_ratio: args.val_ratio,
        test_ratio: args.test_ratio,
        seed: args.seed,
    };
    
    info!("Splitting dataset...");
    let dataset = otk::data::preprocessing::split_by_sample(vectors, &split_config);
    
    let training_config = TrainingConfig {
        epochs: if args.quick { 5 } else { args.epochs },
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        dropout: args.dropout,
        early_stopping_patience: args.patience,
        seed: args.seed,
        ..TrainingConfig::default()
    };
    
    let model_config = ModelConfig::ecdna_default()
        .with_dropout(args.dropout);
    
    let device = NdArray::<f32>::Device::default();
    
    let checkpoint_dir = args.output.join("checkpoints");
    otk::utils::ensure_dir(&checkpoint_dir)?;
    
    let mut trainer = Trainer::new(training_config, model_config, device)
        .with_checkpoint_dir(&checkpoint_dir)?;
    
    info!("Starting model training...");
    let result = trainer.train(&dataset)
        .context("Training failed")?;
    
    info!("\n=== Training Results ===");
    info!("Total epochs: {}", result.state.epoch);
    info!("Best AUPRC: {:.4}", result.state.best_metric);
    info!("Training time: {:.2}s", result.duration_secs);
    info!("\nFinal Test Metrics:");
    info!("  AUPRC: {:.4}", result.final_metrics.auprc);
    info!("  F1: {:.4}", result.final_metrics.f1);
    info!("  Precision: {:.4}", result.final_metrics.precision);
    info!("  Recall: {:.4}", result.final_metrics.recall);
    info!("  Accuracy: {:.4}", result.final_metrics.accuracy);
    
    if let Some(ref checkpoint) = result.best_checkpoint {
        info!("\nBest model saved to: {:?}", checkpoint);
    }
    
    Ok(())
}

fn run_predict(args: PredictArgs) -> Result<()> {
    info!("Starting prediction...");
    info!("Input file: {:?}", args.input);
    info!("Model: {:?}", args.model);
    info!("Output file: {:?}", args.output);
    
    let device = NdArray::<f32>::Device::default();
    
    info!("Loading model...");
    let predictor = Predictor::from_checkpoint(&args.model, device)
        .with_context(|| format!("Failed to load model from {:?}", args.model))?;
    
    let predictor = predictor.with_batch_size(args.batch_size);
    
    info!("Running prediction...");
    let predictions = predictor.predict_from_file(&args.input)
        .context("Prediction failed")?;
    
    predictions.summary.print();
    
    match args.format.as_str() {
        "csv" => {
            predict_utils::save_predictions_to_csv(&predictions, &args.output)?;
            
            if args.sample_level {
                let sample_output = args.output.with_extension("samples.csv");
                predict_utils::save_sample_predictions_to_csv(&predictions, sample_output)?;
            }
        }
        "json" => {
            predict_utils::save_predictions_to_json(&predictions, &args.output)?;
        }
        _ => {
            anyhow::bail!("Unsupported output format: {}", args.format);
        }
    }
    
    info!("Predictions saved to: {:?}", args.output);
    
    Ok(())
}

fn run_evaluate(args: EvaluateArgs) -> Result<()> {
    info!("Starting evaluation...");
    info!("Input file: {:?}", args.input);
    info!("Model: {:?}", args.model);
    
    let device = NdArray::<f32>::Device::default();
    
    info!("Loading model...");
    let predictor = Predictor::from_checkpoint(&args.model, device)
        .with_context(|| format!("Failed to load model from {:?}", args.model))?;
    
    let predictor = predictor.with_batch_size(args.batch_size);
    
    info!("Running evaluation...");
    let predictions = predictor.predict_from_file(&args.input)
        .context("Evaluation failed")?;
    
    info!("\n=== Evaluation Results ===");
    predictions.summary.print();
    
    if let Some(output) = args.output {
        let report = serde_json::to_string_pretty(&predictions.summary)?;
        std::fs::write(&output, report)?;
        info!("Evaluation report saved to: {:?}", output);
    }
    
    Ok(())
}

fn run_convert(args: otk::cli::ConvertArgs) -> Result<()> {
    info!("Converting model...");
    info!("Input: {:?}", args.input);
    info!("Output: {:?}", args.output);
    info!("Format: {}", args.format);
    
    info!("Model conversion not yet implemented");
    
    Ok(())
}
