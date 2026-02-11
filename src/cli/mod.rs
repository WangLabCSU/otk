use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// OTK: ecDNA prediction tool using deep learning
#[derive(Parser, Debug)]
#[command(name = "otk")]
#[command(about = "ecDNA prediction tool using deep learning")]
#[command(version)]
pub struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,
    
    /// Subcommand
    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train a new model
    Train(TrainArgs),
    
    /// Make predictions using a trained model
    Predict(PredictArgs),
    
    /// Evaluate model performance
    Evaluate(EvaluateArgs),
    
    /// Convert model to different format
    Convert(ConvertArgs),
}

/// Training arguments
#[derive(Parser, Debug)]
pub struct TrainArgs {
    /// Input data file (CSV or TSV)
    #[arg(short, long, required = true)]
    pub input: PathBuf,
    
    /// Output directory for model and checkpoints
    #[arg(short, long, default_value = "./output")]
    pub output: PathBuf,
    
    /// Model configuration file
    #[arg(short, long)]
    pub config: Option<PathBuf>,
    
    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    pub epochs: usize,
    
    /// Batch size
    #[arg(short, long, default_value = "256")]
    pub batch_size: usize,
    
    /// Learning rate
    #[arg(long, default_value = "0.001")]
    pub learning_rate: f64,
    
    /// Dropout rate
    #[arg(long, default_value = "0.3")]
    pub dropout: f64,
    
    /// Random seed
    #[arg(long, default_value = "2026")]
    pub seed: u64,
    
    /// Validation ratio
    #[arg(long, default_value = "0.1")]
    pub val_ratio: f64,
    
    /// Test ratio
    #[arg(long, default_value = "0.2")]
    pub test_ratio: f64,
    
    /// Early stopping patience
    #[arg(long, default_value = "15")]
    pub patience: usize,
    
    /// Device to use (cpu, cuda, wgpu)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,
    
    /// Resume from checkpoint
    #[arg(long)]
    pub resume: Option<PathBuf>,
    
    /// Quick test mode (fewer epochs)
    #[arg(long)]
    pub quick: bool,
}

/// Prediction arguments
#[derive(Parser, Debug)]
pub struct PredictArgs {
    /// Input data file (CSV or TSV)
    #[arg(short, long, required = true)]
    pub input: PathBuf,
    
    /// Model checkpoint file
    #[arg(short, long, required = true)]
    pub model: PathBuf,
    
    /// Output file for predictions
    #[arg(short, long, default_value = "predictions.csv")]
    pub output: PathBuf,
    
    /// Output format (csv, json)
    #[arg(short, long, default_value = "csv")]
    pub format: String,
    
    /// Batch size for prediction
    #[arg(short, long, default_value = "256")]
    pub batch_size: usize,
    
    /// Probability threshold for positive classification
    #[arg(long, default_value = "0.5")]
    pub threshold: f64,
    
    /// Device to use (cpu, cuda, wgpu)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,
    
    /// Save sample-level predictions
    #[arg(long)]
    pub sample_level: bool,
}

/// Evaluation arguments
#[derive(Parser, Debug)]
pub struct EvaluateArgs {
    /// Input data file with ground truth labels
    #[arg(short, long, required = true)]
    pub input: PathBuf,
    
    /// Model checkpoint file
    #[arg(short, long, required = true)]
    pub model: PathBuf,
    
    /// Output file for evaluation report
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    
    /// Batch size for evaluation
    #[arg(short, long, default_value = "256")]
    pub batch_size: usize,
    
    /// Device to use (cpu, cuda, wgpu)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,
}

/// Conversion arguments
#[derive(Parser, Debug)]
pub struct ConvertArgs {
    /// Input model checkpoint
    #[arg(short, long, required = true)]
    pub input: PathBuf,
    
    /// Output file path
    #[arg(short, long, required = true)]
    pub output: PathBuf,
    
    /// Output format (onnx, torch)
    #[arg(short, long, default_value = "onnx")]
    pub format: String,
}

/// Parse CLI arguments
pub fn parse_args() -> Cli {
    Cli::parse()
}

/// Setup logging based on verbosity
pub fn setup_logging(verbose: bool) {
    let filter = if verbose {
        "debug"
    } else {
        "info"
    };
    
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cli_parse() {
        let cli = Cli::parse_from(["otk", "train", "-i", "data.csv"]);
        
        match cli.command {
            Commands::Train(args) => {
                assert_eq!(args.input, PathBuf::from("data.csv"));
                assert_eq!(args.epochs, 100);
            }
            _ => panic!("Expected Train command"),
        }
    }
    
    #[test]
    fn test_predict_args() {
        let cli = Cli::parse_from([
            "otk", "predict",
            "-i", "input.csv",
            "-m", "model.mpk",
            "-o", "output.csv"
        ]);
        
        match cli.command {
            Commands::Predict(args) => {
                assert_eq!(args.input, PathBuf::from("input.csv"));
                assert_eq!(args.model, PathBuf::from("model.mpk"));
                assert_eq!(args.output, PathBuf::from("output.csv"));
            }
            _ => panic!("Expected Predict command"),
        }
    }
}