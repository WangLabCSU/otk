import sys
from pathlib import Path

api_dir = Path(__file__).parent
otk_src_path = api_dir.parent.parent.parent / "src"
sys.path.insert(0, str(otk_src_path))

import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time
import logging
import traceback

from otk.predict.predictor import Predictor
from .config import DEFAULT_COLUMN_VALUES

# Setup logging with file handler
logs_dir = api_dir.parent / "logs"
logs_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler for detailed logs
file_handler = logging.FileHandler(logs_dir / "prediction.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def fill_missing_columns(input_path: str, output_path: str) -> bool:
    """Fill missing optional columns with default values"""
    try:
        df = pd.read_csv(input_path)
        
        # Fill missing columns with default values
        for col, default_val in DEFAULT_COLUMN_VALUES.items():
            if col not in df.columns:
                df[col] = default_val
                logger.info(f"Added missing column '{col}' with default value {default_val}")
        
        # Save to output path
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to fill missing columns: {e}")
        return False

class PredictorWrapper:
    def __init__(self, model_path: str, device_type: str = "cpu", gpu_id: Optional[int] = None):
        self.model_path = model_path
        self.device_type = device_type
        self.gpu_id = gpu_id
        self.predictor = None
        self._init_predictor()
    
    def _init_predictor(self):
        try:
            gpu_param = self.gpu_id if self.device_type == "cuda" and self.gpu_id is not None else -1
            self.predictor = Predictor(self.model_path, gpu=gpu_param)
            logger.info(f"Predictor initialized with model: {self.model_path}, device: {self.device_type}")
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Failed to initialize predictor: {e}")
            logger.error(f"Traceback:\n{error_trace}")
            raise
    
    def predict(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            logger.info(f"Starting prediction for input: {input_path}")
            logger.info(f"Output directory: {output_dir}")
            
            # Check input file
            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            logger.info(f"Input file size: {input_file.stat().st_size} bytes")
            
            # Run prediction
            logger.debug("Calling predictor.run()...")
            results = self.predictor.run(input_path, output_dir)
            logger.debug(f"predictor.run() completed, results shape: {results.shape if hasattr(results, 'shape') else 'N/A'}")
            
            processing_time = time.time() - start_time
            
            # Calculate stats
            stats = {
                "total_rows": len(results),
                "total_samples": results['sample'].nunique() if 'sample' in results.columns else 0,
                "total_genes": results['gene_id'].nunique() if 'gene_id' in results.columns else 0,
                "processing_time": processing_time,
                "ecdna_predictions": int(results['prediction'].sum()) if 'prediction' in results.columns else 0,
                "sample_level_distribution": results['sample_level_prediction_label'].value_counts().to_dict() if 'sample_level_prediction_label' in results.columns else {},
            }
            
            logger.info(f"Prediction completed in {processing_time:.2f}s")
            logger.info(f"Stats: {stats}")
            
            # Check output file
            output_file = Path(output_dir) / "predictions.csv"
            if output_file.exists():
                logger.info(f"Output file created: {output_file} ({output_file.stat().st_size} bytes)")
            else:
                logger.warning(f"Output file not found: {output_file}")
            
            return {
                "success": True,
                "stats": stats,
                "results_path": output_file,
            }
            
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            processing_time = time.time() - start_time
            
            logger.error(f"Prediction failed after {processing_time:.2f}s: {error_msg}")
            logger.error(f"Traceback:\n{error_trace}")
            
            return {
                "success": False,
                "error": error_msg,
                "error_trace": error_trace,
                "stats": {"processing_time": processing_time},
            }
    
    def cleanup(self):
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Predictor cleanup completed")

def run_prediction_job(
    job_id: str,
    input_path: str,
    output_dir: str,
    model_path: str,
    device_type: str,
    gpu_id: Optional[int]
) -> Dict[str, Any]:
    """Run prediction job and return results"""
    job_start_time = time.time()
    
    logger.info("=" * 60)
    logger.info(f"Starting prediction job {job_id}")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Device: {device_type}, GPU: {gpu_id}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info("=" * 60)
    
    try:
        # Fill missing columns with default values before prediction
        from .config import UPLOAD_DIR
        processed_path = Path(UPLOAD_DIR) / f"{job_id}_processed.csv"
        
        logger.info("Step 1: Filling missing columns...")
        if not fill_missing_columns(input_path, str(processed_path)):
            logger.warning("Failed to fill missing columns, using original file")
            processed_path = Path(input_path)
        else:
            logger.info(f"Missing columns filled, processed file: {processed_path}")
        
        logger.info("Step 2: Initializing predictor...")
        wrapper = PredictorWrapper(model_path, device_type, gpu_id)
        
        logger.info("Step 3: Running prediction...")
        result = wrapper.predict(str(processed_path), output_dir)
        
        logger.info("Step 4: Cleaning up...")
        wrapper.cleanup()
        
        # Add job metadata
        result["job_id"] = job_id
        result["model_used"] = Path(model_path).parent.name
        result["device_used"] = f"{device_type}:{gpu_id}" if gpu_id is not None else device_type
        result["total_time"] = time.time() - job_start_time
        
        if result.get("success"):
            logger.info("=" * 60)
            logger.info(f"Job {job_id} completed successfully")
            logger.info(f"Total time: {result['total_time']:.2f}s")
            logger.info(f"Processing time: {result.get('stats', {}).get('processing_time', 'N/A')}s")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error(f"Job {job_id} failed: {result.get('error', 'Unknown error')}")
            logger.error("=" * 60)
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        total_time = time.time() - job_start_time
        
        logger.error("=" * 60)
        logger.error(f"Job {job_id} failed after {total_time:.2f}s: {error_msg}")
        logger.error(f"Traceback:\n{error_trace}")
        logger.error("=" * 60)
        
        return {
            "success": False,
            "job_id": job_id,
            "error": error_msg,
            "error_trace": error_trace,
            "total_time": total_time,
        }
