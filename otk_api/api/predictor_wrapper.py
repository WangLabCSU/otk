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

from otk.predict.predictor import Predictor
from .config import DEFAULT_COLUMN_VALUES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.error(f"Failed to initialize predictor: {e}")
            raise
    
    def predict(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            results = self.predictor.run(input_path, output_dir)
            
            processing_time = time.time() - start_time
            
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
            
            return {
                "success": True,
                "stats": stats,
                "results_path": Path(output_dir) / "predictions.csv",
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": {"processing_time": time.time() - start_time},
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
    logger.info(f"Starting prediction job {job_id}")
    logger.info(f"Model: {model_path}, Device: {device_type}, GPU: {gpu_id}")
    
    try:
        # Fill missing columns with default values before prediction
        from .config import UPLOAD_DIR
        processed_path = Path(UPLOAD_DIR) / f"{job_id}_processed.csv"
        
        if not fill_missing_columns(input_path, str(processed_path)):
            logger.warning("Failed to fill missing columns, using original file")
            processed_path = Path(input_path)
        
        wrapper = PredictorWrapper(model_path, device_type, gpu_id)
        result = wrapper.predict(str(processed_path), output_dir)
        wrapper.cleanup()
        
        result["job_id"] = job_id
        result["model_used"] = Path(model_path).parent.name
        result["device_used"] = f"{device_type}:{gpu_id}" if gpu_id is not None else device_type
        
        return result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        return {
            "success": False,
            "job_id": job_id,
            "error": str(e),
        }
