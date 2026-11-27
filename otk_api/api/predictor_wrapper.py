import os
import sys
from pathlib import Path

api_dir = Path(__file__).parent
otk_src_path = api_dir.parent.parent.parent / "src"
sys.path.insert(0, str(otk_src_path))

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time
import logging
import traceback

from otk.predict.predictor import Predictor
from .config import DEFAULT_COLUMN_VALUES


def _torch():
    """Import torch lazily.

    The prediction wrapper only needs torch for CUDA cache/telemetry. Importing
    it eagerly at module scope would load a second OpenMP (libomp) runtime
    alongside xgboost and segfault the API server on macOS.
    """
    import torch

    return torch

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
            output_file = Path(output_dir) / "predictions.csv"
            results = self.predictor.run(input_path, output_file)
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
            _torch().cuda.empty_cache()
        logger.info("Predictor cleanup completed")


def _run_prediction_in_subprocess(
    job_id: str,
    input_path: str,
    output_dir: str,
    model_path: str,
    device_type: str,
    gpu_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Run a prediction job in a fresh interpreter, returning its result.

    The API server is long-lived: if it served an XGBoost model and then a torch
    model (or vice versa) in the same process, two OpenMP (libomp) runtimes
    would end up loaded together and the process would hang/segfault on macOS.
    Isolating every job in its own process guarantees a job only ever loads the
    single backend its model needs.

    Returns ``None`` only when the worker could not be launched at all (for
    example an unavailable interpreter), so the caller can fall back to
    in-process execution. A worker that starts but crashes returns an error
    result instead.
    """
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        result_file = handle.name

    cmd = [
        sys.executable, "-m", "otk_api.api.predict_worker",
        "--job-id", job_id,
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--model", str(model_path),
        "--device-type", device_type,
        "--result-file", result_file,
    ]
    if gpu_id is not None:
        cmd.extend(["--gpu-id", str(gpu_id)])

    env = os.environ.copy()
    env.setdefault("OTK_MODELS_DIR", str(Path(model_path).parent.parent))

    # Make ``otk_api`` (repo root) and ``otk`` (src layout) importable in the
    # child regardless of the server's working directory.
    repo_root = api_dir.parent.parent
    python_path = [str(repo_root), str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    try:
        from .config import JOB_TIMEOUT
    except Exception:
        JOB_TIMEOUT = 7200

    logger.info(f"Running prediction job in subprocess: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=JOB_TIMEOUT)
    except OSError as exc:
        logger.warning(f"Could not launch prediction subprocess: {exc}")
        try:
            Path(result_file).unlink()
        except OSError:
            pass
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"Prediction subprocess for job {job_id} timed out after {JOB_TIMEOUT}s")
        try:
            Path(result_file).unlink()
        except OSError:
            pass
        return {
            "success": False,
            "job_id": job_id,
            "error": f"Prediction timed out after {JOB_TIMEOUT}s",
        }

    try:
        payload = json.loads(Path(result_file).read_text())
    except (OSError, json.JSONDecodeError):
        logger.error(
            f"Prediction subprocess did not produce a result file "
            f"(exit {proc.returncode}).\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        return {
            "success": False,
            "job_id": job_id,
            "error": f"Prediction subprocess failed (exit {proc.returncode})",
            "error_trace": proc.stderr,
        }
    finally:
        try:
            Path(result_file).unlink()
        except OSError:
            pass

    if not payload.get("success"):
        logger.error(f"Prediction subprocess reported failure: {payload.get('error')}")
        if payload.get("error_trace"):
            logger.debug(payload["error_trace"])
    return payload


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
    logger.info("=" * 60)

    # Prefer an isolated subprocess: see _run_prediction_in_subprocess.
    subprocess_result = _run_prediction_in_subprocess(
        job_id, input_path, output_dir, model_path, device_type, gpu_id
    )
    if subprocess_result is not None:
        subprocess_result["total_time"] = time.time() - job_start_time
        if subprocess_result.get("success"):
            logger.info(f"Job {job_id} completed successfully in a subprocess")
        return subprocess_result

    logger.warning("Falling back to in-process prediction execution")

    if device_type == "cuda":
        torch = _torch()
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
    
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
