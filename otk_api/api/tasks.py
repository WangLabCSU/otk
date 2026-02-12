import os
import sys
sys.path.insert(0, '/data/home/wsx/Projects/otk/otk/src')

from celery import Celery
from datetime import datetime
from pathlib import Path

from .config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, RESULTS_DIR
from .predictor_wrapper import run_prediction_job
from .resource_manager import resource_manager

celery_app = Celery(
    "otk_api",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_prefetch_multiplier=1,
)

@celery_app.task(bind=True, max_retries=3)
def process_prediction_task(self, job_id: str, input_path: str):
    """Celery task for processing prediction jobs"""
    from .models import SessionLocal, Job, Statistics
    
    db = SessionLocal()
    
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"success": False, "error": "Job not found"}
        
        job.status = "processing"
        job.updated_at = datetime.utcnow()
        db.commit()
        
        model_name, model_path, device_type, gpu_id = resource_manager.allocate_resource(job_id)
        
        job.model_used = model_name
        job.device_used = f"{device_type}:{gpu_id}" if gpu_id is not None else device_type
        db.commit()
        
        output_dir = RESULTS_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        
        result = run_prediction_job(
            job_id=job_id,
            input_path=input_path,
            output_dir=str(output_dir),
            model_path=str(model_path),
            device_type=device_type,
            gpu_id=gpu_id,
        )
        
        if result["success"]:
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.results_file_path = str(output_dir / "predictions.csv")
            
            stats = result.get("stats", {})
            job.total_rows = stats.get("total_rows", 0)
            job.total_samples = stats.get("total_samples", 0)
            job.total_genes = stats.get("total_genes", 0)
            job.processing_time = stats.get("processing_time", 0)
            
            _update_statistics(db, True, stats, device_type)
        else:
            job.status = "failed"
            job.error_message = result.get("error", "Unknown error")
            _update_statistics(db, False, {}, device_type)
        
        job.updated_at = datetime.utcnow()
        db.commit()
        
        resource_manager.release_resource(job_id, device_type, gpu_id)
        
        return result
        
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.updated_at = datetime.utcnow()
        db.commit()
        
        resource_manager.release_resource(job_id, device_type if 'device_type' in locals() else "cpu", 
                                        gpu_id if 'gpu_id' in locals() else None)
        
        self.retry(countdown=60, exc=e)
        
    finally:
        db.close()

def _update_statistics(db, success: bool, stats: dict, device_type: str):
    """Update global statistics"""
    from .models import Statistics
    
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    stat = db.query(Statistics).filter(Statistics.date >= today).first()
    
    if not stat:
        stat = Statistics()
        db.add(stat)
    
    stat.total_jobs += 1
    
    if success:
        stat.completed_jobs += 1
        stat.total_rows_processed += stats.get("total_rows", 0)
        stat.total_samples_processed += stats.get("total_samples", 0)
        
        current_avg = stat.avg_processing_time
        current_count = stat.completed_jobs - 1
        new_time = stats.get("processing_time", 0)
        
        if current_count > 0:
            stat.avg_processing_time = (current_avg * current_count + new_time) / (current_count + 1)
        else:
            stat.avg_processing_time = new_time
    else:
        stat.failed_jobs += 1
    
    if device_type == "cuda":
        stat.gpu_jobs += 1
    else:
        stat.cpu_jobs += 1
    
    db.commit()
