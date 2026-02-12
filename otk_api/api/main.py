import os
import sys
from pathlib import Path

api_dir = Path(__file__).parent
otk_src_path = api_dir.parent.parent.parent / "src"
sys.path.insert(0, str(otk_src_path))

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks, Query, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import uuid
from datetime import datetime
import threading
import queue
from typing import Optional

from . import models, schemas
from .models import init_db, get_db, Job, Statistics
from .config import UPLOAD_DIR, RESULTS_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from .data_validator import DataValidator
from .resource_manager import resource_manager
from .predictor_wrapper import run_prediction_job
from .i18n import get_text, SUPPORTED_LANGUAGES
from .cleanup import start_cleanup_scheduler
import markdown

# Get base path from environment variable or use empty string (root)
BASE_PATH = os.environ.get('OTK_BASE_PATH', '').rstrip('/')

def get_url(path: str) -> str:
    """Generate URL with base path prefix"""
    if not path.startswith('/'):
        path = '/' + path
    return f"{BASE_PATH}{path}"

app = FastAPI(
    title="OTK Prediction API",
    description="High-performance Scientific Computing API - ecDNA Prediction Service",
    version="1.0.0",
    root_path=BASE_PATH if BASE_PATH else None,
)

init_db()

# Start cleanup scheduler for old jobs
start_cleanup_scheduler()

# Mount static files with base path
static_path = f"{BASE_PATH}/static" if BASE_PATH else "/static"
app.mount(static_path, StaticFiles(directory=str(Path(__file__).parent.parent / "static")), name="static")

task_queue = queue.Queue()
active_jobs = {}

def worker_thread():
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break
            
            # Handle both old format (tuple) and new format (dict)
            if isinstance(task, tuple):
                job_id, input_path, _ = task
                preferred_model = None
            else:
                job_id = task.get('job_id')
                input_path = task.get('input_path')
                preferred_model = task.get('preferred_model')
                
            if job_id is None:
                continue
                
            from .models import SessionLocal
            db = SessionLocal()
            
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if not job:
                    continue
                
                job.status = "processing"
                job.updated_at = datetime.utcnow()
                db.commit()
                
                # Use preferred model if specified and valid
                if preferred_model and preferred_model in resource_manager.get_available_models():
                    model_name = preferred_model
                    model_path = resource_manager.get_available_models()[preferred_model]
                else:
                    model_name, model_path = resource_manager.get_best_model()
                
                device_type, gpu_id = resource_manager.select_device()
                
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
                
            except Exception as e:
                job.status = "failed"
                job.error_message = str(e)
                job.updated_at = datetime.utcnow()
                db.commit()
            finally:
                db.close()
                if job_id in active_jobs:
                    del active_jobs[job_id]
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")

worker = threading.Thread(target=worker_thread, daemon=True)
worker.start()

def _update_statistics(db, success: bool, stats: dict, device_type: str):
    from .models import Statistics
    
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    stat = db.query(Statistics).filter(Statistics.date >= today).first()
    
    if not stat:
        stat = Statistics(
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            total_rows_processed=0,
            total_samples_processed=0,
            avg_processing_time=0.0,
            gpu_jobs=0,
            cpu_jobs=0,
            validation_errors=0
        )
        db.add(stat)
        db.commit()
        db.refresh(stat)
    
    stat.total_jobs = (stat.total_jobs or 0) + 1
    
    if success:
        stat.completed_jobs = (stat.completed_jobs or 0) + 1
        stat.total_rows_processed = (stat.total_rows_processed or 0) + stats.get("total_rows", 0)
        stat.total_samples_processed = (stat.total_samples_processed or 0) + stats.get("total_samples", 0)
        
        current_avg = stat.avg_processing_time or 0.0
        current_count = (stat.completed_jobs or 1) - 1
        new_time = stats.get("processing_time", 0)
        
        if current_count > 0:
            stat.avg_processing_time = (current_avg * current_count + new_time) / (current_count + 1)
        else:
            stat.avg_processing_time = new_time
    else:
        stat.failed_jobs = (stat.failed_jobs or 0) + 1
    
    if device_type == "cuda":
        stat.gpu_jobs = (stat.gpu_jobs or 0) + 1
    else:
        stat.cpu_jobs = (stat.cpu_jobs or 0) + 1
    
    db.commit()

def get_nav_html(t, lang, current_page=""):
    """Generate navigation HTML for web pages"""
    nav_items = [
        (get_url("/"), t['nav_home'], "home"),
        (get_url("/web/upload"), t['nav_upload'], "upload"),
        (get_url("/web/jobs"), t['nav_jobs'], "jobs"),
        (get_url("/web/stats"), t['nav_stats'], "stats"),
        (get_url("/web/docs"), t['nav_docs'], "docs"),
    ]
    
    nav_html = '<nav style="background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 20px;">'
    for href, label, page in nav_items:
        active_style = 'font-weight: bold; color: #2196F3;' if page == current_page else ''
        nav_html += f'<a href="{href}?lang={lang}" style="margin-right: 20px; text-decoration: none; {active_style}">{label}</a>'
    nav_html += '</nav>'
    return nav_html

def get_footer_html(t, lang):
    """Generate footer HTML for web pages"""
    home_url = get_url("/")
    return f'''
    <footer style="margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; text-align: center; color: #666; font-size: 12px;">
        <p>{t['retention_notice']}</p>
        <p>OTK Prediction API | <a href="{home_url}?lang={lang}">{t['nav_home']}</a></p>
    </footer>
    '''

@app.get("/", response_class=HTMLResponse)
async def root(lang: str = Query(default="en", description="Language code")):
    t = get_text(lang)
    nav = get_nav_html(t, lang, "home")
    footer = get_footer_html(t, lang)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OTK Prediction API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .endpoint {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .method {{ color: #fff; padding: 3px 8px; border-radius: 3px; font-weight: bold; }}
            .get {{ background: #61affe; }}
            .post {{ background: #49cc90; }}
            code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
            .lang-switch {{ position: absolute; top: 20px; right: 20px; }}
            .lang-switch a {{ margin: 0 5px; text-decoration: none; padding: 5px 10px; border-radius: 3px; }}
            .lang-switch a.active {{ background: #2196F3; color: white; }}
            .lang-switch a:not(.active) {{ background: #f0f0f0; color: #333; }}
            .title-row {{ display: flex; align-items: center; gap: 15px; margin-bottom: 20px; }}
            .title-row h1 {{ margin: 0; }}
            .project-links {{ font-size: 14px; display: flex; gap: 8px; }}
            .project-links a {{ text-decoration: none; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 12px; transition: all 0.3s; }}
            .project-links a.otk {{ background: #2196F3; color: white; }}
            .project-links a.otk:hover {{ background: #1976D2; }}
            .project-links a.gcap {{ background: #4CAF50; color: white; }}
            .project-links a.gcap:hover {{ background: #45a049; }}
        </style>
    </head>
    <body>
        <div class="lang-switch">
            <a href="{get_url('/')}?lang=en" class="{'active' if lang == 'en' else ''}">English</a>
            <a href="{get_url('/')}?lang=zh" class="{'active' if lang == 'zh' else ''}">中文</a>
        </div>
        <div class="title-row">
            <h1>OTK Prediction API</h1>
            <div class="project-links">
                <a href="https://github.com/WangLabCSU/otk" target="_blank" class="otk">OTK</a>
                <a href="https://github.com/shixiangwang/gcap" target="_blank" class="gcap">GCAP</a>
            </div>
        </div>
        <p>{t['description']}</p>
        
        {nav}
        
        <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <strong>⚠️ {t['job_id_notice_title']}</strong><br>
            {t['job_id_notice']}
        </div>

        <h2>{t['api_endpoints']}</h2>

        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/v1/predict</code>
            <p>{t['submit_task']} (Async)</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/v1/predict-sync</code>
            <p>Synchronous prediction - upload and wait for results (for pipeline integration)</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/v1/jobs/{{job_id}}</code>
            <p>{t['query_status']}</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/v1/jobs/{{job_id}}/download</code>
            <p>{t['download_result']}</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/v1/validation-report/{{job_id}}</code>
            <p>{t['validation_report']}</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/v1/statistics</code>
            <p>{t['statistics']}</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/v1/health</code>
            <p>{t['health_check']}</p>
        </div>
        
        <h2>{t['web_interface']}</h2>
        <ul>
            <li><a href="{get_url('/web/upload')}?lang={lang}">{t['task_upload']}</a></li>
            <li><a href="{get_url('/web/jobs')}?lang={lang}">{t['task_list']}</a></li>
            <li><a href="{get_url('/web/stats')}?lang={lang}">{t['statistics_dashboard']}</a></li>
        </ul>

        {footer}
    </body>
    </html>
    """

@app.post("/api/v1/predict", response_model=schemas.JobResponse)
async def create_prediction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None, description="Model name to use (optional)"),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(status_code=400, detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported")
    
    # Validate model if specified
    if model and model not in resource_manager.get_available_models():
        available = list(resource_manager.get_available_models().keys())
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Available: {available}")
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds limit {MAX_FILE_SIZE / 1024 / 1024}MB")
    
    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    with open(upload_path, "wb") as f:
        f.write(contents)
    
    validator = DataValidator()
    validation_report = validator.validate(str(upload_path))
    
    job = Job(
        id=job_id,
        status="validating" if validation_report["is_valid"] else "validation_failed",
        original_filename=file.filename,
        uploaded_file_path=str(upload_path),
        validation_report=validation_report,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    if validation_report["is_valid"]:
        job.status = "pending"
        db.commit()
        active_jobs[job_id] = True
        # Pass model preference to worker
        task_queue.put({
            'job_id': job_id,
            'input_path': str(upload_path),
            'preferred_model': model
        })
    
    return job


@app.post("/api/v1/predict-sync")
async def predict_sync(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None, description="Model name to use (optional)"),
    timeout: int = Form(default=300, description="Timeout in seconds (default: 300)"),
    db: Session = Depends(get_db)
):
    """
    Synchronous prediction API - upload file and wait for results.
    
    This endpoint is designed for bioinformatics pipeline integration.
    The connection will be held open until prediction completes or timeout.
    
    Example:
        curl -X POST "http://localhost:8000/api/v1/predict-sync" \
             -F "file=@data.csv" \
             -F "model=advanced_ecdna" \
             --output results.csv
    """
    from .predictor_wrapper import run_prediction_job
    import asyncio
    import time
    
    if not file.filename.endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(status_code=400, detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported")
    
    # Validate model if specified
    available_models = resource_manager.get_available_models()
    if model and model not in available_models:
        available = list(available_models.keys())
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Available: {available}")
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds limit {MAX_FILE_SIZE / 1024 / 1024}MB")
    
    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    with open(upload_path, "wb") as f:
        f.write(contents)
    
    # Validate data
    validator = DataValidator()
    validation_report = validator.validate(str(upload_path))
    
    if not validation_report["is_valid"]:
        raise HTTPException(status_code=400, detail=f"Data validation failed: {validation_report['errors']}")
    
    # Create job record
    job = Job(
        id=job_id,
        status="processing",
        original_filename=file.filename,
        uploaded_file_path=str(upload_path),
        validation_report=validation_report,
    )
    db.add(job)
    db.commit()
    
    try:
        # Select model
        if model:
            model_path = available_models[model]
        else:
            # Use default or first available
            default_model = resource_manager.get_selected_model() or list(available_models.keys())[0]
            model_path = available_models[default_model]
        
        # Select device
        device_type, gpu_id = resource_manager.select_device()
        
        # Create output directory
        output_dir = RESULTS_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        
        # Run prediction synchronously
        start_time = time.time()
        result = run_prediction_job(
            job_id=job_id,
            input_path=str(upload_path),
            output_dir=str(output_dir),
            model_path=str(model_path),
            device_type=device_type,
            gpu_id=gpu_id
        )
        processing_time = time.time() - start_time
        
        if not result["success"]:
            job.status = "failed"
            job.error_message = result.get("error", "Unknown error")
            db.commit()
            raise HTTPException(status_code=500, detail=f"Prediction failed: {job.error_message}")
        
        # Update job status
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.model_used = result.get("model_used", "unknown")
        job.device_used = result.get("device_used", "unknown")
        job.processing_time = processing_time
        job.results_file_path = str(result["results_path"])
        db.commit()
        
        # Return results file
        return FileResponse(
            path=result["results_path"],
            filename=f"predictions_{job_id}.csv",
            media_type="text/csv"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/v1/jobs/recent")
async def get_recent_jobs(limit: int = 50, db: Session = Depends(get_db)):
    """Get recent jobs for the task list page"""
    from sqlalchemy import desc
    jobs = db.query(Job).order_by(desc(Job.created_at)).limit(limit).all()
    
    return [
        {
            "id": job.id,
            "status": job.status,
            "original_filename": job.original_filename,
            "model_used": job.model_used,
            "device_used": job.device_used,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
        for job in jobs
    ]

@app.get("/api/v1/jobs/{job_id}", response_model=schemas.JobStatusResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    progress = None
    if job.status == "processing":
        progress = 0.5
    elif job.status == "completed":
        progress = 1.0
    elif job.status == "pending":
        progress = 0.0
    
    return {
        "id": job.id,
        "status": job.status,
        "progress": progress,
        "message": job.error_message if job.status == "failed" else None,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
    }

@app.get("/api/v1/jobs/{job_id}/download")
async def download_results(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if not job.results_file_path or not Path(job.results_file_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        job.results_file_path,
        filename=f"predictions_{job_id}.csv",
        media_type="text/csv"
    )

@app.get("/api/v1/sample-file")
async def download_sample_file():
    """Download sample CSV file for testing"""
    sample_path = Path(__file__).parent.parent / "sample_data" / "sample.csv"
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")
    
    return FileResponse(
        str(sample_path),
        filename="sample.csv",
        media_type="text/csv"
    )

@app.get("/api/v1/validation-report/{job_id}", response_model=schemas.ValidationReport)
async def get_validation_report(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not job.validation_report:
        raise HTTPException(status_code=404, detail="Validation report not found")
    
    return job.validation_report

@app.get("/api/v1/statistics", response_model=schemas.StatisticsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    total_jobs = db.query(Job).count()
    completed_jobs = db.query(Job).filter(Job.status == "completed").count()
    failed_jobs = db.query(Job).filter(Job.status == "failed").count()
    pending_jobs = db.query(Job).filter(Job.status == "pending").count()
    processing_jobs = db.query(Job).filter(Job.status == "processing").count()
    
    stats = db.query(Statistics).order_by(Statistics.date.desc()).first()
    
    daily_stats = []
    for stat in db.query(Statistics).order_by(Statistics.date.desc()).limit(30).all():
        daily_stats.append({
            "date": stat.date.isoformat(),
            "total_jobs": stat.total_jobs,
            "completed_jobs": stat.completed_jobs,
            "failed_jobs": stat.failed_jobs,
        })
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "pending_jobs": pending_jobs,
        "processing_jobs": processing_jobs,
        "total_rows_processed": stats.total_rows_processed if stats else 0,
        "total_samples_processed": stats.total_samples_processed if stats else 0,
        "avg_processing_time": stats.avg_processing_time if stats else 0.0,
        "gpu_jobs": stats.gpu_jobs if stats else 0,
        "cpu_jobs": stats.cpu_jobs if stats else 0,
        "validation_errors": stats.validation_errors if stats else 0,
        "daily_stats": daily_stats,
    }

@app.get("/api/v1/health", response_model=schemas.HealthResponse)
async def health_check():
    import psutil
    
    system_stats = resource_manager.get_system_stats()
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "gpu_available": system_stats["gpu"]["available"],
        "gpu_count": system_stats["gpu"]["count"],
        "cpu_count": system_stats["cpu"]["count"],
        "active_jobs": len(active_jobs),
        "queue_size": task_queue.qsize(),
    }

@app.get("/api/v1/models", response_model=schemas.ModelsListResponse)
async def list_models():
    """List all available models"""
    models = resource_manager.get_model_list()
    return {
        "models": models,
        "default_model": resource_manager.get_selected_model(),
        "priority": resource_manager.get_system_stats()["models"]["priority"],
    }

@app.get("/api/v1/config", response_model=schemas.ConfigResponse)
async def get_config():
    """Get current API configuration"""
    from .config import (
        API_HOST, API_PORT, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
        MAX_CONCURRENT_JOBS, JOB_TIMEOUT, GPU_MEMORY_THRESHOLD, PREFER_GPU,
        MODEL_PRIORITY, DEFAULT_MODEL
    )
    
    return {
        "api": {
            "host": API_HOST,
            "port": API_PORT,
        },
        "upload": {
            "max_file_size": MAX_FILE_SIZE,
            "allowed_extensions": list(ALLOWED_EXTENSIONS),
        },
        "resources": {
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "job_timeout": JOB_TIMEOUT,
            "gpu_memory_threshold": GPU_MEMORY_THRESHOLD,
            "prefer_gpu": PREFER_GPU,
        },
        "models": {
            "priority": MODEL_PRIORITY,
            "default_model": DEFAULT_MODEL,
            "selected_model": resource_manager.get_selected_model(),
        },
    }

@app.get("/web/upload", response_class=HTMLResponse)
async def upload_page(lang: str = Query(default="en", description="Language code")):
    t = get_text(lang)
    nav = get_nav_html(t, lang, "upload")
    footer = get_footer_html(t, lang)
    
    # Get available models for the dropdown
    models = resource_manager.get_model_list()
    
    # Get default model from config
    from .config import DEFAULT_MODEL, MODEL_PRIORITY
    default_model_name = DEFAULT_MODEL if DEFAULT_MODEL else (MODEL_PRIORITY[0] if MODEL_PRIORITY else t['auto_select'])
    
    # Build model options with default selected
    model_options = f'<option value="">{t["auto_select"]} ({default_model_name})</option>'
    for model in models:
        selected = 'selected' if model["name"] == default_model_name else ''
        model_options += f'<option value="{model["name"]}" {selected}>{model["name"]}</option>'
    
    # Sample file path
    sample_file_path = Path(__file__).parent.parent / "sample_data" / "sample.csv"
    sample_file_exists = sample_file_path.exists()
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{t['upload_title']} - OTK API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            .upload-area {{ border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px; }}
            .upload-area:hover {{ border-color: #666; }}
            button {{ background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }}
            button:hover {{ background: #45a049; }}
            #result {{ margin-top: 20px; padding: 15px; border-radius: 5px; display: none; }}
            .success {{ background: #d4edda; color: #155724; }}
            .error {{ background: #f8d7da; color: #721c24; }}
            .lang-switch {{ position: absolute; top: 20px; right: 20px; }}
            .lang-switch a {{ margin: 0 5px; text-decoration: none; padding: 5px 10px; border-radius: 3px; }}
            .lang-switch a.active {{ background: #2196F3; color: white; }}
            .lang-switch a:not(.active) {{ background: #f0f0f0; color: #333; }}
            .model-select {{ margin: 15px 0; }}
            .model-select label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
            .model-select select {{ padding: 8px 12px; font-size: 14px; border-radius: 4px; border: 1px solid #ccc; min-width: 200px; }}
            .model-description {{ font-size: 12px; color: #666; margin-top: 5px; }}
            .sample-file {{ margin: 20px 0; padding: 15px; background: #e3f2fd; border-radius: 5px; }}
            .sample-file a {{ color: #2196F3; text-decoration: none; }}
            .sample-file a:hover {{ text-decoration: underline; }}
            .default-model-info {{ font-size: 12px; color: #666; margin-top: 5px; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="lang-switch">
            <a href="{get_url('/web/upload')}?lang=en" class="{'active' if lang == 'en' else ''}">English</a>
            <a href="{get_url('/web/upload')}?lang=zh" class="{'active' if lang == 'zh' else ''}">中文</a>
        </div>
        <h1>{t['upload_title']}</h1>
        
        {nav}
        
        <div class="sample-file">
            <strong>{t['sample_file']}:</strong> 
            {'<a href="' + get_url('/api/v1/sample-file') + '" download>📥 ' + t['download_sample'] + '</a>' if sample_file_exists else '<span style="color: #999;">Sample file not available</span>'}
        </div>
        
        <div class="upload-area">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv" required><br><br>
                
                <div class="model-select">
                    <label for="model">{t['select_model']}</label>
                    <select name="model" id="model">
                        {model_options}
                    </select>
                    <div class="model-description">{t['model_description']}</div>
                    <div class="default-model-info">{t['default_model_label']}: {default_model_name}</div>
                </div>
                
                <button type="submit">{t['submit']}</button>
            </form>
        </div>
        <div id="result"></div>
        
        {footer}
        
        <script>
            document.getElementById('uploadForm').onsubmit = async function(e) {{
                e.preventDefault();
                const formData = new FormData(this);
                const resultDiv = document.getElementById('result');
                
                // Show selected model in result
                const modelSelect = document.getElementById('model');
                const selectedModel = modelSelect.value || '{t['auto_select']}';
                
                try {{
                    const response = await fetch('{get_url('/api/v1/predict')}', {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await response.json();
                    
                    if (response.ok) {{
                        resultDiv.className = 'success';
                        resultDiv.innerHTML = `
                            <h3>{t['task_created']}</h3>
                            <p>Job ID: <code>${{data.id}}</code></p>
                            <p>{t['status']}: ${{data.status}}</p>
                            <p>{t['selected_model']}: ${{selectedModel}}</p>
                            <p><a href="{get_url('/web/jobs')}?lang={lang}">{t['view_task_list']}</a></p>
                        `;
                    }} else {{
                        resultDiv.className = 'error';
                        resultDiv.innerHTML = `<p>{t['error']}: ${{data.detail}}</p>`;
                    }}
                    resultDiv.style.display = 'block';
                }} catch (error) {{
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `<p>{t['upload_failed']}: ${{error.message}}</p>`;
                    resultDiv.style.display = 'block';
                }}
            }};
        </script>
    </body>
    </html>
    """

@app.get("/web/jobs", response_class=HTMLResponse)
async def jobs_page(lang: str = Query(default="en", description="Language code")):
    t = get_text(lang)
    nav = get_nav_html(t, lang, "jobs")
    footer = get_footer_html(t, lang)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{t['task_list_title']} - OTK API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 50px auto; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f5f5f5; font-weight: bold; }}
            .status-pending {{ color: #ff9800; }}
            .status-processing {{ color: #2196F3; }}
            .status-completed {{ color: #4CAF50; }}
            .status-failed {{ color: #f44336; }}
            .status-validating {{ color: #9c27b0; }}
            .btn {{ background: #2196F3; color: white; padding: 5px 10px; border: none; border-radius: 3px; cursor: pointer; text-decoration: none; display: inline-block; font-size: 12px; margin-right: 5px; }}
            .btn:hover {{ background: #1976D2; }}
            .btn:disabled {{ background: #ccc; cursor: not-allowed; }}
            .btn-download {{ background: #4CAF50; }}
            .btn-download:hover {{ background: #45a049; }}
            .lang-switch {{ position: absolute; top: 20px; right: 20px; }}
            .lang-switch a {{ margin: 0 5px; text-decoration: none; padding: 5px 10px; border-radius: 3px; }}
            .lang-switch a.active {{ background: #2196F3; color: white; }}
            .lang-switch a:not(.active) {{ background: #f0f0f0; color: #333; }}
            .job-id {{ font-family: monospace; font-size: 12px; color: #666; }}
            .stats-bar {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .stats-bar span {{ margin-right: 20px; }}
        </style>
    </head>
    <body>
        <div class="lang-switch">
            <a href="{get_url('/web/jobs')}?lang=en" class="{'active' if lang == 'en' else ''}">English</a>
            <a href="{get_url('/web/jobs')}?lang=zh" class="{'active' if lang == 'zh' else ''}">中文</a>
        </div>
        <h1>{t['task_list_title']}</h1>
        
        {nav}
        
        <div class="stats-bar" id="statsBar">
            <span><strong>{t['total_jobs']}:</strong> <span id="statTotal">-</span></span>
            <span><strong>{t['completed']}:</strong> <span id="statCompleted">-</span></span>
            <span><strong>{t['processing']}:</strong> <span id="statProcessing">-</span></span>
            <span><strong>{t['failed']}:</strong> <span id="statFailed">-</span></span>
            <span><strong>{t['pending']}:</strong> <span id="statPending">-</span></span>
        </div>
        
        <table id="jobsTable">
            <thead>
                <tr>
                    <th>Job ID</th>
                    <th>{t['filename']}</th>
                    <th>{t['status']}</th>
                    <th>{t['model']}</th>
                    <th>{t['device']}</th>
                    <th>{t['created_at']}</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
        
        {footer}
        
        <script>
            async function loadJobs() {{
                const tbody = document.querySelector('#jobsTable tbody');
                
                try {{
                    // Fetch recent jobs from a new endpoint
                    const response = await fetch('{get_url('/api/v1/jobs/recent')}');
                    const jobs = await response.json();
                    
                    // Update stats
                    const statsResponse = await fetch('{get_url('/api/v1/statistics')}');
                    const stats = await statsResponse.json();
                    document.getElementById('statTotal').textContent = stats.total_jobs;
                    document.getElementById('statCompleted').textContent = stats.completed_jobs;
                    document.getElementById('statProcessing').textContent = stats.processing_jobs;
                    document.getElementById('statFailed').textContent = stats.failed_jobs;
                    document.getElementById('statPending').textContent = stats.pending_jobs;
                    
                    if (jobs.length === 0) {{
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666;">{t['no_jobs']}</td></tr>';
                        return;
                    }}

                    tbody.innerHTML = jobs.map(job => {{
                        const statusClass = 'status-' + job.status;
                        const createdAt = new Date(job.created_at).toLocaleString();
                        // Mask job ID: show first 4 and last 4 chars only
                        const maskedId = job.id.substring(0, 4) + '****' + job.id.substring(job.id.length - 4);

                        return `
                            <tr>
                                <td><span class="job-id" title="${{job.id}}">${{maskedId}}</span></td>
                                <td>${{job.original_filename}}</td>
                                <td><span class="${{statusClass}}">${{job.status}}</span></td>
                                <td>${{job.model_used || '-'}}</td>
                                <td>${{job.device_used || '-'}}</td>
                                <td>${{createdAt}}</td>
                            </tr>
                        `;
                    }}).join('');

                }} catch (error) {{
                    console.error('Failed to load jobs:', error);
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #f44336;">{t['load_failed']}</td></tr>';
                }}
            }}
            loadJobs();
            setInterval(loadJobs, 5000);
        </script>
    </body>
    </html>
    """

@app.get("/web/stats", response_class=HTMLResponse)
async def stats_page(lang: str = Query(default="en", description="Language code")):
    t = get_text(lang)
    nav = get_nav_html(t, lang, "stats")
    footer = get_footer_html(t, lang)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{t['stats_title']} - OTK API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }}
            .stat-card {{ background: #f5f5f5; padding: 20px; border-radius: 10px; text-align: center; }}
            .stat-value {{ font-size: 36px; font-weight: bold; color: #2196F3; }}
            .stat-label {{ color: #666; margin-top: 10px; }}
            .lang-switch {{ position: absolute; top: 20px; right: 20px; }}
            .lang-switch a {{ margin: 0 5px; text-decoration: none; padding: 5px 10px; border-radius: 3px; }}
            .lang-switch a.active {{ background: #2196F3; color: white; }}
            .lang-switch a:not(.active) {{ background: #f0f0f0; color: #333; }}
        </style>
    </head>
    <body>
        <div class="lang-switch">
            <a href="{get_url('/web/stats')}?lang=en" class="{'active' if lang == 'en' else ''}">English</a>
            <a href="{get_url('/web/stats')}?lang=zh" class="{'active' if lang == 'zh' else ''}">中文</a>
        </div>
        <h1>{t['stats_title']}</h1>
        
        {nav}
        
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <div class="stat-value" id="totalJobs">-</div>
                <div class="stat-label">{t['total_jobs']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="completedJobs">-</div>
                <div class="stat-label">{t['completed']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="failedJobs">-</div>
                <div class="stat-label">{t['failed']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgTime">-</div>
                <div class="stat-label">{t['avg_time']}</div>
            </div>
        </div>
        
        {footer}
        
        <script>
            async function loadStats() {{
                try {{
                    const response = await fetch('{get_url('/api/v1/statistics')}');
                    const data = await response.json();
                    
                    document.getElementById('totalJobs').textContent = data.total_jobs;
                    document.getElementById('completedJobs').textContent = data.completed_jobs;
                    document.getElementById('failedJobs').textContent = data.failed_jobs;
                    document.getElementById('avgTime').textContent = data.avg_processing_time.toFixed(2);
                }} catch (error) {{
                    console.error('Failed to load stats:', error);
                }}
            }}
            loadStats();
            setInterval(loadStats, 30000);
        </script>
    </body>
    </html>
    """

@app.get("/web/docs", response_class=HTMLResponse)
async def docs_page(lang: str = Query(default="en", description="Language code")):
    """Render README.md as HTML documentation page"""
    t = get_text(lang)
    nav = get_nav_html(t, lang, "docs")
    footer = get_footer_html(t, lang)

    # Read and convert README.md to HTML
    readme_path = Path(__file__).parent.parent / "README.md"
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        # Convert markdown to HTML
        html_content = markdown.markdown(readme_content, extensions=['tables', 'fenced_code'])
    except Exception as e:
        html_content = f"<p>Error loading documentation: {e}</p>"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{t['docs_title']} - OTK API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; line-height: 1.6; }}
            h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
            h2 {{ color: #444; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            h3 {{ color: #555; }}
            code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: Consolas, monospace; }}
            pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            pre code {{ background: none; padding: 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #f5f5f5; font-weight: bold; }}
            a {{ color: #2196F3; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            ul, ol {{ padding-left: 25px; }}
            blockquote {{ border-left: 4px solid #2196F3; margin: 0; padding-left: 15px; color: #666; }}
            .lang-switch {{ position: absolute; top: 20px; right: 20px; }}
            .lang-switch a {{ margin: 0 5px; text-decoration: none; padding: 5px 10px; border-radius: 3px; }}
            .lang-switch a.active {{ background: #2196F3; color: white; }}
            .lang-switch a:not(.active) {{ background: #f0f0f0; color: #333; }}
            .docs-content {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="lang-switch">
            <a href="{get_url('/web/docs')}?lang=en" class="{'active' if lang == 'en' else ''}">English</a>
            <a href="{get_url('/web/docs')}?lang=zh" class="{'active' if lang == 'zh' else ''}">中文</a>
        </div>
        <h1>{t['docs_title']}</h1>

        {nav}

        <div class="docs-content">
            {html_content}
        </div>

        {footer}
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    from .config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
