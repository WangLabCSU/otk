import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import logging

from .models import SessionLocal, Job
from .config import RETENTION_DAYS, UPLOAD_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


def cleanup_old_jobs():
    """Clean up job files older than retention period, but keep database records"""
    db = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=RETENTION_DAYS)

        # Find old jobs
        old_jobs = db.query(Job).filter(Job.created_at < cutoff_date).all()

        cleaned_count = 0
        for job in old_jobs:
            try:
                # Delete uploaded file
                if job.uploaded_file_path:
                    upload_path = Path(job.uploaded_file_path)
                    if upload_path.exists():
                        upload_path.unlink()
                        logger.info(f"Deleted old upload: {upload_path}")
                        job.uploaded_file_path = None  # Clear path in DB

                # Delete results directory
                if job.results_file_path:
                    results_dir = Path(job.results_file_path).parent
                    if results_dir.exists():
                        shutil.rmtree(results_dir)
                        logger.info(f"Deleted old results: {results_dir}")
                    job.results_file_path = None  # Clear path in DB

                # Mark job as cleaned (files removed)
                job.status = "cleaned"
                cleaned_count += 1

            except Exception as e:
                logger.error(f"Error cleaning up job {job.id}: {e}")

        db.commit()

        if cleaned_count > 0:
            logger.info(f"Cleaned up files for {cleaned_count} old jobs (older than {RETENTION_DAYS} days), database records retained")

        return cleaned_count

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        db.rollback()
        return 0
    finally:
        db.close()


def start_cleanup_scheduler():
    """Start background thread for periodic cleanup"""
    import threading
    import time
    from .config import CLEANUP_INTERVAL_HOURS
    
    def cleanup_worker():
        while True:
            try:
                time.sleep(CLEANUP_INTERVAL_HOURS * 3600)  # Sleep for interval
                cleanup_old_jobs()
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    # Run initial cleanup
    cleanup_old_jobs()
    
    # Start background thread
    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()
    logger.info(f"Cleanup scheduler started (interval: {CLEANUP_INTERVAL_HOURS} hours, retention: {RETENTION_DAYS} days)")
