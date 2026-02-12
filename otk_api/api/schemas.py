from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class JobCreate(BaseModel):
    pass

class JobResponse(BaseModel):
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    original_filename: str
    model_used: Optional[str] = None
    device_used: Optional[str] = None
    total_rows: Optional[int] = None
    total_samples: Optional[int] = None
    total_genes: Optional[int] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    validation_report: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

class JobStatusResponse(BaseModel):
    id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

class ValidationReport(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]
    column_status: Dict[str, str]
    data_summary: Dict[str, Any]

class StatisticsResponse(BaseModel):
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    processing_jobs: int
    
    total_rows_processed: int
    total_samples_processed: int
    
    avg_processing_time: float
    
    gpu_jobs: int
    cpu_jobs: int
    
    validation_errors: int
    
    daily_stats: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    gpu_available: bool
    gpu_count: int
    cpu_count: int
    active_jobs: int
    queue_size: int

class ModelInfo(BaseModel):
    name: str
    path: str
    exists: bool
    is_default: bool

class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    default_model: Optional[str]
    priority: List[str]

class ModelSelectRequest(BaseModel):
    model_name: str

class ModelSelectResponse(BaseModel):
    success: bool
    message: str
    selected_model: Optional[str]
    available_models: List[str]

class ConfigResponse(BaseModel):
    api: Dict[str, Any]
    upload: Dict[str, Any]
    resources: Dict[str, Any]
    models: Dict[str, Any]
