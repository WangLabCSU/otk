import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

API_DIR = Path(__file__).parent.parent
OTK_BASE_DIR = API_DIR.parent.parent

CONFIG_FILE = API_DIR / "config.yml"

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

_config = load_config()

def get_config_value(path: str, default: Any = None, env_var: Optional[str] = None) -> Any:
    """Get configuration value from YAML or environment variable"""
    # Check environment variable first
    if env_var and env_var in os.environ:
        value = os.environ[env_var]
        # Try to convert to appropriate type
        if isinstance(default, bool):
            return value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default, int):
            return int(value)
        elif isinstance(default, float):
            return float(value)
        return value
    
    # Navigate config dict by path (e.g., "api.port")
    keys = path.split('.')
    value = _config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value if value is not None else default

# API Settings
API_HOST = get_config_value("api.host", "0.0.0.0", "API_HOST")
API_PORT = get_config_value("api.port", 8000, "API_PORT")
API_DEBUG = get_config_value("api.debug", False, "API_DEBUG")
API_TITLE = get_config_value("api.title", "OTK Prediction API")
API_DESCRIPTION = get_config_value("api.description", "High-performance Scientific Computing API")
API_VERSION = get_config_value("api.version", "1.0.0")

# Upload Settings
MAX_FILE_SIZE = get_config_value("upload.max_file_size", 100 * 1024 * 1024)
ALLOWED_EXTENSIONS = set(get_config_value("upload.allowed_extensions", [".csv"]))
UPLOAD_DIR = API_DIR / get_config_value("upload.upload_dir", "uploads")
RESULTS_DIR = API_DIR / get_config_value("upload.results_dir", "results")
LOGS_DIR = API_DIR / get_config_value("upload.logs_dir", "logs")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database Settings
DATABASE_URL = get_config_value("database.url", f"sqlite:///{API_DIR}/api.db", "DATABASE_URL")

# Redis Settings
REDIS_URL = get_config_value("redis.url", "redis://localhost:6379/0", "REDIS_URL")
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

# Model Settings
DEFAULT_MODEL = get_config_value("models.default_model", "baseline_mlp")
# Only search in otk_api/models directory for released models
MODEL_SEARCH_DIRS = get_config_value("models.search_dirs", [
    str(API_DIR / "models"),
])
MODEL_PRIORITY = get_config_value("models.priority", [
    "advanced_ecdna",
    "precision_focused",
    "transformer_optimized",
    "improved_v2_ecdna",
    "baseline_optimized",
])

# Resource Settings
MAX_CONCURRENT_JOBS = get_config_value("resources.max_concurrent_jobs", 100, "MAX_CONCURRENT_JOBS")
JOB_TIMEOUT = get_config_value("resources.job_timeout", 7200)
GPU_MEMORY_THRESHOLD = get_config_value("resources.gpu_memory_threshold", 0.8)
PREFER_GPU = get_config_value("resources.prefer_gpu", True)

# Retention Settings
RETENTION_DAYS = get_config_value("retention.days", 3)
CLEANUP_INTERVAL_HOURS = get_config_value("retention.cleanup_interval", 24)

# Validation Settings
# Minimal required columns for prediction (others can be auto-filled)
REQUIRED_COLUMNS = get_config_value("validation.required_columns", [
    "sample", "gene_id", "segVal"
])
CN_COLUMNS = get_config_value("validation.cn_columns", [f"CN{i}" for i in range(1, 20)])
OPTIONAL_COLUMNS = get_config_value("validation.optional_columns", [
    "minor_cn", "purity", "ploidy", "AScore", "pLOH", "cna_burden",
    "age", "gender", "y", "type", "intersect_ratio"
])
DEFAULT_COLUMN_VALUES = get_config_value("validation.default_values", {
    "minor_cn": 0,
    "purity": 0.8,
    "ploidy": 2.0,
    "AScore": 10.0,
    "pLOH": 0.1,
    "cna_burden": 0.2,
    "age": 60,
    "gender": 0,
    "intersect_ratio": 1.0,
})
VALID_CANCER_TYPES = get_config_value("validation.valid_cancer_types", [
    'BLCA', 'BRCA', 'CESC', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
    'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV',
    'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'THCA', 'UCEC', 'UVM'
])

# Logging Settings
LOG_LEVEL = get_config_value("logging.level", "INFO")
LOG_FORMAT = get_config_value("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = API_DIR / get_config_value("logging.file", "logs/api.log")

def get_model_search_paths() -> List[Path]:
    """Get list of model search paths (resolved to absolute paths)"""
    paths = []
    for dir_path in MODEL_SEARCH_DIRS:
        path = Path(dir_path)
        if path.is_absolute():
            paths.append(path)
        else:
            # Try relative to API dir first (for otk_api/models)
            api_path = API_DIR / dir_path
            if api_path.exists():
                paths.append(api_path)
            else:
                # Fallback to OTK base dir
                paths.append(OTK_BASE_DIR / dir_path)
    return paths

def reload_config():
    """Reload configuration from file"""
    global _config
    _config = load_config()
