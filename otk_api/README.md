# OTK Prediction API

High-performance Scientific Computing API - ecDNA (extrachromosomal DNA) Prediction Service based on the [OTK](https://github.com/WangLabCSU/otk) and [GCAP](https://github.com/shixiangwang/gcap) projects.

## Features

- **Intelligent Resource Scheduling**: Automatically selects the best model and available GPU/CPU resources
- **Model Management**: Auto-discovers models from `models/` directory, configurable via API
- **Unified Configuration**: All settings managed through a single YAML configuration file
- **Data Validation**: Comprehensive integrity checks when uploading data
- **Asynchronous & Synchronous Processing**: Supports both async tasks and sync predictions for pipeline integration
- **Statistics**: Task count, processing time, resource usage, and more
- **Web Interface**: User-friendly interface for task upload, status viewing, and management
- **REST API**: Complete API interface supporting curl and other tools
- **Multi-language Support**: English and Chinese (default: English)
- **Job Record Retention**: Task metadata kept permanently, result files retained for 3 days
- **Security**: Job IDs are masked in web interface for privacy protection

## Quick Start

### 1. Install Dependencies

```bash
cd otk/otk_api
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yml` to customize settings:

```yaml
api:
  host: "0.0.0.0"
  port: 8000

models:
  default_model: null  # Set to auto-select, or specify a model name
  search_dirs:
    - "otk/output_advanced_ecdna"
    - "otk/output_precision_focused"
```

### 3. Model Setup

Place your trained models in the `models/` directory:

```
otk_api/models/
├── baseline/                    # Model name: "baseline"
│   ├── best_model.pth
│   └── config.yml
└── your_custom_model/           # Model name: "your_custom_model"
    ├── best_model.pth
    └── config.yml
```

Models are auto-discovered from subdirectories containing `best_model.pth`.

### 4. Start API Service

```bash
cd otk/otk_api
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or use the provided startup script:

```bash
cd otk/otk_api
./start_api.sh
```

## API Usage Examples

### Submit Prediction Task

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@data.csv"
```

### List Available Models

```bash
curl "http://localhost:8000/api/v1/models"
```

Response:
```json
{
  "models": [
    {"name": "advanced_ecdna", "path": "...", "exists": true, "is_default": false},
    {"name": "precision_focused", "path": "...", "exists": true, "is_default": false}
  ],
  "default_model": null,
  "priority": ["advanced_ecdna", "precision_focused", "..."]
}
```

### Get API Configuration

```bash
curl "http://localhost:8000/api/v1/config"
```

### Query Task Status

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}"
```

### Download Prediction Results

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/download" -o results.csv
```

### Get Data Validation Report

```bash
curl "http://localhost:8000/api/v1/validation-report/{job_id}"
```

### Get Statistics

```bash
curl "http://localhost:8000/api/v1/statistics"
```

### Health Check

```bash
curl "http://localhost:8000/api/v1/health"
```

## Configuration

All settings are managed through `config.yml`:

```yaml
# API Server Settings
api:
  host: "0.0.0.0"
  port: 8000

# File Upload Settings
upload:
  max_file_size: 104857600  # 100MB
  allowed_extensions: [".csv"]

# Model Settings
models:
  default_model: null  # null for auto-select, or specify model name
  search_dirs:
    - "otk/output_advanced_ecdna"
    - "otk/output_precision_focused"
  priority:
    - "advanced_ecdna"
    - "precision_focused"

# Resource Management
resources:
  max_concurrent_jobs: 4
  job_timeout: 3600
  gpu_memory_threshold: 0.8
  prefer_gpu: true

# Data Retention
retention:
  days: 3  # Jobs and results are retained for 3 days
  cleanup_interval: 24  # Cleanup runs every 24 hours
```

Environment variables can override YAML settings:
- `API_HOST` / `API_PORT`: Server settings
- `DATABASE_URL`: Database connection
- `REDIS_URL`: Redis connection
- `MAX_CONCURRENT_JOBS`: Concurrent job limit

## Web Interface

- Homepage: http://localhost:8000/
- Task Upload: http://localhost:8000/web/upload
- Task List: http://localhost:8000/web/jobs
- Statistics Dashboard: http://localhost:8000/web/stats

### Language Switching

Add `?lang=en` or `?lang=zh` parameter to any URL:
- English: http://localhost:8000/?lang=en
- Chinese: http://localhost:8000/?lang=zh

### Data Retention

- **Result files**: Automatically deleted after **3 days**
- **Job records**: Kept **permanently** for audit and tracking purposes
- Please download your results promptly or save your Job ID for async tasks

## Data Format Requirements

The uploaded CSV file must contain the following columns:

**Required Columns:**
- `sample`: Sample ID
- `gene_id`: Gene ID
- `segVal`: Copy number
- `minor_cn`: Minor copy number
- `purity`: Tumor purity
- `ploidy`: Ploidy
- `AScore`: Aneuploidy score
- `pLOH`: Loss of heterozygosity ratio
- `cna_burden`: CNA burden
- `CN1-CN19`: Copy number signatures

**Optional Columns:**
- `age`: Age
- `gender`: Gender
- `type`: Cancer type
- `intersect_ratio`: Overlap ratio (default: 1.0 if not provided)

## Project Structure

```
otk/otk_api/
├── api/
│   ├── __init__.py
│   ├── main.py              # Main application
│   ├── models.py            # Database models
│   ├── schemas.py           # Pydantic models
│   ├── i18n.py              # Internationalization
│   ├── data_validator.py    # Data validation
│   ├── resource_manager.py  # Resource management
│   ├── predictor_wrapper.py # otk predictor wrapper
│   ├── cleanup.py           # Cleanup scheduler
│   └── config.py            # Configuration loader
├── models/                  # ML models directory
│   └── baseline/            # Example model
│       ├── best_model.pth
│       └── config.yml
├── static/                  # Static files
├── templates/               # HTML templates
├── uploads/                 # Uploaded files
├── results/                 # Prediction results
├── logs/                    # Logs
├── config.yml               # Main configuration file
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start_api.sh             # Startup script
└── README.md
```

## Model Selection Strategy

The API selects models in the following priority:
1. Model specified in prediction request (`model` parameter)
2. Default model from config (`models.default_model`)
3. Priority order from config (`models.priority`)

## GPU/CPU Scheduling Strategy

- Prioritize GPU for inference (configurable via `prefer_gpu`)
- Monitor GPU memory usage, automatically switch to CPU when threshold is exceeded
- Support multi-GPU load balancing
