# OTK Prediction API

High-performance Scientific Computing API for ecDNA (extrachromosomal DNA) Prediction Service based on the [OTK](https://github.com/WangLabCSU/otk) and [GCAP](https://github.com/shixiangwang/gcap) projects.

## 🌐 Public API Address

**Production API**: http://biotree.top:38123/otk/

**API Base URL**: http://biotree.top:38123/otk/api/v1/

## ✨ Features

- **Intelligent Resource Scheduling**: Automatically selects optimal model and available resources
- **Model Management**: Auto-discovers models from `models/` directory
- **Data Validation**: Comprehensive integrity checks during upload
- **Asynchronous & Synchronous Processing**: Supports both async tasks and sync predictions
- **Real-time Statistics**: Task counts, processing times, resource usage
- **User-friendly Web Interface**: For task upload, status viewing, and management
- **Complete REST API**: Supports curl and other HTTP clients
- **Multi-language Support**: English and Chinese interfaces
- **Job Record Management**: Task metadata retained permanently, results for 3 days
- **Security**: Job IDs are masked in web interface for privacy

## 🚀 Quick Start

### Using the Public API

You can immediately start using the public API without any installation:

```bash
# Health check
curl http://biotree.top:38123/otk/api/v1/health

# Submit prediction (async)
curl -X POST "http://biotree.top:38123/otk/api/v1/predict" \
  -F "file=@your_data.csv"

# Submit prediction (sync)
curl -X POST "http://biotree.top:38123/otk/api/v1/predict-sync" \
  -F "file=@your_data.csv"
```

### Running Locally

1. **Install Dependencies**
   ```bash
   cd otk/otk_api
   pip install -r requirements.txt
   ```

2. **Start the API**
   ```bash
   cd otk/otk_api
   ./start_api.sh
   ```

3. **Access**
   - API: http://localhost:8000/api/v1/
   - Web Interface: http://localhost:8000/

## 📡 API Documentation

### 1. Health Check

**Endpoint**: `GET /api/v1/health`

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_available": false,
  "gpu_count": 0,
  "cpu_count": 192,
  "active_jobs": 0,
  "queue_size": 0
}
```

### 2. Submit Prediction (Async)

**Endpoint**: `POST /api/v1/predict`

**Parameters**:
- `file`: CSV file with prediction data

**Response**:
```json
{
  "id": "af0e5298-b326-40ca-83b5-76f54ad212e6",
  "status": "pending",
  "created_at": "2026-02-12T09:54:25.495083",
  "validation_report": {
    "is_valid": true,
    "errors": [],
    "warnings": ["Optional column missing: intersect_ratio, using default value 1.0"]
  }
}
```

### 3. Submit Prediction (Sync)

**Endpoint**: `POST /api/v1/predict-sync`

**Parameters**:
- `file`: CSV file with prediction data

**Response**:
- Returns CSV file directly for immediate use in pipelines

### 4. Get Task Status

**Endpoint**: `GET /api/v1/jobs/{job_id}`

**Response**:
```json
{
  "id": "af0e5298-b326-40ca-83b5-76f54ad212e6",
  "status": "completed",
  "progress": 1.0,
  "completed_at": "2026-02-12T09:54:26.292634"
}
```

### 5. Download Results

**Endpoint**: `GET /api/v1/jobs/{job_id}/download`

**Response**:
- Returns CSV file with prediction results

### 6. Get Statistics

**Endpoint**: `GET /api/v1/statistics`

**Response**:
```json
{
  "total_jobs": 28,
  "completed_jobs": 14,
  "failed_jobs": 13,
  "avg_processing_time": 0.605,
  "cpu_jobs": 14,
  "gpu_jobs": 5
}
```

## 📊 Data Format Requirements

### Required Columns

Your CSV file must include these columns:

| Column         | Description                          |
|----------------|--------------------------------------|
| `sample`       | Sample ID                            |
| `gene_id`      | Gene identifier                      |
| `segVal`       | Segment value                        |
| `minor_cn`     | Minor copy number                    |
| `purity`       | Tumor purity                         |
| `ploidy`       | Ploidy level                         |
| `AScore`       | A-score value                        |
| `pLOH`         | Loss of heterozygosity probability   |
| `cna_burden`   | Copy number alteration burden        |
| `age`          | Sample age                           |
| `gender`       | Gender (0/1 or Male/Female)          |
| `type`         | Cancer type                          |
| `CN1` to `CN19` | Chromosome copy numbers             |

### Optional Columns

| Column           | Description                          | Default Value |
|------------------|--------------------------------------|---------------|
| `intersect_ratio` | Intersection ratio                  | 1.0           |
| `y`              | Ground truth label (for training)    | N/A           |

### Example Data

```csv
sample,gene_id,segVal,minor_cn,purity,ploidy,AScore,pLOH,cna_burden,age,gender,type,intersect_ratio,CN1,CN2,CN3,CN4,CN5,CN6,CN7,CN8,CN9,CN10,CN11,CN12,CN13,CN14,CN15,CN16,CN17,CN18,CN19
TCGA-TEST-01,ENSG00000284662,3.2,1.1,0.85,2.8,0.75,0.45,0.3,65,1,LUSC,1.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9
```

## 🎯 Prediction Output

### Output Format

The prediction result CSV includes:

| Column                         | Description                          |
|--------------------------------|--------------------------------------|
| `sample`                       | Sample ID                            |
| `gene_id`                      | Gene identifier                      |
| `prediction_prob`              | Probability of ecDNA occurrence      |
| `prediction`                   | Binary prediction (0=no, 1=yes)      |
| `sample_level_prediction_label` | Overall sample prediction label       |
| `sample_level_prediction`      | Overall sample prediction (0/1)      |

### Example Output

```csv
sample,gene_id,prediction_prob,prediction,sample_level_prediction_label,sample_level_prediction
TCGA-TEST-01,ENSG00000284662,0.000279,0,nofocal,0
TCGA-TEST-01,ENSG00000187634,0.002650,0,nofocal,0
TCGA-TEST-01,ENSG00000243073,0.000036,0,nofocal,0
```

## 🌐 Web Interface

The API includes a user-friendly web interface:

### Access
- **Homepage**: http://biotree.top:38123/otk/
- **Task Upload**: http://biotree.top:38123/otk/web/upload
- **Task List**: http://biotree.top:38123/otk/web/jobs
- **Statistics**: http://biotree.top:38123/otk/web/stats

### Language Support
- Add `?lang=en` for English: http://biotree.top:38123/otk/?lang=en
- Add `?lang=zh` for Chinese: http://biotree.top:38123/otk/?lang=zh

## 📁 Project Structure

```
otk_api/
├── api/                  # API implementation
│   ├── main.py           # FastAPI application
│   ├── predictor_wrapper.py  # Prediction job handler
│   └── routes/           # API endpoints
├── config.yml           # Configuration file
├── models/              # Model storage
│   └── baseline/         # Example model
├── uploads/              # Uploaded files
├── results/              # Prediction results
├── logs/                 # Log files
├── start_api.sh          # Startup script
└── README.md             # This documentation
```

## ⚠️ Important Notes

1. **Job ID Security**: Save your Job ID securely for async tasks. It's needed to query status and download results.

2. **Data Retention**: 
   - **Result files**: Automatically deleted after 3 days
   - **Job records**: Kept permanently for audit purposes

3. **File Size Limit**: Maximum upload size is 100MB

4. **Processing Time**: Depends on data size and server load, typically 1-5 seconds per sample

5. **Error Handling**: If you receive an error, check your data format and try again

## 🛠️ Troubleshooting

### Common Issues

1. **File Upload Errors**
   - Ensure your file is a valid CSV
   - Check that all required columns are present
   - Verify file size is under 100MB

2. **Prediction Failed**
   - Check server logs for detailed error messages
   - Verify your data format matches requirements
   - Try with a smaller dataset first

3. **API Unresponsive**
   - Check if the server is running
   - Verify network connectivity
   - Try the health check endpoint

## 📞 Support

For questions or issues:

1. **GitHub Issues**: [OTK Repository](https://github.com/WangLabCSU/otk/issues)
2. **Email**: Contact the maintainers
3. **Documentation**: This README and API endpoints

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Last Updated**: February 12, 2026
**Version**: 1.0.0
**Maintainers**: Wang Lab @ CSU
