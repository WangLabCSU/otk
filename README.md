# otk: ecDNA Analysis Toolkit

[![PyPI version](https://badge.fury.io/py/otk-ecdna.svg)](https://pypi.org/project/otk-ecdna/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)

**otk** (ecDNA Analysis Toolkit) is a machine learning toolkit for predicting extrachromosomal DNA (ecDNA) cargo genes. It classifies genes at the gene level (ecDNA cargo vs. non-ecDNA) and identifies focal amplification types at the sample level (nofocal, noncircular, circular/ecDNA).

Based on the paper: Wang, S., et al. (2024). Machine learning-based extrachromosomal DNA identification in large-scale cohorts reveals its clinical implications in cancer. Nature Communications.

## Core Features

- Deep learning-based ecDNA cargo gene prediction at gene level
- Sample-level focal amplification type classification (nofocal/noncircular/circular)
- Multiple pre-trained models (XGBoost, Neural Networks, TabPFN)
- Efficient command-line interface for training and prediction
- GPU acceleration support
- Pre-trained models ready to use after pip install
- RESTful API for web service deployment
- Chinese mirror support for large model downloads
- Copy-number preprocessing for ASCAT, FACETS, and Sequenza results

See [Copy-number preprocessing](#copy-number-preprocessing) for supported inputs and preparation steps.

## Installation

### From PyPI (Recommended)

```bash
pip install otk-ecdna
```

This installs the `otk` CLI command and all pre-trained models (except TabPFN which is ~275MB and needs separate download).

### Download Large Models

The TabPFN model (~275MB) is hosted on GitHub Release:

```bash
# List available large models
otk download --list

# Download TabPFN model
otk download --model tabpfn
```

### From Source

```bash
git clone https://github.com/WangLabCSU/otk.git
cd otk
pip install -e .
```

## Quick Start

```bash
# Check installation
otk --version

# List available models
otk models

# Run prediction (example)
otk predict --input data.csv --output predictions.csv --model xgb_new

# Start API server
otk api --port 8000
```

## CLI Commands

### Model Management

```bash
# List all available models with performance metrics
otk models

# Analyze a specific model
otk analyze --model xgb_new

# Generate model configuration
otk config generate --model xgb_new
```

### Training

```bash
# Train single model
otk train --model xgb_new --gpu 0

# Train neural network model
otk train --model transformer --gpu 0

# Train all models sequentially
otk train --all --gpu 0

# Train all models in parallel on multiple GPUs
otk train --all --parallel --gpus 0,1,2,3

# CPU-only training
otk train --model xgb_new --gpu -1
```

### Prediction

```bash
# Basic prediction
otk predict --input data.csv --output predictions.csv --model xgb_new

# With GPU acceleration
otk predict -i data.csv -o results/ -m transformer --gpu 0

# With custom threshold
otk predict -i data.csv -o predictions.csv -m xgb_new --threshold 0.5
```

### API Server

```bash
# Start API with default settings (base path /otk)
otk api

# Custom port
otk api --port 8080

# Serve at root (no base path)
otk api --base-path ""

# Development mode with auto-reload
otk api --reload

# Multiple workers
otk api --workers 4
```

### Model Download

```bash
# List large models requiring download
otk download --list

# Download TabPFN model
otk download --model tabpfn

# Force re-download
otk download --model tabpfn --force
```

## Copy-number preprocessing

OTK can prepare gene-level prediction inputs from copy-number results generated
by the R packages [ASCAT](https://github.com/VanLoo-lab/ascat),
[FACETS](https://github.com/mskcc/facets), and
[Sequenza](https://cran.r-project.org/package=sequenza). Run the appropriate
tool separately; OTK does not run the R analyses. ASCAT and FACETS results must
be exported in the formats described below. Sequenza can produce the required
files directly.

Place the paired result files in one directory and pass that directory to OTK.
Only one tool's result family may be present in a directory; files are discovered
in that directory, not in its subdirectories. Multiple batches or samples of
the same family may be included.

| Tool | Segment file | Paired sample-parameter file |
|------|--------------|------------------------------|
| ASCAT | `<batch>_ascat_segments.csv` | `<batch>_ascat_pp.csv` |
| FACETS | `<sample>_facets_segments.csv` | `<sample>_facets_pp.csv` |
| Sequenza | `<sample>_segments.txt` | `<sample>_alternative_solutions.txt` |

The two files in a pair must have exactly the same prefix and be in the same
directory. Use a unique sample identifier across the input files.

```bash
otk predict \
  --input results_dir \
  --output predictions.csv \
  --model xgb_new \
  --gene-build hg38 \
  --extra-info samples.csv \
  --overlap 1.0
```

`--extra-info` is optional. `--gene-build` selects reference coordinates
(`hg19`, `hg38`, or `mm10`) and defaults to `hg38`; it must match the genome
assembly used by the upstream analysis. `--overlap` is the minimum
fraction of a gene covered by a segment, from 0 to 1; the default of 1.0
requires full gene coverage.

The optional metadata CSV can contain one row per sample, with a `sample`
column and any of `age`, `gender`, `type`, or one-hot `type_*` columns. Sample
identifiers must match those in the results. For example:

```csv
sample,age,gender,type
sample1,65,XX,LUAD
sample2,59,XY,BRCA
```

OTK also accepts an already gene-level `.csv` or `.tsv` file instead of a
results directory; it is passed through without segment-to-gene annotation.
See [Input Data Format](#input-data-format) for the required columns and
other format details.
For a gene-level file or DataFrame, cancer type information belongs in the
input itself: `extra_info` must not contain `type` or `type_*` columns.

In Python, `predict_input` also accepts a pandas DataFrame, and `extra_info`
can be a DataFrame or a CSV path:

```python
from otk import predict_input
from otk_preprocess import run_preprocessing_workflow

gene_data = run_preprocessing_workflow(
    "results_dir", gene_build="hg38", extra_info="samples.csv", overlap=1.0
)
predictions = predict_input(
    "results_dir", model="xgb_new", gene_build="hg38",
    extra_info="samples.csv", overlap=1.0
)
```

`run_preprocessing_workflow` returns gene-level input without running a model.

### Prepare ASCAT files

Follow the [ASCAT instructions](https://github.com/VanLoo-lab/ascat) to install
the R package and prepare LogR and BAF inputs and suitable correction files
for the assay and reference assembly. The following illustrates a matched
tumor/normal analysis; replace the paths and sample sex with your own values:

```r
library(ascat)
library(data.table)

ascat_data <- ascat.loadData(
  Tumor_LogR_file = "Tumor_LogR.txt",
  Tumor_BAF_file = "Tumor_BAF.txt",
  Germline_LogR_file = "Germline_LogR.txt",
  Germline_BAF_file = "Germline_BAF.txt",
  gender = "XX",
  genomeVersion = "hg19"
)
ascat_data <- ascat.correctLogR(
  ascat_data,
  GCcontentfile = "GC_example.txt",
  replictimingfile = "RT_example.txt"
)
ascat_data <- ascat.aspcf(ascat_data)
ascat_output <- ascat.runAscat(ascat_data, gamma = 1, write_segments = TRUE)
```

Here `gamma = 1` is for high-throughput sequencing data. For targeted
sequencing, consult ASCAT's guidance on `isTargetedSeq` and the segmentation
penalty; provide a gender value for each sample in a batch. ASCAT can export
separate segment files for each sample. For easier review of large batches, we
recommend the following export: combine all sample segments in one CSV file and
save their purity and ploidy in a paired CSV file.

```r
dir.create("ascat_results", showWarnings = FALSE)
fwrite(as.data.table(ascat_output$segments),
       "ascat_results/batch1_ascat_segments.csv")

sample_ids <- names(ascat_output$purity)
purity_ploidy <- data.table(
  sample = sample_ids,
  purity = as.numeric(ascat_output$purity[sample_ids]),
  ploidy = as.numeric(ascat_output$ploidy[sample_ids])
)
fwrite(purity_ploidy, "ascat_results/batch1_ascat_pp.csv")
```

The segments CSV must have `sample`, `chr`, `startpos`, `endpos`, `nMajor`,
and `nMinor` columns. The parameter CSV must have `sample`, `purity`, and
`ploidy`. A pair may contain multiple samples, but both files must contain
the same sample IDs; each sample must occur only once in the parameter file
and in only one batch. Inspect failed ASCAT fits before exporting.

### Prepare FACETS files

Install FACETS and its required `pctGCdata` package as described in the
[FACETS repository](https://github.com/mskcc/facets). The following example
uses the `data.table` R package for CSV export. First generate a `snp-pileup`
input according to the FACETS documentation, then analyze one tumor/normal
pair:

```r
library(facets)
library(pctGCdata)
library(data.table)

rcmat <- readSnpMatrix("sample1.snp_pileup.csv.gz")
preprocessed <- preProcSample(rcmat, gbuild = "hg38")
segmented <- procSample(preprocessed, cval = 300)
fit <- emcncf(segmented)
```

Choose the genome build and FACETS parameters for your data. After reviewing
the fit, export one pair of files per sample:

```r
dir.create("facets_results", showWarnings = FALSE)
segments <- as.data.table(fit$cncf)[
  , c("chrom", "start", "end", "tcn.em", "lcn.em"), with = FALSE
]
segments <- segments[end > start]
fwrite(segments, "facets_results/sample1_facets_segments.csv")
fwrite(data.table(purity = fit$purity, ploidy = fit$ploidy),
       "facets_results/sample1_facets_pp.csv")
```

The segment file requires `chrom`, `start`, `end`, `tcn.em`, and `lcn.em`.
The parameter file requires `purity` and `ploidy` and exactly one data row.
OTK derives the sample ID from the filename prefix (`sample1` in this example).

### Prepare Sequenza files

Use the [Sequenza package documentation and vignette](https://cran.r-project.org/package=sequenza)
to prepare a `.seqz` file, including the appropriate reference assembly and
GC information. For example, after producing a `sample1.seqz.gz` file:

```r
library(sequenza)

sequenza_data <- sequenza.extract("sample1.seqz.gz", assembly = "hg19")
fit <- sequenza.fit(sequenza_data)
dir.create("sequenza_results", showWarnings = FALSE)
sequenza.results(
  sequenza_data, cp.table = fit,
  sample.id = "sample1", out.dir = "sequenza_results"
)
```

Review the fit and adjust analysis parameters, including sample sex, as
appropriate. `sequenza.results` generates the two tab-separated files OTK
requires: `sample1_segments.txt` and `sample1_alternative_solutions.txt`.
The former must contain `chromosome`, `start.pos`, `end.pos`, `CNt`, and `B`.
OTK uses the  alternative-solutions file to get purity and ploidy, respectively. 
The filename prefix is the sample ID.

## Data Format

### Input Data Format

Already gene-level input data should be in CSV or TSV format. For copy-number
result directories, see [Copy-number preprocessing](#copy-number-preprocessing).

**Minimal required columns:**

| Column | Description |
|--------|-------------|
| `sample` | Tumor sample ID |
| `gene_id` | Gene ID (e.g., ENSG00000284662) |
| `segVal` | Total gene copy number |

**Auto-filled columns (defaults applied if missing):**

| Column | Default | Description |
|--------|---------|-------------|
| `minor_cn` | 0 | Minor copy number |
| `intersect_ratio` | 1.0 | Segment-gene overlap ratio |
| `purity` | 0.8 | Tumor purity |
| `ploidy` | 2.0 | Genome ploidy |
| `AScore` | 10.0 | Aneuploidy score |
| `pLOH` | 0.1 | LOH proportion |
| `cna_burden` | 0.2 | CNA burden |
| `CN1-CN19` | 0.05 each | Copy number signatures |
| `type` | - | Cancer type → auto-converts to `type_*` columns |

**Automatically generated features (from gene_id matching):**

| Column | Description |
|--------|-------------|
| `freq_Linear` | Prior frequency in linear amplifications |
| `freq_BFB` | Prior frequency in BFB events |
| `freq_Circular` | Prior frequency in ecDNA |
| `freq_HR` | Prior frequency in HR events |

**Training data requires:**

| Column | Description |
|--------|-------------|
| `y` | Binary label (1=ecDNA cargo gene, 0=not) |

**Supported cancer types (24):**
BLCA, BRCA, CESC, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LGG, LIHC, LUAD, LUSC, OV, PRAD, READ, SARC, SKCM, STAD, THCA, UCEC, UVM

### Output Format

| Column | Description |
|--------|-------------|
| `sample` | Sample ID |
| `gene_id` | Gene ID |
| `prediction_prob` | Probability of ecDNA (0-1) |
| `prediction` | Binary classification (0/1) |
| `sample_level_prediction_label` | Sample type: nofocal/noncircular/circular |
| `sample_level_prediction` | Sample type code (0/1/2) |

Sample classification rules:
- `circular` (2): Any gene predicted as ecDNA cargo
- `noncircular` (1): No ecDNA but segVal > ploidy + 2
- `nofocal` (0): Otherwise

## Available Models

| Model | Type | Test auPRC | Description |
|-------|------|------------|-------------|
| xgb_new | XGBoost | 0.8339 | Optimized with feature engineering |
| tabpfn | TabPFN | 0.8323 | TabPFN ensemble (~275MB, needs download) |
| deep_residual | Neural | 0.8132 | Deep residual network |
| xgb_tuned | XGBoost | 0.8065 | Hyperparameter tuned |
| optimized_residual | Neural | 0.7906 | Optimized residual network |
| baseline_mlp | Neural | 0.7663 | Simple MLP baseline |
| dgit_super | Neural | 0.7662 | Deep gated interaction transformer |
| xgb_paper | XGBoost | 0.7138 | Paper reproduction (11 features) |
| transformer | Neural | 0.6875 | Transformer architecture |

All models use unified 80/10/10 data split with seed=2026 for reproducibility.

## API Service

Start a RESTful API for web-based prediction:

```bash
# Start API (default base path /otk)
otk api

# Access points:
# - API docs: http://localhost:8000/otk/docs
# - Health: http://localhost:8000/otk/health
# - Web UI: http://localhost:8000/otk/
```

See [otk_api/README.md](otk_api/README.md) for full API documentation.

## Project Structure

```
otk/
├── src/
│   ├── otk/                # Core library
│   │   ├── data/           # Data handling
│   │   ├── models/         # Model implementations
│   │   ├── predict/        # Prediction utilities
│   │   └── cli.py          # Command-line interface
│   └── otk_preprocess/     # Copy-number preprocessing
│       ├── annotation/     # Gene annotation and reference data
│       ├── readers/        # ASCAT, FACETS, Sequenza and generic readers
│       └── workflow.py     # Input routing and preprocessing
├── otk_api/                # FastAPI web service
│   ├── api/                # API implementation
│   ├── models/             # Pre-trained models
│   └── static/             # Performance charts
├── configs/                # Model configurations
└── tests/                  # Unit tests
```

## Citation

If you use otk in your research, please cite:

```bibtex
Wang, S., et al. (2024). Machine learning-based extrachromosomal DNA 
identification in large-scale cohorts reveals its clinical implications 
in cancer. Nature Communications.
```

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Contact

- **Homepage**: https://github.com/WangLabCSU/otk
- **PyPI**: https://pypi.org/project/otk-ecdna/
- **Email**: wangshx@csu.edu.cn
