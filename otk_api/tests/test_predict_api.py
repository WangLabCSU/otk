#!/usr/bin/env python3
"""
System test for otk_api predict functionality.
Generates test files with various edge cases and tests the predict API.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import json
import time
import requests
from typing import Dict, List, Any, Optional
import gzip
import shutil

SCRIPT_DIR = Path(__file__).parent
OTK_API_DIR = SCRIPT_DIR.parent
OTK_BASE_DIR = OTK_API_DIR.parent.parent
DATA_DIR = OTK_BASE_DIR / "data"
TEST_DATA_DIR = SCRIPT_DIR / "test_data"
RESULTS_DIR = SCRIPT_DIR / "test_results"

REQUIRED_COLUMNS = [
    "sample", "gene_id", "segVal", "minor_cn",
    "purity", "ploidy", "AScore", "pLOH", "cna_burden",
]
CN_COLUMNS = [f"CN{i}" for i in range(1, 20)]
OPTIONAL_COLUMNS = ["age", "gender", "y", "type", "intersect_ratio"]
TYPE_COLUMNS = [
    'type_BLCA', 'type_BRCA', 'type_CESC', 'type_COAD', 'type_DLBC', 'type_ESCA',
    'type_GBM', 'type_HNSC', 'type_KICH', 'type_KIRC', 'type_KIRP', 'type_LGG',
    'type_LIHC', 'type_LUAD', 'type_LUSC', 'type_OV', 'type_PRAD', 'type_READ',
    'type_SARC', 'type_SKCM', 'type_STAD', 'type_THCA', 'type_UCEC', 'type_UVM'
]
FREQ_COLUMNS = ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']

VALID_CANCER_TYPES = [
    'BLCA', 'BRCA', 'CESC', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
    'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV',
    'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'THCA', 'UCEC', 'UVM'
]

API_BASE_URL = "http://localhost:8000"


def load_source_data(n_samples_per_class: int = 50, random_seed: int = 42) -> pd.DataFrame:
    """Load and sample data from gcap_modeling_data.csv.gz"""
    np.random.seed(random_seed)
    
    source_file = DATA_DIR / "gcap_modeling_data.csv.gz"
    if not source_file.exists():
        raise FileNotFoundError(f"Source data not found: {source_file}")
    
    print(f"Loading data from {source_file}...")
    df = pd.read_csv(source_file, compression='gzip')
    print(f"Total rows: {len(df)}, columns: {len(df.columns)}")
    
    df_y0 = df[df['y'] == 0].sample(n=n_samples_per_class, random_state=random_seed)
    df_y1 = df[df['y'] == 1].sample(n=n_samples_per_class, random_state=random_seed)
    
    sampled_df = pd.concat([df_y0, df_y1], ignore_index=True)
    print(f"Sampled {len(df_y0)} y=0 and {len(df_y1)} y=1 samples")
    
    return sampled_df


def create_test_directories():
    """Create test data and results directories"""
    TEST_DATA_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Created test directories: {TEST_DATA_DIR}, {RESULTS_DIR}")


def generate_test_files(df: pd.DataFrame) -> Dict[str, Path]:
    """Generate various test files with different edge cases"""
    test_files = {}
    
    test_files["baseline"] = generate_baseline_file(df)
    test_files["minimal"] = generate_minimal_file(df)
    test_files["with_type_column"] = generate_with_type_column(df)
    test_files["missing_values"] = generate_missing_values_file(df)
    test_files["missing_optional_columns"] = generate_missing_optional_columns(df)
    test_files["missing_cn_columns"] = generate_missing_cn_columns(df)
    test_files["column_order_random"] = generate_random_column_order(df)
    test_files["missing_intersect_ratio"] = generate_missing_intersect_ratio(df)
    test_files["missing_freq_columns"] = generate_missing_freq_columns(df)
    test_files["invalid_cancer_type"] = generate_invalid_cancer_type(df)
    test_files["mixed_type_encoding"] = generate_mixed_type_encoding(df)
    
    return test_files


def generate_baseline_file(df: pd.DataFrame) -> Path:
    """Generate baseline test file with all columns"""
    output_path = TEST_DATA_DIR / "01_baseline.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_minimal_file(df: pd.DataFrame) -> Path:
    """Generate minimal file with only sample, gene_id, segVal"""
    minimal_cols = ["sample", "gene_id", "segVal"]
    output_path = TEST_DATA_DIR / "02_minimal.csv"
    df[minimal_cols].to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_with_type_column(df: pd.DataFrame) -> Path:
    """Generate file with 'type' column instead of type_* columns"""
    df_type = df.copy()
    
    type_cols_to_drop = [col for col in TYPE_COLUMNS if col in df_type.columns]
    df_type = df_type.drop(columns=type_cols_to_drop)
    
    for _, cancer_type in enumerate(VALID_CANCER_TYPES):
        type_col = f'type_{cancer_type}'
        if type_col in df.columns:
            mask = df[type_col] == 1
            df_type.loc[mask, 'type'] = cancer_type
            break
    
    df_type['type'] = df_type['type'].fillna('BRCA')
    
    output_path = TEST_DATA_DIR / "03_with_type_column.csv"
    df_type.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_missing_values_file(df: pd.DataFrame) -> Path:
    """Generate file with missing values (NaN) in various columns"""
    df_missing = df.copy()
    
    np.random.seed(123)
    missing_indices = np.random.choice(len(df_missing), size=min(10, len(df_missing)), replace=False)
    
    for idx in missing_indices:
        col = np.random.choice(['purity', 'ploidy', 'AScore', 'age'])
        df_missing.loc[idx, col] = np.nan
    
    output_path = TEST_DATA_DIR / "04_missing_values.csv"
    df_missing.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_missing_optional_columns(df: pd.DataFrame) -> Path:
    """Generate file missing optional columns (age, gender, intersect_ratio)"""
    cols_to_drop = ['age', 'gender', 'intersect_ratio', 'y']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    df_missing = df.drop(columns=cols_to_drop)
    
    output_path = TEST_DATA_DIR / "05_missing_optional.csv"
    df_missing.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_missing_cn_columns(df: pd.DataFrame) -> Path:
    """Generate file missing some CN columns"""
    df_missing = df.copy()
    
    cols_to_drop = ['CN1', 'CN5', 'CN10', 'CN15']
    cols_to_drop = [col for col in cols_to_drop if col in df_missing.columns]
    df_missing = df_missing.drop(columns=cols_to_drop)
    
    output_path = TEST_DATA_DIR / "06_missing_cn.csv"
    df_missing.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_random_column_order(df: pd.DataFrame) -> Path:
    """Generate file with columns in random order (different from training)"""
    cols = list(df.columns)
    np.random.seed(456)
    np.random.shuffle(cols)
    
    df_shuffled = df[cols]
    
    output_path = TEST_DATA_DIR / "07_random_order.csv"
    df_shuffled.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_missing_intersect_ratio(df: pd.DataFrame) -> Path:
    """Generate file missing intersect_ratio column"""
    df_missing = df.copy()
    if 'intersect_ratio' in df_missing.columns:
        df_missing = df_missing.drop(columns=['intersect_ratio'])
    
    output_path = TEST_DATA_DIR / "08_missing_intersect_ratio.csv"
    df_missing.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_missing_freq_columns(df: pd.DataFrame) -> Path:
    """Generate file missing freq_* columns (should be filled from prior data)"""
    df_missing = df.copy()
    
    cols_to_drop = [col for col in FREQ_COLUMNS if col in df_missing.columns]
    df_missing = df_missing.drop(columns=cols_to_drop)
    
    output_path = TEST_DATA_DIR / "09_missing_freq.csv"
    df_missing.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_invalid_cancer_type(df: pd.DataFrame) -> Path:
    """Generate file with invalid cancer type values"""
    df_invalid = df.copy()
    
    type_cols_to_drop = [col for col in TYPE_COLUMNS if col in df_invalid.columns]
    df_invalid = df_invalid.drop(columns=type_cols_to_drop)
    
    df_invalid['type'] = 'INVALID_TYPE'
    
    output_path = TEST_DATA_DIR / "10_invalid_cancer_type.csv"
    df_invalid.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def generate_mixed_type_encoding(df: pd.DataFrame) -> Path:
    """Generate file with both type column and some type_* columns"""
    df_mixed = df.copy()
    
    type_cols_to_drop = ['type_BRCA', 'type_LUAD']
    type_cols_to_drop = [col for col in type_cols_to_drop if col in df_mixed.columns]
    df_mixed = df_mixed.drop(columns=type_cols_to_drop)
    
    df_mixed['type'] = 'BRCA'
    
    output_path = TEST_DATA_DIR / "11_mixed_type_encoding.csv"
    df_mixed.to_csv(output_path, index=False)
    print(f"Generated: {output_path}")
    return output_path


def check_api_health() -> bool:
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> List[str]:
    """Get list of available models from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('models', [])
    except Exception as e:
        print(f"Error getting models: {e}")
    return []


def run_api_prediction(test_name: str, input_file: Path, model: str = None) -> Dict[str, Any]:
    """Run prediction via API"""
    result = {
        "test_name": test_name,
        "input_file": str(input_file),
        "success": False,
        "error": None,
        "response": None,
        "prediction_stats": None,
    }
    
    try:
        with open(input_file, 'rb') as f:
            files = {'file': (input_file.name, f, 'text/csv')}
            data = {}
            if model:
                data['model'] = model
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/predict",
                files=files,
                data=data,
                timeout=300
            )
        
        result["http_status"] = response.status_code
        
        if response.status_code == 200:
            result["response"] = response.json()
            result["success"] = True
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text}"
    
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_cli_prediction(test_name: str, input_file: Path, model_name: str) -> Dict[str, Any]:
    """Run prediction via CLI (otk predict)"""
    result = {
        "test_name": test_name,
        "input_file": str(input_file),
        "success": False,
        "error": None,
        "output": None,
    }
    
    output_dir = RESULTS_DIR / f"{test_name}_output"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    cmd = [
        "otk", "predict",
        "-i", str(input_file),
        "-m", model_name,
        "-o", str(output_dir)
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(OTK_API_DIR.parent.parent)
        )
        
        result["return_code"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        
        if proc.returncode == 0:
            result["success"] = True
            predictions_file = output_dir / "predictions.csv"
            if predictions_file.exists():
                result["predictions_file"] = str(predictions_file)
                pred_df = pd.read_csv(predictions_file)
                result["prediction_stats"] = {
                    "total_rows": len(pred_df),
                    "columns": list(pred_df.columns),
                    "prediction_distribution": pred_df['prediction'].value_counts().to_dict() if 'prediction' in pred_df.columns else None,
                }
        else:
            result["error"] = proc.stderr or proc.stdout
    
    except subprocess.TimeoutExpired:
        result["error"] = "Command timed out"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_tests(test_files: Dict[str, Path], use_api: bool = True, use_cli: bool = True) -> Dict[str, Any]:
    """Run all tests"""
    results = {
        "api_tests": {},
        "cli_tests": {},
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    }
    
    model_name = "transformer"
    
    if use_api:
        print("\n" + "=" * 60)
        print("Running API Tests")
        print("=" * 60)
        
        if not check_api_health():
            print("WARNING: API server not running. Skipping API tests.")
            print("Start the API server with: cd otk/otk_api && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        else:
            models = get_available_models()
            print(f"Available models: {models}")
            default_model = models[0]['name'] if models and isinstance(models[0], dict) else (models[0] if models else None)
            
            for test_name, test_file in test_files.items():
                print(f"\nTesting: {test_name}...")
                result = run_api_prediction(test_name, test_file, default_model)
                results["api_tests"][test_name] = result
                
                results["summary"]["total"] += 1
                if result["success"]:
                    results["summary"]["passed"] += 1
                    print(f"  PASSED")
                else:
                    results["summary"]["failed"] += 1
                    results["summary"]["errors"].append({
                        "test": test_name,
                        "error": result["error"]
                    })
                    print(f"  FAILED: {result['error']}")
    
    if use_cli:
        print("\n" + "=" * 60)
        print("Running CLI Tests")
        print("=" * 60)
        
        for test_name, test_file in test_files.items():
            print(f"\nTesting: {test_name}...")
            result = run_cli_prediction(test_name, test_file, model_name)
            results["cli_tests"][test_name] = result
            
            if not use_api:
                results["summary"]["total"] += 1
                if result["success"]:
                    results["summary"]["passed"] += 1
                    print(f"  PASSED")
                else:
                    results["summary"]["failed"] += 1
                    results["summary"]["errors"].append({
                        "test": test_name,
                        "error": result["error"]
                    })
                    print(f"  FAILED: {result['error']}")
            else:
                status = "PASSED" if result["success"] else f"FAILED: {result['error']}"
                print(f"  {status}")
    
    return results


def save_results(results: Dict[str, Any]):
    """Save test results to file"""
    results_file = RESULTS_DIR / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


def main():
    """Main function"""
    print("=" * 60)
    print("OTK API Predict System Test")
    print("=" * 60)
    
    create_test_directories()
    
    df = load_source_data(n_samples_per_class=50, random_seed=42)
    
    print("\nGenerating test files...")
    test_files = generate_test_files(df)
    
    print(f"\nGenerated {len(test_files)} test files:")
    for name, path in test_files.items():
        print(f"  - {name}: {path}")
    
    results = run_tests(test_files, use_api=True, use_cli=True)
    
    save_results(results)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total tests: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    
    if results['summary']['errors']:
        print("\nErrors:")
        for err in results['summary']['errors']:
            print(f"  - {err['test']}: {err['error']}")
    
    return results['summary']['failed'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
