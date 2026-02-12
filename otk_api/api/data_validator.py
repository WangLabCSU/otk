import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from .config import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, CN_COLUMNS, VALID_CANCER_TYPES, DEFAULT_COLUMN_VALUES

class DataValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = {}
        self.column_status = {}
        
    def validate(self, file_path: str) -> Dict[str, Any]:
        self.errors = []
        self.warnings = []
        self.info = {}
        self.column_status = {}
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self.errors.append(f"无法读取CSV文件: {str(e)}")
            return self._get_report(df=None)
        
        self.info["total_rows"] = len(df)
        self.info["total_columns"] = len(df.columns)
        
        self._validate_columns(df)
        self._validate_data_types(df)
        self._validate_missing_values(df)
        self._validate_cancer_types(df)
        self._validate_data_ranges(df)
        self._generate_summary(df)
        
        return self._get_report(df)
    
    def _validate_columns(self, df: pd.DataFrame):
        columns = set(df.columns)
        
        for col in REQUIRED_COLUMNS:
            if col in columns:
                self.column_status[col] = "present"
            else:
                self.column_status[col] = "missing"
                self.errors.append(f"缺少必需列: {col}")
        
        for col in OPTIONAL_COLUMNS:
            if col in columns:
                self.column_status[col] = "present"
            else:
                self.column_status[col] = "missing"
                # Check if this column has a default value
                if col in DEFAULT_COLUMN_VALUES:
                    self.warnings.append(f"可选列缺失: {col}，将使用默认值 {DEFAULT_COLUMN_VALUES[col]}")
                elif col in ["age", "gender"]:
                    self.warnings.append(f"可选列缺失: {col}，将使用默认值")
        
        cn_present = 0
        for col in CN_COLUMNS:
            if col in columns:
                cn_present += 1
                self.column_status[col] = "present"
            else:
                self.column_status[col] = "missing"
        
        if cn_present < len(CN_COLUMNS):
            self.warnings.append(f"CN签名列不完整: 找到 {cn_present}/{len(CN_COLUMNS)} 列")
        
        extra_cols = columns - set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS + CN_COLUMNS)
        if extra_cols:
            self.info["extra_columns"] = list(extra_cols)
    
    def fill_default_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing optional columns with default values"""
        df = df.copy()
        for col, default_val in DEFAULT_COLUMN_VALUES.items():
            if col not in df.columns:
                df[col] = default_val
                self.info[f"{col}_filled"] = f"使用默认值 {default_val}"
        return df
    
    def _validate_data_types(self, df: pd.DataFrame):
        numeric_columns = [
            "segVal", "minor_cn", "intersect_ratio", "purity", "ploidy",
            "AScore", "pLOH", "cna_burden", "age", "gender"
        ] + CN_COLUMNS
        
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.errors.append(f"列 {col} 应为数值类型")
                    self.column_status[col] = "type_error"
    
    def _validate_missing_values(self, df: pd.DataFrame):
        for col in REQUIRED_COLUMNS:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / len(df)) * 100
                    if missing_pct > 50:
                        self.errors.append(f"列 {col} 缺失值过多: {missing_pct:.1f}%")
                    else:
                        self.warnings.append(f"列 {col} 有 {missing_count} 个缺失值 ({missing_pct:.1f}%)")
        
        if "age" in df.columns:
            missing_age = df["age"].isnull().sum()
            if missing_age > 0:
                self.warnings.append(f"age列有 {missing_age} 个缺失值，将使用均值填充")
    
    def _validate_cancer_types(self, df: pd.DataFrame):
        if "type" not in df.columns:
            self.warnings.append("缺少type列，无法验证癌症类型")
            return
        
        unique_types = df["type"].dropna().unique()
        invalid_types = [t for t in unique_types if t not in VALID_CANCER_TYPES]
        
        if invalid_types:
            self.warnings.append(f"发现未识别的癌症类型: {invalid_types}")
        
        self.info["cancer_types_found"] = list(unique_types)
    
    def _validate_data_ranges(self, df: pd.DataFrame):
        range_checks = {
            "purity": (0, 1),
            "ploidy": (0, 10),
            "segVal": (0, 100),
            "intersect_ratio": (0, 1),
            "AScore": (0, 100),
            "pLOH": (0, 1),
            "cna_burden": (0, 1),
        }
        
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    self.warnings.append(f"列 {col} 有 {len(out_of_range)} 个值超出范围 [{min_val}, {max_val}]")
    
    def _generate_summary(self, df: pd.DataFrame):
        if "sample" in df.columns:
            self.info["unique_samples"] = df["sample"].nunique()
        
        if "gene_id" in df.columns:
            self.info["unique_genes"] = df["gene_id"].nunique()
        
        if "type" in df.columns:
            type_counts = df["type"].value_counts().to_dict()
            self.info["cancer_type_distribution"] = type_counts
    
    def _get_report(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        is_valid = len(self.errors) == 0
        
        data_summary = {}
        if df is not None:
            data_summary = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            }
            if "sample" in df.columns:
                data_summary["unique_samples"] = df["sample"].nunique()
            if "gene_id" in df.columns:
                data_summary["unique_genes"] = df["gene_id"].nunique()
        
        return {
            "is_valid": is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "column_status": self.column_status,
            "data_summary": data_summary,
        }
