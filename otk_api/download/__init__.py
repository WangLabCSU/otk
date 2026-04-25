"""
Model Download Module

Provides auto-download functionality for large model files from GitHub Release.
Supports Chinese mirror acceleration.
"""

from .model_downloader import (
    download_model,
    ensure_model,
    model_exists,
    verify_model,
    get_model_path,
    get_download_info,
    list_large_models,
    DOWNLOAD_MIRRORS,
    LARGE_MODELS,
    GITHUB_REPO,
    GITHUB_RELEASE_TAG,
)

__all__ = [
    'download_model',
    'ensure_model',
    'model_exists',
    'verify_model',
    'get_model_path',
    'get_download_info',
    'list_large_models',
    'DOWNLOAD_MIRRORS',
    'LARGE_MODELS',
    'GITHUB_REPO',
    'GITHUB_RELEASE_TAG',
]