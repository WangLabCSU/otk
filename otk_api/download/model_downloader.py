#!/usr/bin/env python
"""
Model Downloader - 自动下载大型模型文件

支持从 GitHub Release 下载大模型，带中国镜像加速。
"""

import os
import logging
import hashlib
import requests
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)

# GitHub 仓库信息
GITHUB_REPO = "WangLabCSU/otk"
GITHUB_RELEASE_TAG = "v1.0.0-models"

# 大模型配置
LARGE_MODELS = {
    "tabpfn": {
        "filename": "tabpfn_model.pkl",  # GitHub Release 上的文件名
        "local_filename": "best_model.pkl",  # 下载后保存的本地文件名
        "size": 288510107,  # ~275MB
        "sha256": None,  # 可选校验
        "required": True,
        "description": "TabPFN ensemble model for ecDNA prediction"
    },
}

# 中国镜像加速
DOWNLOAD_MIRRORS = [
    # Original GitHub (default)
    "https://github.com",
    # Chinese mirrors
    "https://ghproxy.vip/https://github.com",
    "https://gh-proxy.org/https://github.com",
    "https://ghfast.top/https://github.com",
    # Additional mirrors
    "https://mirror.ghproxy.com/https://github.com",
    "https://ghproxy.net/https://github.com",
]


def get_model_dir(model_name: str, base_dir: Optional[Path] = None) -> Path:
    """获取模型目录路径"""
    if base_dir is None:
        # 默认路径: otk_api/models/{model_name}
        current = Path(__file__).resolve()
        # otk/otk_api/download/model_downloader.py -> otk/otk_api -> otk
        project_root = current.parent.parent.parent
        base_dir = project_root / "otk_api" / "models"

    return base_dir / model_name


def get_model_path(model_name: str, base_dir: Optional[Path] = None) -> Path:
    """获取模型文件路径"""
    model_dir = get_model_dir(model_name, base_dir)
    if model_name in LARGE_MODELS:
        # 使用 local_filename 作为本地保存的文件名
        local_name = LARGE_MODELS[model_name].get("local_filename", LARGE_MODELS[model_name]["filename"])
        return model_dir / local_name
    else:
        # 默认文件名
        return model_dir / "best_model.pkl"


def model_exists(model_name: str, base_dir: Optional[Path] = None) -> bool:
    """检查模型是否存在"""
    model_path = get_model_path(model_name, base_dir)
    return model_path.exists()


def verify_model(model_name: str, base_dir: Optional[Path] = None) -> bool:
    """验证模型文件（大小和可选的 SHA256）"""
    model_path = get_model_path(model_name, base_dir)

    if not model_path.exists():
        return False

    if model_name in LARGE_MODELS:
        config = LARGE_MODELS[model_name]

        # 检查文件大小
        actual_size = model_path.stat().st_size
        expected_size = config["size"]

        # 允许 5% 的偏差（压缩等）
        if abs(actual_size - expected_size) > expected_size * 0.05:
            logger.warning(f"Model size mismatch: expected ~{expected_size}, got {actual_size}")
            return False

        # 可选的 SHA256 校验
        if config["sha256"]:
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            actual_hash = sha256_hash.hexdigest()
            if actual_hash != config["sha256"]:
                logger.warning(f"SHA256 mismatch: expected {config['sha256']}, got {actual_hash}")
                return False

    return True


def download_file(url: str, dest_path: Path, timeout: int = 300) -> bool:
    """下载文件到指定路径"""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 流式下载，显示进度条
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dest_path.name}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        logger.error(f"Download failed from {url}: {e}")
        # 清理部分下载的文件
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_model(model_name: str, base_dir: Optional[Path] = None,
                   mirrors: Optional[List[str]] = None,
                   force: bool = False) -> Path:
    """
    下载模型文件

    Args:
        model_name: 模型名称 (e.g., "tabpfn")
        base_dir: 模型目录基路径
        mirrors: 自定义镜像列表
        force: 强制重新下载

    Returns:
        模型文件路径

    Raises:
        RuntimeError: 所有镜像下载失败
    """
    model_path = get_model_path(model_name, base_dir)

    # 检查是否已存在且有效
    if not force and verify_model(model_name, base_dir):
        logger.info(f"Model {model_name} already exists and verified at {model_path}")
        return model_path

    # 获取模型配置
    if model_name not in LARGE_MODELS:
        raise ValueError(f"Unknown large model: {model_name}")

    config = LARGE_MODELS[model_name]

    # 使用默认镜像或自定义镜像
    mirrors_to_use = mirrors or DOWNLOAD_MIRRORS

    # 构建 Release URL
    # GitHub Release URL 格式: https://github.com/{repo}/releases/download/{tag}/{filename}
    filename = config["filename"]

    logger.info(f"Downloading {model_name} model ({filename}, ~{config['size']//1024//1024}MB)")
    logger.info(f"Target path: {model_path}")

    # 尝试每个镜像
    success = False
    for mirror in mirrors_to_use:
        # 构建下载 URL
        if mirror == "https://github.com":
            url = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{filename}"
        else:
            # 镜像 URL 格式
            base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{filename}"
            url = f"{mirror}/{base_url}" if not mirror.endswith('/') else f"{mirror}{base_url}"
            # 处理已经是完整格式的镜像
            if "/github.com" in mirror:
                url = f"{mirror}/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{filename}"

        logger.info(f"Trying mirror: {mirror}")

        if download_file(url, model_path):
            # 验证下载的文件
            if verify_model(model_name, base_dir):
                logger.info(f"Successfully downloaded {model_name} from {mirror}")
                success = True
                break
            else:
                logger.warning(f"Downloaded file from {mirror} is corrupted, trying next mirror")
                model_path.unlink()

    if not success:
        raise RuntimeError(f"Failed to download {model_name} from all mirrors. "
                          f"Please download manually from: "
                          f"https://github.com/{GITHUB_REPO}/releases/tag/{GITHUB_RELEASE_TAG}")

    return model_path


def ensure_model(model_name: str, base_dir: Optional[Path] = None,
                 auto_download: bool = True) -> Path:
    """
    确保模型存在，如不存在则自动下载

    Args:
        model_name: 模型名称
        base_dir: 模型目录基路径
        auto_download: 是否自动下载

    Returns:
        模型文件路径
    """
    model_path = get_model_path(model_name, base_dir)

    # 如果存在且有效，直接返回
    if verify_model(model_name, base_dir):
        return model_path

    # 如果不自动下载，报错
    if not auto_download:
        raise FileNotFoundError(
            f"Model {model_name} not found at {model_path}. "
            f"Please download manually from: "
            f"https://github.com/{GITHUB_REPO}/releases/tag/{GITHUB_RELEASE_TAG}"
        )

    # 自动下载
    logger.info(f"Model {model_name} not found, downloading...")
    return download_model(model_name, base_dir)


def list_large_models() -> Dict[str, Dict]:
    """列出所有需要下载的大模型"""
    return LARGE_MODELS.copy()


def get_download_info(model_name: str) -> str:
    """获取模型下载信息"""
    if model_name not in LARGE_MODELS:
        return f"Model {model_name} is not a large model requiring download"

    config = LARGE_MODELS[model_name]
    remote_filename = config["filename"]
    local_filename = config.get("local_filename", remote_filename)
    manual_url = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{remote_filename}"

    return (
        f"Model: {model_name}\n"
        f"Remote filename: {remote_filename}\n"
        f"Local filename: {local_filename}\n"
        f"Size: ~{config['size']//1024//1024}MB\n"
        f"Description: {config['description']}\n"
        f"Manual download URL: {manual_url}\n"
        f"Chinese mirrors:\n" +
        "\n".join(f"  - {m}" for m in DOWNLOAD_MIRRORS[1:])  # 排除原始 GitHub
    )


if __name__ == "__main__":
    # 测试下载功能
    import argparse

    parser = argparse.ArgumentParser(description="Model Downloader")
    parser.add_argument("--model", default="tabpfn", help="Model to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--info", action="store_true", help="Show download info")
    parser.add_argument("--list", action="store_true", help="List all large models")

    args = parser.parse_args()

    if args.list:
        print("Large models requiring download:")
        for name, config in LARGE_MODELS.items():
            print(f"  {name}: {config['filename']} (~{config['size']//1024//1024}MB)")
        print()

    if args.info:
        print(get_download_info(args.model))

    if not args.info and not args.list:
        print(f"Downloading {args.model}...")
        path = download_model(args.model, force=args.force)
        print(f"Model downloaded to: {path}")