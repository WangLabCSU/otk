import torch
import psutil
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from .config import (
    DEFAULT_MODEL, MODEL_PRIORITY, GPU_MEMORY_THRESHOLD, 
    PREFER_GPU, get_model_search_paths
)

class ResourceManager:
    def __init__(self):
        self._selected_model: Optional[str] = DEFAULT_MODEL
        self.available_models = self._scan_models()
        self.gpu_usage = {}
        self._init_gpu_monitoring()
    
    def _scan_models(self) -> Dict[str, Path]:
        """Scan for available models in configured directories"""
        models = {}
        
        for search_dir in get_model_search_paths():
            if search_dir.exists():
                # Scan subdirectories for model files
                for model_dir in search_dir.iterdir():
                    if model_dir.is_dir():
                        model_file = model_dir / "best_model.pth"
                        if model_file.exists():
                            model_name = model_dir.name.replace("output_", "")
                            models[model_name] = model_file
                # Also check the search_dir itself (for backward compatibility)
                model_file = search_dir / "best_model.pth"
                if model_file.exists():
                    model_name = search_dir.name.replace("output_", "")
                    models[model_name] = model_file
        
        return models
    
    def _init_gpu_monitoring(self):
        """Initialize GPU usage tracking"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.gpu_usage[i] = {
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                    "used_memory": 0,
                    "active_jobs": 0,
                }
    
    def get_available_models(self) -> Dict[str, Path]:
        """Get dictionary of available models"""
        return self.available_models.copy()
    
    def get_model_list(self) -> List[Dict]:
        """Get list of models with their info"""
        models = []
        for name, path in self.available_models.items():
            models.append({
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "is_default": name == self._selected_model,
            })
        return models
    
    def select_model(self, model_name: str) -> bool:
        """Select a specific model to use"""
        if model_name in self.available_models:
            self._selected_model = model_name
            return True
        return False
    
    def get_selected_model(self) -> Optional[str]:
        """Get currently selected model name"""
        return self._selected_model
    
    def rescan_models(self) -> Dict[str, Path]:
        """Rescan for available models in configured directories"""
        self.available_models = self._scan_models()
        return self.available_models.copy()
    
    def get_best_model(self) -> Tuple[str, Path]:
        """Get the best available model based on priority"""
        # If a model is explicitly selected and exists, use it
        if self._selected_model and self._selected_model in self.available_models:
            return self._selected_model, self.available_models[self._selected_model]
        
        # Use priority order if defined
        for model_name in MODEL_PRIORITY:
            if model_name in self.available_models:
                return model_name, self.available_models[model_name]
        
        # If no priority defined, use default_model from config if available
        if DEFAULT_MODEL and DEFAULT_MODEL in self.available_models:
            return DEFAULT_MODEL, self.available_models[DEFAULT_MODEL]
        
        # Fallback to any available model (alphabetically sorted for consistency)
        if self.available_models:
            sorted_models = sorted(self.available_models.items())
            return sorted_models[0]
            
        raise RuntimeError("No available models found")
    
    def select_device(self) -> Tuple[str, Optional[int]]:
        """Select the best available device (GPU or CPU)"""
        if not PREFER_GPU or not torch.cuda.is_available():
            return "cpu", None
        
        best_gpu = None
        best_available_memory = 0
        
        for gpu_id, usage in self.gpu_usage.items():
            try:
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                reserved = torch.cuda.memory_reserved(gpu_id)
                total = usage["total_memory"]
                
                used_ratio = (allocated + reserved) / total
                available_memory = total - allocated - reserved
                
                if used_ratio < GPU_MEMORY_THRESHOLD and available_memory > best_available_memory:
                    best_gpu = gpu_id
                    best_available_memory = available_memory
                    
            except Exception:
                continue
        
        if best_gpu is not None:
            return "cuda", best_gpu
        else:
            return "cpu", None
    
    def allocate_resource(self, job_id: str) -> Tuple[str, Path, str, Optional[int]]:
        """Allocate resources for a job"""
        model_name, model_path = self.get_best_model()
        device_type, gpu_id = self.select_device()
        
        if device_type == "cuda" and gpu_id is not None:
            self.gpu_usage[gpu_id]["active_jobs"] += 1
        
        return model_name, model_path, device_type, gpu_id
    
    def release_resource(self, job_id: str, device_type: str, gpu_id: Optional[int]):
        """Release resources after job completion"""
        if device_type == "cuda" and gpu_id is not None:
            if gpu_id in self.gpu_usage:
                self.gpu_usage[gpu_id]["active_jobs"] = max(0, self.gpu_usage[gpu_id]["active_jobs"] - 1)
                torch.cuda.empty_cache()
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        stats = {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "available": True,
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent": memory.percent,
            },
            "gpu": {
                "available": torch.cuda.is_available(),
                "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "devices": [],
            },
            "models": {
                "available": list(self.available_models.keys()),
                "selected": self._selected_model,
                "priority": MODEL_PRIORITY,
            },
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    
                    stats["gpu"]["devices"].append({
                        "id": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024**3),
                        "allocated_memory_gb": allocated / (1024**3),
                        "reserved_memory_gb": reserved / (1024**3),
                        "active_jobs": self.gpu_usage.get(i, {}).get("active_jobs", 0),
                    })
                except Exception as e:
                    stats["gpu"]["devices"].append({
                        "id": i,
                        "error": str(e),
                    })
        
        return stats

# Global resource manager instance
resource_manager = ResourceManager()
