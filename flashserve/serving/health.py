"""
Health checks and monitoring metrics.
"""

import torch
import psutil
from dataclasses import dataclass
from typing import Dict, Any
import time


@dataclass
class HealthStatus:
    """Health status information."""
    status: str  # "healthy" or "degraded"
    uptime_seconds: float
    device_type: str
    device_memory_gb: float
    device_utilization_percent: float
    system_memory_percent: float
    queue_depth: int
    throughput_tokens_per_sec: float


def get_device_memory() -> Dict[str, float]:
    """Get GPU/CPU memory statistics."""
    if torch.cuda.is_available():
        # GPU memory
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        utilization = 100 * allocated / total if total > 0 else 0
        return {
            "device_type": "cuda",
            "total_gb": total,
            "allocated_gb": allocated,
            "free_gb": total - allocated,
            "utilization_percent": utilization,
        }
    else:
        # CPU memory
        vm = psutil.virtual_memory()
        return {
            "device_type": "cpu",
            "total_gb": vm.total / (1024 ** 3),
            "allocated_gb": vm.used / (1024 ** 3),
            "free_gb": vm.available / (1024 ** 3),
            "utilization_percent": vm.percent,
        }


def get_system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }
