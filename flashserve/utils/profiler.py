"""
GPU/CPU profiling utilities for performance analysis.
"""

import torch
import time
from typing import Dict, Any, Optional


class GPUProfiler:
    """Profile GPU memory and computation time."""

    def __init__(self, device: str = "cuda"):
        """Initialize profiler."""
        self.device = torch.device(device)
        self.memory_allocated = 0
        self.memory_peak = 0

    def start(self) -> None:
        """Start profiling."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self.memory_allocated = torch.cuda.memory_allocated()

    def end(self) -> Dict[str, Any]:
        """End profiling and return statistics."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated()
            memory_peak = torch.cuda.max_memory_allocated()

            return {
                "memory_allocated_mb": memory_allocated / (1024 ** 2),
                "memory_peak_mb": memory_peak / (1024 ** 2),
                "memory_delta_mb": (memory_allocated - self.memory_allocated) / (1024 ** 2),
            }
        else:
            return {
                "memory_allocated_mb": 0,
                "memory_peak_mb": 0,
            }


def profile_forward_pass(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    Profile a forward pass.

    Args:
        model: Model to profile
        input_ids: Input tensor
        num_iterations: Number of iterations for averaging

    Returns:
        Dictionary with timing and memory stats
    """
    device = next(model.parameters()).device
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time profiling
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

    return {
        "mean_time_ms": sum(times) / len(times) * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
    }
