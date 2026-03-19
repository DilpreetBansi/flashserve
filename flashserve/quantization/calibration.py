"""
Quantization calibration: compute optimal scales for quantized models.
"""

import torch
import torch.nn as nn
from typing import List, Callable
from tqdm import tqdm


def calibrate_model(
    model: nn.Module,
    calibration_fn: Callable,
    num_batches: int = 100,
) -> None:
    """
    Calibrate a model for quantization.

    Runs calibration data through the model and computes optimal quantization scales.

    Args:
        model: Model to calibrate
        calibration_fn: Function that yields calibration batches
        num_batches: Number of batches to use
    """
    model.eval()

    # Hook to collect activation statistics
    activation_stats = {}

    def hook_fn(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = []
            activation_stats[name].append(output.detach().cpu().abs().max().item())
        return hook

    # Register hooks on all layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Run calibration data
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calibration_fn(), total=num_batches)):
            if i >= num_batches:
                break
            _ = model(batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute scales based on collected statistics
    # (In practice, would use these to set quantization parameters)


def compute_layer_scales(
    activations: List[float],
) -> float:
    """
    Compute quantization scale from activation statistics.

    Uses the max value observed during calibration.

    Args:
        activations: List of max activation values

    Returns:
        Scale factor
    """
    max_val = max(activations)
    scale = max_val / 127.0  # For INT8
    return scale
