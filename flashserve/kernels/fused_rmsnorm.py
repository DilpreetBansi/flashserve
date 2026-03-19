"""
Fused RMSNorm kernel implementation (PyTorch with optional Triton).
"""

import torch
import torch.nn.functional as F


def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused RMSNorm: normalize and scale in one operation.

    Computes: (x / sqrt(mean(x^2) + eps)) * weight + bias

    Args:
        x: Input tensor
        weight: Scale weight
        bias: Optional bias (typically None for RMSNorm)
        eps: Numerical stability constant

    Returns:
        Normalized tensor
    """
    # Compute RMS: sqrt(mean(x^2))
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)

    # Normalize
    normed = x / rms

    # Scale with weight
    output = normed * weight

    if bias is not None:
        output = output + bias

    return output
