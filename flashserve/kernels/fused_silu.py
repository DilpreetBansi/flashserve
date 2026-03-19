"""
Fused SiLU (Swish) activation kernel.
"""

import torch
import torch.nn.functional as F


def fused_silu(x: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU activation: x * sigmoid(x)

    More efficient than computing sigmoid and multiplication separately.

    Args:
        x: Input tensor

    Returns:
        SiLU activated tensor
    """
    return x * torch.sigmoid(x)
