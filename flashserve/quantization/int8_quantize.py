"""
INT8 weight quantization: per-channel quantization.

Reduces model size by 4x (FP32 -> INT8) while maintaining accuracy.
No quantization of activations (FP32 activations, INT8 weights).
"""

import torch
import torch.nn as nn
from typing import Tuple


def quantize_int8(
    tensor: torch.Tensor,
    per_channel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8.

    Args:
        tensor: Input tensor to quantize
        per_channel: If True, compute scale per channel (better accuracy)

    Returns:
        Quantized tensor (INT8), scale factors
    """
    assert tensor.dim() >= 2, "Tensor must be at least 2D"

    if per_channel:
        # Per-channel quantization (scale per output channel)
        # For weight matrices: (out_features, in_features)
        out_features = tensor.shape[0]
        scales = torch.zeros(out_features, device=tensor.device, dtype=tensor.dtype)

        q_tensor = torch.zeros_like(tensor, dtype=torch.int8)

        for i in range(out_features):
            # Scale based on absolute max value
            channel_data = tensor[i]
            max_val = torch.max(torch.abs(channel_data))
            scale = max_val / 127.0  # INT8 max value
            scales[i] = scale

            # Quantize
            q_tensor[i] = torch.round(channel_data / (scale + 1e-10)).clamp(-128, 127).to(torch.int8)

        return q_tensor, scales

    else:
        # Tensor-wide quantization
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / 127.0

        q_tensor = torch.round(tensor / (scale + 1e-10)).clamp(-128, 127).to(torch.int8)

        return q_tensor, scale


def dequantize_int8(
    q_tensor: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize INT8 tensor to FP32.

    Args:
        q_tensor: Quantized tensor (INT8)
        scales: Scale factors

    Returns:
        Dequantized tensor (FP32)
    """
    if scales.dim() == 0:
        # Tensor-wide scale
        return q_tensor.to(scales.dtype) * scales

    # Per-channel scales
    out_features = q_tensor.shape[0]
    tensor = torch.zeros_like(q_tensor, dtype=scales.dtype)

    for i in range(out_features):
        tensor[i] = q_tensor[i].to(scales.dtype) * scales[i]

    return tensor


class Int8LinearLayer(nn.Module):
    """Linear layer with INT8 weight quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        # Store quantized weights and scales
        self.register_buffer(
            "weight_q",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scales",
            torch.ones(out_features, dtype=torch.float32),
        )

        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights."""
        # Dequantize weights
        weight = dequantize_int8(self.weight_q, self.weight_scales)

        # Apply linear transformation
        output = torch.nn.functional.linear(x, weight, self.bias)

        return output

    def quantize_weights(self, weights: torch.Tensor) -> None:
        """Quantize and store weights."""
        q_weights, scales = quantize_int8(weights)
        self.weight_q.copy_(q_weights)
        self.weight_scales.copy_(scales)
