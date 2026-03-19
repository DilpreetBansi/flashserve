"""
Fused rotary position embedding kernel (PyTorch implementation).
"""

import torch
import math


def fused_rotary_pos_emb(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Fused rotary position embedding application.

    Applies RoPE directly without materializing intermediate tensors.

    Args:
        x: Input tensor of shape (..., seq_len, num_heads, head_dim)
        positions: Position indices of shape (seq_len,)
        theta: Base for frequency calculation

    Returns:
        Tensor with RoPE applied
    """
    seq_len = x.shape[-3]
    head_dim = x.shape[-1]
    device = x.device
    dtype = x.dtype

    # Compute frequencies: theta_i = theta^(-2i/d)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))

    # Compute rotations: m * theta_i
    t = positions.to(device).to(dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, head_dim//2)

    # Create rotation matrices
    cos = freqs.cos()  # (seq_len, head_dim//2)
    sin = freqs.sin()  # (seq_len, head_dim//2)

    # Apply rotation in-place style (actually returns new tensor)
    # Reshape x to access pairs
    x_pairs = x.view(*x.shape[:-1], head_dim // 2, 2)  # (..., seq_len, num_heads, head_dim//2, 2)

    # Apply rotation matrix to each pair
    # [cos(m*theta)  -sin(m*theta)] [x_{2i}  ]
    # [sin(m*theta)   cos(m*theta)] [x_{2i+1}]

    x_rotated = torch.zeros_like(x)

    for i in range(head_dim // 2):
        x_rotated[..., 2 * i] = (
            x[..., 2 * i] * cos[:, i:i + 1].unsqueeze(-2) -
            x[..., 2 * i + 1] * sin[:, i:i + 1].unsqueeze(-2)
        )
        x_rotated[..., 2 * i + 1] = (
            x[..., 2 * i] * sin[:, i:i + 1].unsqueeze(-2) +
            x[..., 2 * i + 1] * cos[:, i:i + 1].unsqueeze(-2)
        )

    return x_rotated
