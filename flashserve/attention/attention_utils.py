"""
Attention utilities: RoPE embeddings, causal masking, and attention score computation.
"""

import torch
import math
from typing import Optional, Tuple


def apply_rotary_pos_emb(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings (RoPE) to input tensor.

    RoPE applies rotation matrices to pairs of elements in the embedding space,
    encoding absolute position information without learned parameters.

    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        positions: Position indices of shape (seq_len,)
        rope_theta: Base for the exponential in the frequency formula

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    assert x.shape[-1] % 2 == 0, "Head dimension must be even for RoPE"

    head_dim = x.shape[-1]
    batch_size, seq_len, num_heads = x.shape[0], x.shape[1], x.shape[2]
    device = x.device
    dtype = x.dtype

    # Compute frequencies: theta_i = rope_theta^(-2i/d)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))

    # Compute rotations: theta_i * m
    t = positions.to(device).to(dtype)  # (seq_len,)
    freqs = torch.einsum('i,j->ij', t, inv_freq)  # (seq_len, head_dim//2)

    # Duplicate to get (seq_len, head_dim)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)

    # Create rotation matrix: cos(theta), sin(theta)
    cos = emb.cos()[None, :, None, :]  # (1, seq_len, 1, head_dim)
    sin = emb.sin()[None, :, None, :]  # (1, seq_len, 1, head_dim)

    # Apply rotation to pairs of elements
    x_rot = torch.stack([
        x[..., 0::2] * cos[..., 0::2] - x[..., 1::2] * sin[..., 1::2],
        x[..., 0::2] * sin[..., 0::2] + x[..., 1::2] * cos[..., 1::2],
    ], dim=-1)

    # Interleave back to original shape
    x_rot = x_rot.flatten(-2)

    return x_rot.to(dtype)


def get_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a causal (lower triangular) attention mask.

    Prevents tokens from attending to future positions during autoregressive generation.

    Args:
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type (should be float for softmax)

    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len), with -inf for future positions
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    # Convert 0s to -inf (attended), 1s to 0 (not attended)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    return mask[None, None, :, :]


def compute_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention scores: Q @ K^T / sqrt(d).

    Args:
        q: Query tensor of shape (..., seq_len_q, head_dim)
        k: Key tensor of shape (..., seq_len_k, head_dim)
        scale: Scaling factor (default: 1 / sqrt(head_dim))

    Returns:
        Attention scores of shape (..., seq_len_q, seq_len_k)
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores * scale

    return scores


def apply_attention_mask(
    scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply attention mask to scores before softmax.

    Args:
        scores: Attention scores of shape (..., seq_len_q, seq_len_k)
        mask: Attention mask of shape (..., seq_len_q, seq_len_k) with -inf for masked positions

    Returns:
        Masked scores
    """
    return scores + mask


def softmax_with_mask(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = -1,
) -> torch.Tensor:
    """
    Apply softmax with optional masking.

    Args:
        scores: Scores to apply softmax to
        mask: Optional mask with -inf for positions to mask out
        dim: Dimension to apply softmax over

    Returns:
        Softmax probabilities
    """
    if mask is not None:
        scores = apply_attention_mask(scores, mask)

    return torch.softmax(scores, dim=dim)
