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

    Uses the interleaved convention from the original Llama code: element pairs
    (2i, 2i+1) are rotated by angle position * theta^(-2i/d).

    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        positions: Position indices of shape (seq_len,) or (batch, seq_len)
        rope_theta: Base for the exponential in the frequency formula

    Returns:
        Tensor with RoPE applied, same shape and dtype as input
    """
    assert x.shape[-1] % 2 == 0, "Head dimension must be even for RoPE"
    head_dim = x.shape[-1]
    device = x.device

    # Compute angles in float32 for accuracy, whatever the activation dtype.
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = positions.to(device=device, dtype=torch.float32)[..., None] * inv_freq  # (..., seq, head_dim/2)
    if angles.dim() == 2:          # (seq, d/2) -> (1, seq, 1, d/2)
        angles = angles[None, :, None, :]
    else:                          # (batch, seq, d/2) -> (batch, seq, 1, d/2)
        angles = angles[:, :, None, :]
    cos, sin = angles.cos(), angles.sin()

    xf = x.float()
    x_even, x_odd = xf[..., 0::2], xf[..., 1::2]
    out = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


def build_attention_mask(
    token_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Combine a causal mask with a key-padding mask.

    Args:
        token_mask: (batch, seq_len) bool, True for real tokens and False for padding
        dtype: Floating dtype of the returned additive mask

    Returns:
        Additive mask of shape (batch, 1, seq_len, seq_len): 0 where attention is
        allowed and the dtype's minimum value where it is not.
    """
    seq_len = token_mask.shape[-1]
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_mask.device))
    allowed = causal[None, None, :, :] & token_mask[:, None, None, :].bool()
    mask = torch.zeros(allowed.shape, dtype=dtype, device=token_mask.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def positions_from_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Positions that start at 0 on each sequence's first real token (for left padding)."""
    return (token_mask.long().cumsum(-1) - 1).clamp(min=0)


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
