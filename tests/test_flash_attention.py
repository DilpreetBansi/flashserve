"""Tests for Flash Attention implementation."""

import torch
import pytest
from flashserve.attention.flash_attention import FlashAttention
from flashserve.attention.attention_utils import get_causal_mask


def test_flash_attention_shapes():
    """Test that Flash Attention preserves tensor shapes."""
    batch_size = 2
    seq_len = 128
    num_heads = 4
    head_dim = 64
    hidden_size = num_heads * head_dim

    flash_attn = FlashAttention(hidden_size, num_heads)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    output, _ = flash_attn(q, k, v)

    assert output.shape == (batch_size, seq_len, num_heads, head_dim)


def test_flash_attention_with_causal_mask():
    """Test Flash Attention with causal masking."""
    batch_size = 2
    seq_len = 32
    num_heads = 4
    head_dim = 64
    hidden_size = num_heads * head_dim

    flash_attn = FlashAttention(hidden_size, num_heads)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    causal_mask = get_causal_mask(seq_len, device=q.device, dtype=q.dtype)

    output, _ = flash_attn(q, k, v, causal_mask=causal_mask)

    assert output.shape == (batch_size, seq_len, num_heads, head_dim)
    assert not torch.isnan(output).any()


def test_flash_attention_numerical_stability():
    """Test that Flash Attention doesn't produce NaNs."""
    batch_size = 1
    seq_len = 64
    num_heads = 2
    head_dim = 32
    hidden_size = num_heads * head_dim

    flash_attn = FlashAttention(hidden_size, num_heads)

    # Large values that could cause numerical issues
    q = torch.randn(batch_size, seq_len, num_heads, head_dim) * 10
    k = torch.randn(batch_size, seq_len, num_heads, head_dim) * 10
    v = torch.randn(batch_size, seq_len, num_heads, head_dim) * 10

    output, _ = flash_attn(q, k, v)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_causal_mask():
    """Test causal mask creation."""
    seq_len = 4
    mask = get_causal_mask(seq_len, device="cpu")

    # Verify shape
    assert mask.shape == (1, 1, seq_len, seq_len)

    # Verify causality: positions can only attend to past
    # Indices where mask should be -inf (future positions)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert mask[0, 0, i, j] < 0  # Should be very negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
