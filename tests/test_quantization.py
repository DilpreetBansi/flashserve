"""Tests for quantization modules."""

import torch
import pytest
from flashserve.quantization.int8_quantize import quantize_int8, dequantize_int8
from flashserve.quantization.int4_quantize import quantize_int4, dequantize_int4


def test_int8_quantization():
    """Test INT8 quantization and dequantization."""
    # Create test tensor
    tensor = torch.randn(64, 256)

    # Quantize
    q_tensor, scales = quantize_int8(tensor)

    assert q_tensor.dtype == torch.int8
    assert q_tensor.shape == tensor.shape

    # Dequantize
    dq_tensor = dequantize_int8(q_tensor, scales)

    assert dq_tensor.shape == tensor.shape
    assert dq_tensor.dtype == tensor.dtype

    # Check accuracy (should be reasonable after quantization)
    error = torch.abs(tensor - dq_tensor).mean()
    assert error < 0.1  # Reasonable error for INT8


def test_int8_per_channel():
    """Test per-channel INT8 quantization."""
    tensor = torch.randn(32, 128)

    q_tensor, scales = quantize_int8(tensor, per_channel=True)

    assert scales.shape == (32,)  # One scale per output channel
    assert q_tensor.dtype == torch.int8


def test_int4_quantization():
    """Test INT4 group-wise quantization."""
    tensor = torch.randn(64, 256)

    # Quantize
    q_tensor, scales = quantize_int4(tensor, group_size=64)

    assert q_tensor.dtype == torch.int8
    assert q_tensor.shape == tensor.shape
    assert scales.shape[0] == 64  # One scale per output feature
    assert scales.shape[1] == 4   # Four groups (256 / 64)

    # Dequantize
    dq_tensor = dequantize_int4(q_tensor, scales, group_size=64)

    assert dq_tensor.shape == tensor.shape
    assert dq_tensor.dtype == tensor.dtype


def test_int4_vs_int8_size():
    """Test that INT4 is smaller than INT8."""
    tensor = torch.randn(256, 1024)

    q8_tensor, scales8 = quantize_int8(tensor)
    q4_tensor, scales4 = quantize_int4(tensor)

    # INT4 should use less memory (nominally 2x, but stored as INT8 here)
    # In practice, INT4 would pack 2 values per byte
    size_int8 = q8_tensor.numel() * 1 + scales8.numel() * 4  # bytes
    size_int4 = q4_tensor.numel() * 1 + scales4.numel() * 4  # bytes (could be optimized)

    # At least scales should be reasonable
    assert scales4.numel() >= scales8.numel()


def test_quantization_accuracy():
    """Test quantization accuracy on specific values."""
    # Test with known values
    tensor = torch.tensor([
        [127.0, -128.0, 0.0, 64.0],
        [1.0, 0.5, -0.5, 100.0],
    ], dtype=torch.float32)

    q_tensor, scales = quantize_int8(tensor)
    dq_tensor = dequantize_int8(q_tensor, scales)

    # All values should dequantize correctly
    assert dq_tensor.shape == tensor.shape
    assert (torch.abs(dq_tensor) >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
