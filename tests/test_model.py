"""Tests for Llama model architecture."""

import torch
import pytest
from flashserve.model.config import LlamaConfig
from flashserve.model.llama import LlamaForCausalLM, RMSNorm, SwiGLU


def test_llama_config_tiny():
    """Test tiny model configuration."""
    config = LlamaConfig.tiny()

    assert config.hidden_size == 256
    assert config.num_attention_heads == 4
    assert config.num_hidden_layers == 2


def test_llama_config_7b():
    """Test Llama-2-7B configuration."""
    config = LlamaConfig.llama2_7b()

    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.num_hidden_layers == 32


def test_rmsnorm():
    """Test RMSNorm layer."""
    hidden_size = 256
    batch_size = 2
    seq_len = 32

    norm = RMSNorm(hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)

    output = norm(x)

    # Output shape preserved
    assert output.shape == x.shape

    # No NaNs or Infs
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_swiglu():
    """Test SwiGLU activation."""
    hidden_size = 256
    intermediate_size = 512
    batch_size = 2
    seq_len = 32

    swiglu = SwiGLU(hidden_size, intermediate_size)
    x = torch.randn(batch_size, seq_len, hidden_size)

    output = swiglu(x)

    # Output shape correct
    assert output.shape == (batch_size, seq_len, hidden_size)

    # No NaNs
    assert not torch.isnan(output).any()


def test_llama_model_forward():
    """Test Llama model forward pass."""
    config = LlamaConfig.tiny()
    model = LlamaForCausalLM(config)

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)

    # Check output shape
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # No NaNs
    assert not torch.isnan(logits).any()


def test_llama_model_generation():
    """Test model text generation."""
    config = LlamaConfig.tiny()
    model = LlamaForCausalLM(config)

    batch_size = 1
    prompt_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))

    # Generate tokens
    output_ids = model.generate(input_ids, max_new_tokens=20)

    # Check output length
    assert output_ids.shape[1] == prompt_len + 20

    # All IDs in vocab range
    assert (output_ids >= 0).all()
    assert (output_ids < config.vocab_size).all()


def test_llama_num_params():
    """Test parameter counting."""
    config = LlamaConfig.tiny()
    model = LlamaForCausalLM(config)

    num_params = model.get_num_params()

    # Tiny model should have reasonable size
    assert num_params > 0
    assert num_params < 100_000_000  # Less than 100M


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
