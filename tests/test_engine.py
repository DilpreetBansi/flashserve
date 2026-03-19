"""Tests for inference engine."""

import torch
import pytest
from flashserve.model.config import LlamaConfig
from flashserve.engine.inference_engine import InferenceEngine
from flashserve.engine.continuous_batching import ContinuousBatchingScheduler, RequestStage


def test_inference_engine_creation():
    """Test InferenceEngine initialization."""
    config = LlamaConfig.tiny()
    engine = InferenceEngine(config, device="cpu")

    assert engine.model is not None
    assert engine.tokenizer is not None
    assert engine.device.type == "cpu"


def test_single_generation():
    """Test single text generation."""
    config = LlamaConfig.tiny()
    engine = InferenceEngine(config, device="cpu")

    prompt = "The future is"
    output = engine.generate(prompt, max_tokens=20)

    assert isinstance(output, str)
    assert len(output) > 0


def test_batch_generation():
    """Test batch text generation."""
    config = LlamaConfig.tiny()
    engine = InferenceEngine(config, device="cpu")

    prompts = ["The future is", "Machine learning", "Deep learning"]
    outputs = engine.generate_batch(prompts, max_tokens=20, batch_size=2)

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert isinstance(output, str)


def test_streaming_generation():
    """Test streaming generation."""
    config = LlamaConfig.tiny()
    engine = InferenceEngine(config, device="cpu")

    prompt = "The future is"
    tokens = list(engine.stream_generate(prompt, max_tokens=10))

    assert len(tokens) > 0
    for token in tokens:
        assert isinstance(token, str)


def test_stats_tracking():
    """Test statistics tracking."""
    config = LlamaConfig.tiny()
    engine = InferenceEngine(config, device="cpu")

    # Generate some text
    engine.generate("Test prompt", max_tokens=10)

    stats = engine.get_stats()

    assert stats["total_tokens_generated"] > 0
    assert stats["total_inference_time_seconds"] > 0
    assert stats["throughput_tokens_per_sec"] > 0


def test_continuous_batching_scheduler():
    """Test continuous batching scheduler."""
    scheduler = ContinuousBatchingScheduler(max_batch_size=4, max_wait_time_ms=50)

    # Add requests
    req_id_1 = scheduler.add_request("prompt 1", 50)
    req_id_2 = scheduler.add_request("prompt 2", 50)

    assert req_id_1 == 0
    assert req_id_2 == 1

    # Get batch
    batch = scheduler.get_next_batch()
    assert batch is not None
    assert batch.stage == RequestStage.PREFILL


def test_batching_request_tracking():
    """Test that scheduler tracks requests correctly."""
    scheduler = ContinuousBatchingScheduler(max_batch_size=2)

    req1 = scheduler.add_request("prompt 1", 50)
    req2 = scheduler.add_request("prompt 2", 50)

    # Check request is in map
    assert req1 in scheduler.request_map
    assert req2 in scheduler.request_map

    # Mark as complete
    scheduler.mark_request_complete(req1)
    assert scheduler.request_map[req1].is_completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
