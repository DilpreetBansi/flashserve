#!/usr/bin/env python3
"""
Benchmark end-to-end throughput: tokens/second.

Measures:
- Throughput (tokens/sec)
- Latency per token
- Different batch sizes
"""

import time
import numpy as np
from flashserve import LlamaConfig, InferenceEngine


def benchmark_throughput():
    """Benchmark end-to-end throughput."""
    config = LlamaConfig.tiny()
    engine = InferenceEngine(config, device="cpu")

    print("End-to-End Throughput Benchmarks")
    print(f"Config: {config.hidden_size}D, {config.num_hidden_layers} layers")
    print(f"Num parameters: {sum(p.numel() for p in engine.model.parameters()) / 1e6:.1f}M")
    print()

    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning is a subset",
        "Deep neural networks are inspired",
        "Natural language processing helps",
    ]

    # Single request
    print("Single Request Throughput:")
    times = []
    token_count = 0

    for _ in range(5):
        prompt = test_prompts[0]
        start = time.time()
        output = engine.generate(prompt, max_tokens=30, temperature=0.7)
        elapsed = time.time() - start
        times.append(elapsed)
        token_count += len(engine.tokenizer.encode(output))

    mean_time = np.mean(times)
    throughput = token_count / sum(times)
    print(f"  Average time: {mean_time*1000:.1f} ms")
    print(f"  Throughput: {throughput:.1f} tokens/sec")
    print()

    # Batch requests
    print("Batch Throughput (4 requests):")
    times = []
    token_count = 0

    for _ in range(3):
        start = time.time()
        outputs = engine.generate_batch(test_prompts, max_tokens=30, batch_size=2)
        elapsed = time.time() - start
        times.append(elapsed)
        for output in outputs:
            token_count += len(engine.tokenizer.encode(output))

    mean_time = np.mean(times)
    throughput = token_count / sum(times)
    print(f"  Average time: {mean_time*1000:.1f} ms")
    print(f"  Throughput: {throughput:.1f} tokens/sec")
    print(f"  Requests/sec: {len(test_prompts) / mean_time:.2f}")


if __name__ == "__main__":
    benchmark_throughput()
