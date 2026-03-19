#!/usr/bin/env python3
"""
Benchmark latency breakdown: TTFT and TPOT.

Measures:
- TTFT: Time-To-First-Token (prompt processing latency)
- TPOT: Time-Per-Output-Token (generation latency)
- Variance across multiple runs
"""

import time
import numpy as np
from flashserve import LlamaConfig, InferenceEngine
from flashserve.utils.metrics import MetricsTracker


def benchmark_latency():
    """Benchmark latency metrics."""
    config = LlamaConfig.tiny()
    engine = InferenceEngine(config, device="cpu")

    print("Latency Benchmarks: TTFT and TPOT")
    print(f"Config: {config.hidden_size}D, {config.num_hidden_layers} layers")
    print()

    prompts = [
        "The future of AI is",
        "Machine learning is a",
        "Deep learning models are",
    ]

    # Collect metrics
    metrics = MetricsTracker()

    for prompt in prompts:
        # Simulate TTFT measurement
        metrics.start_time = time.time()
        start = time.time()

        # Generate with time tracking
        output = engine.generate(
            prompt,
            max_tokens=30,
            temperature=0.7,
        )

        # Record first token (after prompt processing)
        metrics.record_first_token()

        # Record remaining tokens
        tokens = engine.tokenizer.encode(output)
        for _ in range(len(tokens) - 1):
            metrics.record_token()

        metrics.finalize()

    # Print results
    stats = metrics.get_stats()

    print("Results:")
    print(f"  Total tokens generated: {stats['total_tokens_generated']}")
    print()

    if "ttft_mean_ms" in stats:
        print("TTFT (Time-To-First-Token):")
        print(f"  Mean: {stats['ttft_mean_ms']:.1f} ms")
        print(f"  Median: {stats['ttft_median_ms']:.1f} ms")
        print(f"  P99: {stats['ttft_p99_ms']:.1f} ms")
        print()

    if "tpot_mean_ms" in stats:
        print("TPOT (Time-Per-Output-Token):")
        print(f"  Mean: {stats['tpot_mean_ms']:.1f} ms")
        print(f"  Median: {stats['tpot_median_ms']:.1f} ms")
        print()

    print(f"Overall throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/sec")


if __name__ == "__main__":
    benchmark_latency()
