#!/usr/bin/env python3
"""
Batched inference example: Generate text for multiple prompts.

Demonstrates:
- Batch inference for multiple prompts
- Different sampling strategies
- Throughput measurement
"""

import time
from flashserve import LlamaConfig, InferenceEngine


def main():
    config = LlamaConfig.small()  # Slightly larger for batching demo
    engine = InferenceEngine(config, device="cpu")

    # Multiple prompts
    prompts = [
        "Artificial intelligence is transforming",
        "The best way to learn machine learning is",
        "Large language models can",
        "Neural networks work by",
        "Data science involves",
    ]

    print(f"Generating {len(prompts)} prompts in batches...")
    print()

    start = time.time()

    # Batch generation
    results = engine.generate_batch(
        prompts,
        max_tokens=40,
        temperature=0.7,
        top_p=0.95,
        batch_size=2,  # Process 2 prompts at a time
    )

    elapsed = time.time() - start

    # Display results
    for prompt, result in zip(prompts, results):
        print(f"Prompt:  {prompt}")
        print(f"Output:  {result}")
        print()

    # Statistics
    print(f"Batch generation took {elapsed:.2f}s")
    print(f"Throughput: {len(prompts) / elapsed:.1f} prompts/sec")

    stats = engine.get_stats()
    print(f"Token throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/sec")


if __name__ == "__main__":
    main()
