#!/usr/bin/env python3
"""
Basic inference example: Simple text generation with tiny model.

This example demonstrates the simplest usage of FlashServe:
- Load tiny model config (CPU-friendly)
- Create inference engine
- Generate text from a prompt
"""

import torch
from flashserve import LlamaConfig, InferenceEngine


def main():
    # Use tiny model for CPU inference
    config = LlamaConfig.tiny()

    # Create inference engine
    engine = InferenceEngine(config, device="cpu")

    print(f"Model config: {config.hidden_size}D, {config.num_hidden_layers} layers")
    print(f"Num parameters: {sum(p.numel() for p in engine.model.parameters()) / 1e6:.1f}M")
    print()

    # Generate from prompt
    prompts = [
        "The future of AI is",
        "In machine learning, the key",
        "Deep learning models excel at",
    ]

    print("Generating text...")
    for prompt in prompts:
        print(f"\n>> Prompt: {prompt}")
        generated = engine.generate(
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.95,
        )
        print(f"<< Output: {generated}")

    # Print statistics
    stats = engine.get_stats()
    print(f"\nGeneration stats:")
    print(f"  Total tokens: {stats['total_tokens_generated']}")
    print(f"  Throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/sec")


if __name__ == "__main__":
    main()
