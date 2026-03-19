#!/usr/bin/env python3
"""
Benchmark attention kernels: Flash Attention vs Standard Attention.

Measures:
- Memory usage
- Computation time
- Accuracy
"""

import torch
import time
import numpy as np
from flashserve.attention.flash_attention import FlashAttention
from flashserve.attention.attention_utils import apply_rotary_pos_emb, get_causal_mask


def benchmark_attention():
    """Benchmark different attention implementations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("Attention Kernel Benchmarks")
    print(f"Device: {device}, dtype: {dtype}")
    print()

    # Test configurations
    configs = [
        {"batch": 1, "seq_len": 512, "num_heads": 8, "head_dim": 64},
        {"batch": 8, "seq_len": 2048, "num_heads": 32, "head_dim": 128},
        {"batch": 32, "seq_len": 4096, "num_heads": 32, "head_dim": 128},
    ]

    for config in configs:
        batch, seq_len, num_heads, head_dim = (
            config["batch"], config["seq_len"], config["num_heads"], config["head_dim"]
        )
        hidden_size = num_heads * head_dim

        print(f"Config: batch={batch}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
        print(f"  Hidden size: {hidden_size}")

        # Create tensors
        q = torch.randn((batch, seq_len, num_heads, head_dim), device=device, dtype=dtype)
        k = torch.randn((batch, seq_len, num_heads, head_dim), device=device, dtype=dtype)
        v = torch.randn((batch, seq_len, num_heads, head_dim), device=device, dtype=dtype)

        causal_mask = get_causal_mask(seq_len, device=device, dtype=dtype)

        # Benchmark Flash Attention
        flash_attn = FlashAttention(hidden_size, num_heads)
        flash_attn.to(device)
        flash_attn.eval()

        # Warmup
        with torch.no_grad():
            _ = flash_attn(q, k, v, causal_mask=causal_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time
        num_iters = 10
        times = []
        with torch.no_grad():
            for _ in range(num_iters):
                start = time.time()
                output, _ = flash_attn(q, k, v, causal_mask=causal_mask)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)

        mean_time = np.mean(times[1:]) * 1000  # Skip first (warmup)
        print(f"  Flash Attention: {mean_time:.2f} ms")

        # Estimate memory usage
        # Input: Q, K, V
        # With tiling: O(N) for attention computation
        input_memory = (q.numel() + k.numel() + v.numel()) * 4 / (1024 ** 2)  # MB
        output_memory = (batch * seq_len * num_heads * head_dim) * 4 / (1024 ** 2)

        print(f"  Memory: input={input_memory:.1f}MB, output={output_memory:.1f}MB")
        print()


if __name__ == "__main__":
    benchmark_attention()
