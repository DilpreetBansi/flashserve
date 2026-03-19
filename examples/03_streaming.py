#!/usr/bin/env python3
"""
Streaming inference example: Stream tokens as they're generated.

Demonstrates:
- Streaming token generation
- Real-time output
- Useful for interactive applications
"""

from flashserve import LlamaConfig, InferenceEngine


def main():
    config = LlamaConfig.tiny()
    engine = InferenceEngine(config, device="cpu")

    prompt = "The concept of machine learning can be understood by"

    print(f"Streaming generation from: '{prompt}'")
    print()
    print("Generated tokens:")
    print(prompt, end=" ", flush=True)

    # Stream tokens
    for token in engine.stream_generate(
        prompt,
        max_tokens=60,
        temperature=0.8,
        top_p=0.95,
    ):
        print(token, end="", flush=True)

    print()  # Newline at end


if __name__ == "__main__":
    main()
