"""
Decode throughput with and without the KV cache on a real checkpoint.

    python benchmarks/bench_generate.py --model HuggingFaceTB/SmolLM2-135M-Instruct --new-tokens 128
"""

import argparse
import time

import torch

from flashserve.model.weights import from_pretrained
from flashserve.utils.tokenizer import Tokenizer


def run(model, ids, new_tokens, use_cache):
    start = time.perf_counter()
    out = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, use_cache=use_cache)
    elapsed = time.perf_counter() - start
    return out.shape[1] - ids.shape[1], elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    model, config = from_pretrained(args.model)
    config = model.config
    config.eos_token_id = -1  # never stop early: measure a fixed number of tokens
    tok = Tokenizer.from_pretrained(args.model)
    ids = torch.tensor([(tok.encode("Once upon a time, ") * args.prompt_tokens)[: args.prompt_tokens]])

    run(model, ids, 4, True)  # warm-up
    print(f"model={args.model} prompt={ids.shape[1]} new={args.new_tokens} threads={torch.get_num_threads()}")
    print(f"{'mode':<28}{'tokens':>8}{'seconds':>10}{'tok/s':>10}")
    for label, cache in [("KV cache (prefill+decode)", True), ("full recompute", False)]:
        n, secs = run(model, ids, args.new_tokens, cache)
        print(f"{label:<28}{n:>8}{secs:>10.2f}{n / secs:>10.1f}")


if __name__ == "__main__":
    main()
