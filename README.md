# FlashServe

A from-scratch inference engine for Llama-architecture language models, written in PyTorch.
It loads real Hugging Face checkpoints, generates with a KV cache, batches prompts of different
lengths, supports speculative decoding, and serves an OpenAI-compatible HTTP API with streaming.

Every component is checked against an independent reference in the test suite (PyTorch's fused
attention, a complex-number RoPE, full recomputation, and Hugging Face `transformers` on a real
checkpoint).

```
$ flashserve generate --model HuggingFaceTB/SmolLM2-135M-Instruct --chat \
    --prompt "Give me one tip for writing clean Python code."
One tip for writing clean Python code is to use the `with` statement to automatically
close files and connections when they are no longer needed. ...
```

## What is implemented

| Component | Where | Verified by |
|---|---|---|
| Llama model: RMSNorm, RoPE, SwiGLU, grouped-query attention | `flashserve/model/llama.py` | logits match `transformers` on SmolLM2-135M (max abs diff 2.5e-5) |
| Hugging Face checkpoint loader (safetensors, config.json, tokenizer, chat template) | `flashserve/model/weights.py` | greedy output identical to `transformers.generate` |
| Tiled attention with online softmax (FlashAttention algorithm) | `flashserve/attention/flash_attention.py` | matches `torch.nn.functional.scaled_dot_product_attention`, causal and non-causal |
| KV cache: prefill once, then decode one token per step | `LlamaForCausalLM.generate_iter` | identical tokens and logits to full recomputation |
| Batched generation with left padding and a combined causal + padding mask | `InferenceEngine.generate_batch` | each row matches generating that prompt alone |
| Sampling: temperature, top-k, top-p | `flashserve/utils/sampling.py` | per-row nucleus tests |
| Speculative decoding (draft model proposes, target verifies in one pass) | `flashserve/engine/speculative_decoding.py` | greedy output identical to the target model; 100% acceptance with an identical draft |
| Paged KV cache (16-token pages, page table, page reuse) | `flashserve/attention/paged_attention.py` | write/read round trip, freed pages reused |
| Continuous-batching scheduler (prefill-first) | `flashserve/engine/continuous_batching.py` | scheduling-order tests |
| INT8 / INT4 weight quantization | `flashserve/quantization/` | size and reconstruction-error tests |
| OpenAI-compatible server: `/v1/completions`, `/v1/chat/completions`, SSE streaming | `flashserve/serving/server.py` | FastAPI test client |

## Benchmarks

Decode throughput on SmolLM2-135M-Instruct, 64-token prompt, 128 new tokens, greedy, float32,
**2 vCPUs** (Intel Xeon 2.8 GHz), no GPU:

| Mode | tokens/s |
|---|---|
| KV cache (prefill + decode) | 17.7 |
| Full recomputation every step | 2.9 |

The cache makes each decode step cost one token of work instead of the whole sequence, a 6.1x
speedup at this length that grows with sequence length. Reproduce with
`python benchmarks/bench_generate.py`.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"

pytest -q                                          # unit tests (no downloads)
FLASHSERVE_HF_TESTS=1 pytest -q tests/test_hf_parity.py   # compare with transformers

flashserve generate --model HuggingFaceTB/SmolLM2-135M-Instruct --chat --prompt "Hello!"
flashserve serve    --model HuggingFaceTB/SmolLM2-135M-Instruct --port 8000
```

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "smollm2", "messages": [{"role": "user", "content": "What is RoPE?"}], "max_tokens": 64
}'
```

## Design notes

**RoPE layout.** The model rotates interleaved (even, odd) pairs, as in Meta's reference code.
Hugging Face checkpoints store `q_proj`/`k_proj` rows in the "rotate half" order instead, so the
loader applies the inverse row permutation once at load time. The forward pass stays simple and
the parity test proves the two conventions line up.

**Why left padding.** A decoder-only model continues from the last column, so batched prompts
are left-padded. The attention mask combines causality with key padding, and RoPE positions
restart at 0 on each prompt's first real token, so a padded row produces exactly the tokens it
would produce alone.

**Prefill-first scheduling.** The scheduler runs new prompts before the next decode step. That
keeps time-to-first-token low, at the cost of pausing decode for requests that are already
streaming. A short wait window (50 ms by default) groups arriving prompts into one prefill batch.
Chunked prefill would bound those pauses and is the next thing I would add.

**Speculative decoding.** The target model scores the context plus all draft tokens in one forward
pass. Drafts are accepted with probability min(1, q/p) and the first rejection is resampled from
max(0, q - p), which keeps the output distribution exactly the target's.

## Limitations

- The paged KV cache and the continuous-batching scheduler are standalone, tested components;
  the generation loop and the server use a contiguous per-request cache.
- Speculative decoding recomputes full forward passes and supports batch size 1.
- CPU numbers only; there are no custom CUDA/Triton kernels.

## Layout

```
flashserve/
  model/        config, Llama model, Hugging Face loader
  attention/    tiled attention, RoPE and masks, paged KV cache
  engine/       inference engine, speculative decoding, scheduler
  quantization/ INT8 / INT4
  serving/      FastAPI server and CLI
tests/          unit tests + optional transformers parity test
benchmarks/     bench_generate.py (real checkpoint) and synthetic benchmarks
```

MIT License.
