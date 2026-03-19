# FlashServe Module Reference

Complete module listing with descriptions and key classes/functions.

## Core Package: `flashserve/`

### `__init__.py`
Public API exports.
- `LlamaConfig`: Model configuration
- `InferenceEngine`: Main inference interface

---

## Attention Module: `flashserve/attention/`

### `flash_attention.py` (280 lines)
Flash Attention implementation with tiled computation.

**Classes**:
- `FlashAttention`: Memory-efficient attention with O(N) memory vs O(N²)
  - `forward()`: Tiled attention computation
  - `_standard_attention()`: Fallback for small sequences
  - `_tiled_attention()`: Main tiling algorithm

- `GroupedQueryAttention`: GQA for efficient KV cache
  - Shares K,V heads across query heads

**Algorithm**: Processes queries in blocks, iterates over KV blocks with online softmax to avoid materializing N×N matrix.

### `paged_attention.py` (280 lines)
Paged KV cache management for dynamic allocation.

**Classes**:
- `PageTableEntry`: Maps logical to physical blocks
- `PageStats`: Cache utilization statistics
- `PagedKVCache`: KV cache manager
  - `allocate()`: Allocate pages for request
  - `free()`: Deallocate pages
  - `append_kv()`: Add KV to cache
  - `read_kv_paged()`: Read with page table lookup
  - `copy_on_write()`: Parallel sampling support
  - `get_stats()`: Cache utilization metrics

**Design**: Fixed-size pages (16 tokens) reduce fragmentation and enable efficient page reuse across requests.

### `attention_utils.py` (160 lines)
Attention utility functions.

**Functions**:
- `apply_rotary_pos_emb()`: Rotary Position Embeddings (RoPE)
- `get_causal_mask()`: Create causal attention mask
- `compute_attention_scores()`: Q @ K^T / sqrt(d)
- `apply_attention_mask()`: Apply masking before softmax
- `softmax_with_mask()`: Masked softmax

---

## Model Module: `flashserve/model/`

### `config.py` (170 lines)
Model configurations.

**Classes**:
- `LlamaConfig`: Configuration dataclass
  - `tiny()`: 15M params, CPU-friendly (2 layers, 256D)
  - `small()`: 110M params (6 layers, 512D)
  - `llama2_7b()`: 7B params (32 layers, 4096D)
  - `llama2_13b()`: 13B params
  - `llama2_70b()`: 70B params
  - `to_dict()`, `from_dict()`: Serialization

### `llama.py` (520 lines)
Llama-2 architecture from scratch.

**Classes**:
- `RMSNorm`: Root Mean Square Normalization (vs LayerNorm)
- `SwiGLU`: Swish-gated linear units activation
- `LlamaAttention`: Multi-head attention with GQA support
- `LlamaMLPBlock`: Feed-forward with SwiGLU
- `LlamaDecoderLayer`: Single transformer layer
- `LlamaModel`: Full model (embeddings + layers + norm)
- `LlamaForCausalLM`: Model + language modeling head
  - `forward()`: Forward pass
  - `generate()`: Autoregressive generation with sampling
  - `get_num_params()`: Parameter counting

**Design**: Pre-norm residual connections, no LayerNorm bias, supports KV cache.

### `weights.py` (110 lines)
Weight loading from HuggingFace format.

**Functions**:
- `load_hf_weights()`: Load HF state dict into model
- `convert_hf_state_dict()`: Convert HF names to our names
- `load_from_pretrained()`: Download and load from HuggingFace
- `estimate_model_size()`: Size estimation

---

## Engine Module: `flashserve/engine/`

### `inference_engine.py` (240 lines)
Core inference engine.

**Classes**:
- `InferenceEngine`: High-level generation interface
  - `generate()`: Single text generation
  - `generate_batch()`: Batch generation
  - `stream_generate()`: Streaming token generation
  - `get_stats()`: Collection statistics

**Features**: Automatic device selection, sampling strategies (greedy, top-k, nucleus), streaming support.

### `continuous_batching.py` (220 lines)
Continuous batching scheduler for request management.

**Classes**:
- `RequestStage`: Enum (WAITING, PREFILL, DECODE)
- `Request`: Request metadata
- `Batch`: Batch of ready requests
- `ContinuousBatchingScheduler`: Dynamic scheduler
  - `add_request()`: Queue new request
  - `get_next_batch()`: Get batch when ready
  - `mark_prefill_complete()`: Stage transition
  - `mark_request_complete()`: Request completion
  - `get_queue_stats()`: Queue statistics

**Algorithm**: Prioritizes prefill (minimize TTFT), timeout-based batching, iteration-level scheduling.

### `speculative_decoding.py` (120 lines)
Speculative decoding with draft model verification.

**Classes**:
- `SpeculativeDecoder`: Draft + verify architecture
  - `generate()`: Generate with speculation
  - `_sample_token()`: Sampling with top-k/nucleus

**Algorithm**: Draft model proposes K tokens, large model verifies all in parallel with proper probability adjustment.

### `kv_cache.py` (150 lines)
Linear KV cache manager (simpler alternative to paged).

**Classes**:
- `KVCacheManager`: Manages per-request allocations
  - `allocate()`: Pre-allocate space
  - `append()`: Add KV values
  - `read()`: Retrieve cached values
  - `get_stats()`: Usage statistics

### `scheduler.py` (120 lines)
Request scheduling policies.

**Classes**:
- `SchedulingPolicy`: Enum (FCFS, PRIORITY)
- `ScheduledRequest`: Request with priority metadata
- `RequestScheduler`: Base scheduler
- `FCFSScheduler`: First-Come, First-Served
- `PriorityScheduler`: Priority-based scheduling

---

## Serving Module: `flashserve/serving/`

### `server.py` (180 lines)
FastAPI serving layer with OpenAI-compatible API.

**Functions**:
- `create_app()`: Create FastAPI application
- `_stream_completions()`: Server-Sent Events streaming

**Endpoints**:
- `GET /health`: Health check with metrics
- `POST /v1/completions`: Text completion (streaming support)
- `POST /v1/chat/completions`: Chat completion
- `GET /v1/models`: List available models

**API Format**: OpenAI-compatible for drop-in replacement.

### `request.py` (110 lines)
Request/Response dataclasses.

**Classes**:
- `CompletionRequest`: Text completion input
- `CompletionChoice`: Single completion
- `CompletionUsage`: Token usage stats
- `CompletionResponse`: Completion output
- `ChatMessage`: Chat message
- `ChatCompletionRequest`: Chat input
- `ChatCompletionChoice`: Chat completion
- `ChatCompletionResponse`: Chat output

### `batch_manager.py` (100 lines)
Micro-batching with timeout.

**Classes**:
- `BatchedRequest`: Request in batch
- `BatchManager`: Collects into batches
  - `add_request()`: Queue request
  - `get_next_batch()`: Return ready batch
  - `batch_ready()`: Status check

### `health.py` (60 lines)
Health monitoring and metrics.

**Classes**:
- `HealthStatus`: Health information

**Functions**:
- `get_device_memory()`: GPU/CPU memory stats
- `get_system_stats()`: System utilization

---

## Quantization Module: `flashserve/quantization/`

### `int8_quantize.py` (190 lines)
INT8 per-channel weight quantization.

**Functions**:
- `quantize_int8()`: Quantize to INT8
- `dequantize_int8()`: Dequantize to FP32

**Classes**:
- `Int8LinearLayer`: Linear layer with INT8 weights

**Features**: Per-channel scales (better accuracy), 4x size reduction.

### `int4_quantize.py` (180 lines)
INT4 group-wise quantization (GPTQ-inspired).

**Functions**:
- `quantize_int4()`: Group-wise INT4 quantization
- `dequantize_int4()`: Unpack and scale

**Classes**:
- `Int4LinearLayer`: Linear layer with INT4 weights

**Features**: Group-wise scales, 8x size reduction, minimal accuracy loss.

### `calibration.py` (60 lines)
Quantization calibration utilities.

**Functions**:
- `calibrate_model()`: Run calibration dataset
- `compute_layer_scales()`: Compute optimal scales

---

## Kernels Module: `flashserve/kernels/`

### `triton_attention.py` (60 lines)
Triton flash attention with PyTorch fallback.

**Functions**:
- `triton_attention_forward()`: Try Triton, fall back to PyTorch
- `_pytorch_attention_forward()`: Optimized PyTorch implementation

### `fused_rmsnorm.py` (40 lines)
Fused RMSNorm kernel.

**Functions**:
- `fused_rms_norm()`: Normalize + scale in one operation

### `fused_rotary.py` (90 lines)
Fused rotary position embeddings.

**Functions**:
- `fused_rotary_pos_emb()`: Apply RoPE without intermediate tensors

### `fused_silu.py` (20 lines)
Fused SiLU activation.

**Functions**:
- `fused_silu()`: x * sigmoid(x) in one kernel

---

## Utils Module: `flashserve/utils/`

### `tokenizer.py` (140 lines)
Tokenizer wrapper and utilities.

**Classes**:
- `Tokenizer`: Fallback character-level tokenizer
  - `encode()`: Text → token IDs
  - `decode()`: Token IDs → text
  - `from_pretrained()`: Load from HuggingFace

- `HFTokenizer`: HuggingFace tokenizer wrapper

### `metrics.py` (90 lines)
Performance metrics tracking.

**Classes**:
- `MetricsTracker`: TTFT, TPOT, throughput tracking
  - `record_first_token()`: Track TTFT
  - `record_token()`: Track each token
  - `get_stats()`: Aggregate statistics

### `profiler.py` (100 lines)
GPU/CPU profiling utilities.

**Classes**:
- `GPUProfiler`: GPU memory and timing profiling

**Functions**:
- `profile_forward_pass()`: Benchmark forward pass

### `config.py` (70 lines)
Configuration management.

**Classes**:
- `Config`: Config loading and access
  - `from_yaml()`, `from_json()`, `from_env()`
  - `get()`, `to_dict()`

---

## Examples: `examples/`

### `01_basic_inference.py`
Single prompt text generation with tiny model (CPU-compatible).

### `02_batched_inference.py`
Batch generation with multiple prompts.

### `03_streaming.py`
Streaming token generation with real-time output.

### `04_serve_model.py`
Launch FastAPI server with OpenAI-compatible API.

---

## Benchmarks: `benchmarks/`

### `benchmark_attention.py`
Attention kernel performance (memory, latency).

### `benchmark_throughput.py`
End-to-end throughput (tokens/sec, requests/sec).

### `benchmark_latency.py`
Latency breakdown (TTFT, TPOT, variance).

---

## Tests: `tests/`

### `test_flash_attention.py` (60 lines)
Tests for Flash Attention shapes, causality, stability.

### `test_model.py` (100 lines)
Tests for model config, generation, parameter counting.

### `test_quantization.py` (100 lines)
Tests for INT8/INT4 quantization accuracy.

### `test_engine.py` (110 lines)
Tests for inference engine, batching, scheduling.

---

## Summary Statistics

| Category | Files | Lines | Purpose |
|----------|-------|-------|---------|
| Attention | 3 | 720 | Flash attention, paged KV cache |
| Model | 3 | 800 | Llama architecture, configs |
| Engine | 5 | 750 | Inference, batching, caching |
| Serving | 4 | 450 | FastAPI server, API |
| Quantization | 3 | 430 | INT8/INT4 quantization |
| Kernels | 5 | 210 | Fused operations |
| Utils | 4 | 400 | Tokenizer, metrics, profiler |
| Examples | 4 | 200 | Usage demonstrations |
| Benchmarks | 3 | 150 | Performance measurements |
| Tests | 5 | 400 | Unit tests |
| **Total** | **47** | **4,590** | **Production LLM inference engine** |

---

## Import Hierarchy

```python
# High-level API
from flashserve import LlamaConfig, InferenceEngine

# Specific components
from flashserve.attention import FlashAttention, PagedKVCache
from flashserve.model import LlamaForCausalLM
from flashserve.engine import ContinuousBatchingScheduler, SpeculativeDecoder
from flashserve.serving import create_app
from flashserve.quantization import quantize_int8, quantize_int4
from flashserve.utils import Tokenizer, MetricsTracker
```

---

## Design Principles

1. **Modularity**: Each module has single responsibility
2. **Type Safety**: Full type hints throughout
3. **Testability**: Clean interfaces, minimal coupling
4. **Performance**: Optimized operations, minimal overhead
5. **Extensibility**: Easy to add new models, quantization methods
6. **Documentation**: Comprehensive docstrings and examples

---

*Last updated: March 2026*
