"""Numerical correctness tests: compare each component with an independent reference."""

import math

import pytest
import torch
import torch.nn.functional as F

from flashserve.attention.attention_utils import (
    apply_rotary_pos_emb,
    build_attention_mask,
    get_causal_mask,
    positions_from_mask,
)
from flashserve.attention.flash_attention import FlashAttention
from flashserve.attention.paged_attention import PagedKVCache
from flashserve.engine.continuous_batching import ContinuousBatchingScheduler, RequestStage
from flashserve.engine.speculative_decoding import SpeculativeDecoder
from flashserve.model.config import LlamaConfig
from flashserve.model.llama import LlamaForCausalLM
from flashserve.utils.sampling import filter_logits


def sdpa_reference(q, k, v, causal):
    """PyTorch's fused attention, inputs (batch, seq, heads, dim)."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    return out.transpose(1, 2)


@pytest.mark.parametrize("seq_len", [16, 200])  # 200 exercises the tiled path (> 4096 scores)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_matches_sdpa(seq_len, causal):
    torch.manual_seed(0)
    b, h, d = 2, 4, 32
    q, k, v = (torch.randn(b, seq_len, h, d) for _ in range(3))
    attn = FlashAttention(hidden_size=h * d, num_heads=h, block_size=64)
    mask = get_causal_mask(seq_len, device="cpu") if causal else None
    out, _ = attn(q, k, v, causal_mask=mask)
    torch.testing.assert_close(out, sdpa_reference(q, k, v, causal), atol=1e-5, rtol=1e-4)


def test_tiled_and_standard_paths_agree():
    torch.manual_seed(1)
    b, s, h, d = 1, 96, 2, 16
    q, k, v = (torch.randn(b, s, h, d) for _ in range(3))
    attn = FlashAttention(hidden_size=h * d, num_heads=h, block_size=32)
    mask = get_causal_mask(s, device="cpu")
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    standard, _ = attn._standard_attention(qt, kt, vt, causal_mask=mask)
    tiled, _ = attn._tiled_attention(qt, kt, vt, causal_mask=mask)
    torch.testing.assert_close(tiled, standard, atol=1e-5, rtol=1e-4)


def rope_complex_reference(x, positions, theta=10000.0):
    """Meta's Llama formulation: rotate interleaved pairs as complex numbers."""
    d = x.shape[-1]
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    angles = torch.outer(positions.double(), freqs)
    rot = torch.polar(torch.ones_like(angles), angles)[None, :, None, :]
    xc = torch.view_as_complex(x.double().reshape(*x.shape[:-1], d // 2, 2))
    return torch.view_as_real(xc * rot).flatten(-2).to(x.dtype)


def test_rope_matches_complex_reference():
    torch.manual_seed(2)
    x = torch.randn(2, 10, 3, 16)
    pos = torch.arange(10)
    torch.testing.assert_close(apply_rotary_pos_emb(x, pos), rope_complex_reference(x, pos), atol=1e-5, rtol=1e-5)


def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(3)
    q, k = torch.randn(1, 1, 1, 32), torch.randn(1, 1, 1, 32)

    def score(m, n):
        qm = apply_rotary_pos_emb(q, torch.tensor([m]))
        kn = apply_rotary_pos_emb(k, torch.tensor([n]))
        return (qm * kn).sum().item()

    assert score(5, 2) == pytest.approx(score(105, 102), abs=1e-4)
    assert score(0, 0) == pytest.approx(score(40, 40), abs=1e-4)


def test_rope_accepts_per_sequence_positions():
    torch.manual_seed(4)
    x = torch.randn(2, 6, 2, 8)
    pos = torch.tensor([[0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5]])
    out = apply_rotary_pos_emb(x, pos)
    torch.testing.assert_close(out[1], apply_rotary_pos_emb(x[1:], torch.arange(6))[0])


def test_attention_mask_combines_causal_and_padding():
    token_mask = torch.tensor([[False, True, True]])
    mask = build_attention_mask(token_mask)
    allowed = mask[0, 0] == 0
    expected = torch.tensor([[False, False, False], [False, True, False], [False, True, True]])
    assert torch.equal(allowed, expected)
    assert positions_from_mask(token_mask).tolist() == [[0, 0, 1]]


def tiny_model(seed=0, layers=2):
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
                      num_hidden_layers=layers, vocab_size=97, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval()


def test_left_padded_batch_matches_single_sequence_generation():
    model = tiny_model()
    short, long = torch.tensor([[5, 6, 7]]), torch.tensor([[11, 12, 13, 14, 15, 16]])
    with torch.no_grad():
        ref_short = model.generate(short, max_new_tokens=8, do_sample=False)[:, 3:]
        ref_long = model.generate(long, max_new_tokens=8, do_sample=False)[:, 6:]
        pad = model.config.pad_token_id
        batch = torch.tensor([[pad, pad, pad, 5, 6, 7], [11, 12, 13, 14, 15, 16]])
        mask = torch.tensor([[False] * 3 + [True] * 3, [True] * 6])
        out = model.generate(batch, max_new_tokens=8, do_sample=False, token_mask=mask)[:, 6:]
    n = min(out.shape[1], ref_short.shape[1], ref_long.shape[1])
    assert torch.equal(out[0, :n], ref_short[0, :n])
    assert torch.equal(out[1, :n], ref_long[0, :n])


def test_top_p_keeps_smallest_nucleus_per_row():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.05, 0.15, 0.3, 0.5]])
    kept = filter_logits(probs.log(), top_p=0.7) > torch.finfo(torch.float32).min
    # 0.5 alone < 0.7, so the 0.3 token is also needed; nothing else.
    assert kept.tolist() == [[True, True, False, False], [False, False, True, True]]


def test_top_k_keeps_k_tokens():
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    kept = filter_logits(logits, top_k=2) > torch.finfo(torch.float32).min
    assert kept.tolist() == [[False, True, True, False]]


def test_greedy_speculative_decoding_equals_target_greedy():
    target, draft = tiny_model(seed=0, layers=3), tiny_model(seed=1, layers=1)
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    with torch.no_grad():
        expected = target.generate(prompt, max_new_tokens=12, do_sample=False)
    dec = SpeculativeDecoder(target, draft, gamma=3, device="cpu")
    got = dec.generate(prompt, max_new_tokens=12, temperature=0)
    n = min(expected.shape[1], got.shape[1])
    assert torch.equal(got[:, :n], expected[:, :n])
    assert dec.stats["target_calls"] <= 12


def test_speculative_decoding_with_identical_draft_accepts_everything():
    target = tiny_model(seed=0)
    dec = SpeculativeDecoder(target, target, gamma=4, device="cpu")
    torch.manual_seed(0)
    dec.generate(torch.tensor([[2, 7, 1]]), max_new_tokens=12, temperature=1.0, top_p=1.0)
    assert dec.acceptance_rate == pytest.approx(1.0)
    # 4 drafts + 1 bonus token per target call.
    assert dec.stats["target_calls"] == math.ceil(12 / 5)


def test_paged_cache_round_trip_and_page_reuse():
    cache = PagedKVCache(page_size=4, max_pages=8, hidden_size=16, num_heads=2)
    k = torch.randn(1, 10, 2, 8)
    v = torch.randn(1, 10, 2, 8)
    pages = cache.allocate(request_id=1, num_pages=3)  # 10 tokens need ceil(10/4) = 3 pages
    cache.append_kv(1, k, v)
    k_out, v_out = cache.read_kv_paged(1)
    torch.testing.assert_close(k_out, k[0])
    torch.testing.assert_close(v_out, v[0])
    assert cache.free(1) == 3
    assert set(cache.allocate(request_id=2, num_pages=3)) == set(pages)  # freed pages are reused


def test_scheduler_runs_prefill_before_decode():
    sched = ContinuousBatchingScheduler(max_batch_size=4, max_wait_time_ms=0)
    first = sched.add_request("a", max_tokens=4)
    batch = sched.get_next_batch()
    assert batch.stage == RequestStage.PREFILL and batch.request_ids == [first]
    sched.mark_prefill_complete(first)
    second = sched.add_request("b", max_tokens=4)
    # A new prompt is prefilled before the running request gets its next decode step.
    batch = sched.get_next_batch()
    assert batch.stage == RequestStage.PREFILL and batch.request_ids == [second]
    batch = sched.get_next_batch()
    assert batch.stage == RequestStage.DECODE and batch.request_ids == [first]


@pytest.mark.parametrize("padded", [False, True])
def test_kv_cache_generation_matches_full_recompute(padded):
    model = tiny_model(seed=5, layers=2)
    if padded:
        pad = model.config.pad_token_id
        ids = torch.tensor([[pad, pad, 9, 8, 7], [1, 2, 3, 4, 5]])
        mask = torch.tensor([[False, False, True, True, True], [True] * 5])
    else:
        ids, mask = torch.tensor([[4, 8, 15, 16, 23, 42]]), None
    with torch.no_grad():
        cached = model.generate(ids, max_new_tokens=10, do_sample=False, token_mask=mask, use_cache=True)
        full = model.generate(ids, max_new_tokens=10, do_sample=False, token_mask=mask, use_cache=False)
    assert torch.equal(cached, full)


def test_cached_decode_step_matches_full_forward_logits():
    model = tiny_model(seed=6)
    ids = torch.tensor([[3, 14, 15, 92, 65, 35]])
    with torch.no_grad():
        full = model(ids)
        _, cache = model(ids[:, :-1], use_cache=True)
        step = model(ids[:, -1:], past_key_values=cache)
    torch.testing.assert_close(step[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)
