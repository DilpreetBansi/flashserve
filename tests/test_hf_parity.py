"""
Parity with Hugging Face transformers on a real checkpoint.

Opt-in because it downloads ~270 MB:  FLASHSERVE_HF_TESTS=1 pytest tests/test_hf_parity.py
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLASHSERVE_HF_TESTS") != "1", reason="set FLASHSERVE_HF_TESTS=1")

MODEL_ID = os.environ.get("FLASHSERVE_HF_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")


@pytest.fixture(scope="module")
def models():
    transformers = pytest.importorskip("transformers")
    from flashserve.model.weights import from_pretrained
    from flashserve.utils.tokenizer import Tokenizer

    ours, _ = from_pretrained(MODEL_ID)
    ref = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    return ours, ref, Tokenizer.from_pretrained(MODEL_ID)


def test_logits_match_transformers(models):
    ours, ref, tok = models
    ids = torch.tensor([tok.encode("The capital of France is Paris. The capital of Japan is")])
    with torch.no_grad():
        torch.testing.assert_close(ours(ids), ref(ids).logits, atol=1e-3, rtol=1e-3)


def test_greedy_generation_matches_transformers(models):
    ours, ref, tok = models
    ids = torch.tensor([tok.encode("def fibonacci(n):")])
    with torch.no_grad():
        a = ours.generate(ids, max_new_tokens=24, do_sample=False)
        b = ref.generate(ids, max_new_tokens=24, do_sample=False)
    n = min(a.shape[1], b.shape[1])
    assert torch.equal(a[:, :n], b[:, :n])
