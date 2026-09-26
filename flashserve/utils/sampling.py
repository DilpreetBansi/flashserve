"""Sampling helpers shared by the model, the engine and speculative decoding."""

from typing import Optional

import torch


def filter_logits(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply top-k and nucleus (top-p) filtering along the last dimension.

    Works for any leading batch shape. Filtered entries are set to the dtype's
    minimum value rather than -inf so a fully filtered row can never produce NaN.
    """
    neg = torch.finfo(logits.dtype).min
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, neg)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        # Keep the token that crosses the threshold, and always keep the best token.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_idx, remove)  # back to vocabulary order
        logits = logits.masked_fill(remove, neg)
    return logits


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> torch.Tensor:
    """Pick the next token id(s) from logits of shape (..., vocab)."""
    if not do_sample or temperature == 0:
        return torch.argmax(logits, dim=-1)
    logits = filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, num_samples=1).reshape(probs.shape[:-1])
