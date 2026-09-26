"""
Speculative decoding: a small draft model proposes tokens, the large model verifies them.

Algorithm (Leviathan et al., 2023; Chen et al., 2023), per step:
  1. The draft model samples gamma tokens autoregressively (cheap).
  2. The target model scores context + all gamma drafts in ONE forward pass.
  3. Draft token i is accepted with probability min(1, q_i(x) / p_i(x)), where q is the
     target distribution and p the draft distribution at that position.
  4. On the first rejection, a replacement is sampled from the residual
     distribution max(0, q - p) (renormalised) and the step ends.
  5. If all gamma drafts are accepted, one bonus token is sampled from the target's
     distribution at the last position.

The accept/resample rule makes the output distribution identical to sampling from
the target model alone. With temperature 0 the rule reduces to "accept while the
draft matches the target's argmax", so the output equals plain greedy decoding.

This reference implementation recomputes full forward passes (no KV cache) and
supports batch size 1.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from flashserve.utils.sampling import filter_logits


class SpeculativeDecoder:
    """Speculative decoding with a draft model and a target ("large") model."""

    def __init__(
        self,
        large_model: nn.Module,
        draft_model: nn.Module,
        gamma: int = 4,
        device: str = "cpu",
    ):
        """
        Args:
            large_model: Target model whose distribution the output follows
            draft_model: Small model used to propose tokens
            gamma: Number of draft tokens proposed per verification pass
            device: Device to run on
        """
        if gamma < 1:
            raise ValueError("gamma must be >= 1")
        self.large_model = large_model.to(device)
        self.draft_model = draft_model.to(device)
        self.gamma = gamma
        self.device = device
        self.large_model.eval()
        self.draft_model.eval()
        self.stats: Dict[str, int] = {"proposed": 0, "accepted": 0, "target_calls": 0}

    @staticmethod
    def _probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Next-token distribution after temperature and nucleus filtering."""
        if temperature == 0:
            return torch.softmax(logits, dim=-1)
        return torch.softmax(filter_logits(logits / temperature, top_p=top_p), dim=-1)

    @staticmethod
    def _pick(probs: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return probs.argmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Prompt token IDs of shape (1, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            eos_token_id: Optional id that ends generation early

        Returns:
            Token IDs of shape (1, seq_len + n) with n <= max_new_tokens
        """
        if input_ids.shape[0] != 1:
            raise ValueError("SpeculativeDecoder supports batch size 1")
        self.stats = {"proposed": 0, "accepted": 0, "target_calls": 0}
        start_len = input_ids.shape[1]

        while input_ids.shape[1] - start_len < max_new_tokens:
            k = min(self.gamma, max_new_tokens - (input_ids.shape[1] - start_len))

            # 1. Draft k tokens.
            x = input_ids
            draft_probs, draft_tokens = [], []
            for _ in range(k):
                p = self._probs(self.draft_model(x)[:, -1, :], temperature, top_p)
                t = self._pick(p, temperature)
                draft_probs.append(p)
                draft_tokens.append(t)
                x = torch.cat([x, t[:, None]], dim=1)

            # 2. One target pass over the context plus every draft token.
            target_logits = self.large_model(x)
            self.stats["target_calls"] += 1
            n_ctx = input_ids.shape[1]

            # 3-5. Accept a prefix of the drafts, then fix or extend.
            new_tokens = []
            n_accepted = 0
            for i in range(k):
                q = self._probs(target_logits[:, n_ctx - 1 + i, :], temperature, top_p)
                p, t = draft_probs[i], draft_tokens[i]
                if temperature == 0:
                    accepted = bool((q.argmax(dim=-1) == t).item())
                else:
                    ratio = q.gather(-1, t[:, None]) / p.gather(-1, t[:, None]).clamp_min(1e-12)
                    accepted = torch.rand(()).item() < min(1.0, ratio.item())
                if accepted:
                    new_tokens.append(t)
                    n_accepted += 1
                    continue
                if temperature == 0:
                    fix = q.argmax(dim=-1)
                else:
                    residual = (q - p).clamp_min(0)
                    total = residual.sum(dim=-1, keepdim=True)
                    residual = residual / total if total.item() > 0 else q
                    fix = torch.multinomial(residual, num_samples=1).squeeze(-1)
                new_tokens.append(fix)
                break
            else:
                q = self._probs(target_logits[:, -1, :], temperature, top_p)
                new_tokens.append(self._pick(q, temperature))

            self.stats["proposed"] += k
            self.stats["accepted"] += n_accepted
            input_ids = torch.cat([input_ids] + [t[:, None] for t in new_tokens], dim=1)

            if eos_token_id is not None and any(int(t) == eos_token_id for t in new_tokens):
                cut = input_ids[0, start_len:].tolist().index(eos_token_id)
                return input_ids[:, : start_len + cut + 1]

        return input_ids[:, : start_len + max_new_tokens]

    @property
    def acceptance_rate(self) -> float:
        return self.stats["accepted"] / self.stats["proposed"] if self.stats["proposed"] else 0.0
