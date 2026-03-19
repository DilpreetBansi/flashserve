"""
Speculative Decoding: Accelerate generation with draft model verification.

Algorithm:
  1. Draft model generates K candidate tokens
  2. Large model verifies all K in parallel
  3. Accept matching tokens, resample where they diverge
  4. Proper probability adjustment for rejected tokens
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SpeculativeDecoder:
    """
    Decoder that uses speculative decoding for faster inference.

    Uses a small draft model to propose tokens, then verifies with the large model.
    """

    def __init__(
        self,
        large_model: nn.Module,
        draft_model: nn.Module,
        gamma: int = 4,
        device: str = "cuda",
    ):
        """
        Initialize speculative decoder.

        Args:
            large_model: Large model for verification
            draft_model: Small model for speculation
            gamma: Number of speculative tokens to generate
            device: Device to run on
        """
        self.large_model = large_model.to(device)
        self.draft_model = draft_model.to(device)
        self.gamma = gamma
        self.device = device

        self.large_model.eval()
        self.draft_model.eval()

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Generate with speculative decoding.

        Args:
            input_ids: Initial token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        generated = 0

        while generated < max_new_tokens:
            # Step 1: Draft model generates gamma candidates
            draft_logits_list = []
            draft_tokens = []

            x = input_ids
            for _ in range(min(self.gamma, max_new_tokens - generated)):
                with torch.no_grad():
                    draft_logits = self.draft_model(x)
                draft_logits_list.append(draft_logits[:, -1, :])

                # Sample from draft
                draft_token = self._sample_token(
                    draft_logits[:, -1, :],
                    temperature=temperature,
                    top_p=top_p,
                )
                draft_tokens.append(draft_token)
                x = torch.cat([x, draft_token.unsqueeze(-1)], dim=1)

            # Step 2: Large model verifies all gamma tokens in parallel
            with torch.no_grad():
                large_logits = self.large_model(input_ids)

            # Step 3: Compare and accept/reject
            accepted_count = 0
            for i, draft_token in enumerate(draft_tokens):
                # Get verification logits at this position
                verify_pos = input_ids.shape[1] + i
                verify_logits = large_logits[:, verify_pos - 1, :]

                # Compute acceptance probability
                draft_prob = torch.softmax(draft_logits_list[i], dim=-1)
                large_prob = torch.softmax(verify_logits, dim=-1)

                # Get probabilities for the draft token
                draft_prob_token = draft_prob.gather(-1, draft_token)
                large_prob_token = large_prob.gather(-1, draft_token)

                # Acceptance: min(1, p_large / p_draft)
                accept_prob = torch.min(
                    torch.ones_like(draft_prob_token),
                    large_prob_token / (draft_prob_token + 1e-10)
                )

                # Stochastic accept/reject
                if torch.rand(1).item() < accept_prob.item():
                    # Accept this token
                    input_ids = torch.cat([input_ids, draft_token.unsqueeze(-1)], dim=1)
                    accepted_count += 1
                else:
                    # Reject: resample from large model (adjusted distribution)
                    # Probability: max(0, p_large - p_draft)
                    adjusted_prob = torch.clamp(large_prob - draft_prob, min=0)
                    adjusted_prob = adjusted_prob / (adjusted_prob.sum(dim=-1, keepdim=True) + 1e-10)

                    new_token = torch.multinomial(adjusted_prob, num_samples=1)
                    input_ids = torch.cat([input_ids, new_token], dim=1)
                    break  # Stop accepting, will regenerate next speculative batch

            generated += accepted_count + 1

        return input_ids

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Sample next token with temperature and top-p."""
        if temperature > 0:
            logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = torch.finfo(logits.dtype).min

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
