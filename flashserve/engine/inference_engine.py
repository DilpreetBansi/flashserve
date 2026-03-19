"""
Core inference engine for text generation with support for batching and caching.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Generator
import time

from flashserve.model.config import LlamaConfig
from flashserve.model.llama import LlamaForCausalLM
from flashserve.utils.tokenizer import Tokenizer


class InferenceEngine:
    """
    High-level inference engine for LLM generation.

    Supports:
    - Single and batch inference
    - Streaming generation
    - Various sampling strategies (greedy, top-k, nucleus)
    - Automatic device placement
    """

    def __init__(
        self,
        config: LlamaConfig,
        model: Optional[LlamaForCausalLM] = None,
        tokenizer: Optional[Tokenizer] = None,
        device: str = "auto",
    ):
        """
        Initialize inference engine.

        Args:
            config: Model configuration
            model: Pre-instantiated model (creates new if None)
            tokenizer: Tokenizer for encoding/decoding
            device: Device to use ("cpu", "cuda", or "auto")
        """
        self.config = config
        self.device = self._select_device(device)

        # Initialize model
        if model is None:
            self.model = LlamaForCausalLM(config).to(self.device)
        else:
            self.model = model.to(self.device)

        self.model.eval()

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = Tokenizer.from_pretrained("gpt2")
        else:
            self.tokenizer = tokenizer

        # Statistics
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0

    def _select_device(self, device: str) -> torch.device:
        """Select device based on availability."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], device=self.device)

        start_time = time.time()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )

        elapsed = time.time() - start_time
        generated_ids = output_ids[0, input_ids.shape[1]:].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        # Update statistics
        self.total_tokens_generated += len(generated_ids)
        self.total_inference_time += elapsed

        return generated_text

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        batch_size: int = 4,
    ) -> List[str]:
        """
        Generate text for multiple prompts in batches.

        Args:
            prompts: List of prompt texts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample
            batch_size: Batch size for processing

        Returns:
            List of generated texts
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_inputs = [self.tokenizer.encode(p) for p in batch_prompts]

            # Pad to same length
            max_len = max(len(x) for x in batch_inputs)
            padded = []
            for ids in batch_inputs:
                ids = ids + [self.config.pad_token_id] * (max_len - len(ids))
                padded.append(ids)

            input_ids = torch.tensor(padded, device=self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )

            # Decode each sequence
            for j, output in enumerate(output_ids):
                start_idx = len(batch_inputs[j])
                generated = output[start_idx:].cpu().tolist()
                text = self.tokenizer.decode(generated)
                results.append(text)

        return results

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming output (yields tokens as they're generated).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Yields:
            Generated tokens (as strings)
        """
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], device=self.device)

        for _ in range(max_tokens):
            with torch.no_grad():
                logits = self.model(input_ids)

            next_token_logits = logits[0, -1, :]

            # Temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1]
                next_token_logits[indices_to_remove] = torch.finfo(next_token_logits.dtype).min

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = torch.finfo(next_token_logits.dtype).min

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Decode and yield
            token_text = self.tokenizer.decode([next_token])
            yield token_text

            # Append to sequence
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

            # Stop at EOS
            if next_token == self.config.eos_token_id:
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "total_inference_time_seconds": self.total_inference_time,
            "throughput_tokens_per_sec": (
                self.total_tokens_generated / self.total_inference_time
                if self.total_inference_time > 0 else 0
            ),
        }
