"""
Core inference engine for text generation with support for batching and caching.
"""

import dataclasses
import time
from typing import Any, Dict, Generator, List, Optional

import torch

from flashserve.model.config import LlamaConfig
from flashserve.model.llama import LlamaForCausalLM
from flashserve.utils.sampling import sample_next
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
        self.device = self._select_device(device)

        # Tokenizer first: the model's vocabulary has to cover every id it can emit.
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer.from_pretrained("gpt2")
        tok_vocab = getattr(self.tokenizer, "vocab_size", config.vocab_size)

        if model is None:
            if tok_vocab > config.vocab_size:
                # Randomly initialised model: size the embedding table to the tokenizer.
                config = dataclasses.replace(config, vocab_size=tok_vocab)
            self.model = LlamaForCausalLM(config).to(self.device)
        else:
            if tok_vocab > model.config.vocab_size:
                raise ValueError(
                    f"Tokenizer vocabulary ({tok_vocab}) is larger than the model's "
                    f"({model.config.vocab_size}); pass a matching tokenizer."
                )
            self.model = model.to(self.device)
            config = model.config

        self.config = config
        self.model.eval()

        # Statistics
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: str = "auto") -> "InferenceEngine":
        """Engine for a Hugging Face Llama-architecture checkpoint, with its own tokenizer."""
        from flashserve.model.weights import from_pretrained

        model, config = from_pretrained(model_name_or_path)
        tokenizer = Tokenizer.from_pretrained(model_name_or_path)
        return cls(config, model=model, tokenizer=tokenizer, device=device)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Render chat messages with the tokenizer's chat template when it has one."""
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages)
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        return "\n".join(lines) + "\nAssistant:"

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

            # Left-pad so every prompt ends at the same position, and mask the padding
            # out of attention (decoder-only models continue from the last column).
            max_len = max(len(x) for x in batch_inputs)
            padded, masks = [], []
            for ids in batch_inputs:
                pad = max_len - len(ids)
                padded.append([self.config.pad_token_id] * pad + ids)
                masks.append([False] * pad + [True] * len(ids))

            input_ids = torch.tensor(padded, device=self.device)
            token_mask = torch.tensor(masks, device=self.device)

            start_time = time.time()
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    token_mask=token_mask,
                )
            self.total_inference_time += time.time() - start_time

            # Decode each sequence (everything after the shared prompt width),
            # dropping anything after the first EOS.
            for output in output_ids:
                generated = output[max_len:].cpu().tolist()
                if self.config.eos_token_id in generated:
                    generated = generated[: generated.index(self.config.eos_token_id)]
                self.total_tokens_generated += len(generated)
                results.append(self.tokenizer.decode(generated))

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
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        start_time = time.time()
        produced = 0
        for next_tokens in self.model.generate_iter(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=temperature > 0,
        ):
            token = int(next_tokens[0])
            if token == self.config.eos_token_id:
                break
            produced += 1
            yield self.tokenizer.decode([token])
        self.total_tokens_generated += produced
        self.total_inference_time += time.time() - start_time

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
