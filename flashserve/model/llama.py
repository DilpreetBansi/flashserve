"""
Llama-2 style transformer model architecture.

Implements:
- RMSNorm instead of LayerNorm
- SwiGLU activation instead of GELU
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) for efficient inference
- Full forward pass with KV cache support for autoregressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, List, Optional, Tuple
import math

from flashserve.model.config import LlamaConfig
from flashserve.attention.flash_attention import FlashAttention, GroupedQueryAttention
from flashserve.attention.attention_utils import (
    apply_rotary_pos_emb,
    build_attention_mask,
    get_causal_mask,
    positions_from_mask,
)
from flashserve.utils.sampling import sample_next


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    More stable than LayerNorm, used in Llama models.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm: x / (sqrt(mean(x^2)) + eps) * weight"""
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / norm) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function: (x @ W + b) * sigmoid(x @ V + c)

    More effective than GELU, used in Llama models.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.v = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.out = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: (W(x) * sigmoid(V(x))) -> out"""
        return self.out(self.w(x) * torch.sigmoid(self.v(x)))


class LlamaAttention(nn.Module):
    """Llama multi-head self-attention with Grouped Query Attention support."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Flash attention
        self.flash_attn = FlashAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_dropout,
        )

        # GQA support
        if self.num_kv_heads < self.num_heads:
            self.gqa = GroupedQueryAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )

        # For autoregressive generation with KV cache
        self.kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV cache for autoregressive generation.

        Args:
            x: Input of shape (batch, new_tokens, hidden_size)
            positions: Absolute positions for RoPE, (new_tokens,) or (batch, new_tokens)
            attention_mask: Additive mask of shape (batch or 1, 1, new_tokens, past + new_tokens)
            use_kv_cache: Whether to return the updated K,V cache
            past_kv: Cached (K, V) from earlier steps, each (batch, past, num_kv_heads, head_dim)

        Returns:
            Output of shape (batch, new_tokens, hidden_size)
            Updated (K, V) cache if use_kv_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V: (batch, seq_len, heads, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        past_len = 0 if past_kv is None else past_kv[0].shape[1]
        if positions is None:
            positions = torch.arange(past_len, past_len + seq_len, device=x.device)

        q = apply_rotary_pos_emb(q, positions, rope_theta=self.config.rope_theta)
        k = apply_rotary_pos_emb(k, positions, rope_theta=self.config.rope_theta)

        # Keys/values are cached before GQA expansion, so the cache stays
        # num_kv_heads wide (that is the memory saving GQA exists for).
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)

        if attention_mask is None:
            causal_mask = get_causal_mask(k.shape[1], device=x.device, dtype=x.dtype)[..., -seq_len:, :]
        else:
            causal_mask = attention_mask

        if self.num_kv_heads < self.num_heads:
            attn_output = self.gqa(q, k, v, causal_mask=causal_mask)
        else:
            attn_output, _ = self.flash_attn(q, k, v, causal_mask=causal_mask)

        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)
        return output, ((k, v) if use_kv_cache else None)


class LlamaMLPBlock(nn.Module):
    """Llama feed-forward block with SwiGLU activation."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: down(gate(x) * silu(up(x)))"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Single Llama transformer layer (attention + FFN + normalization)."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLPBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with pre-norm residual connections.

        Follows: x -> norm -> attn -> residual -> norm -> mlp -> residual
        """
        # Self-attention with pre-norm
        normed = self.input_layernorm(hidden_states)
        attn_output, kv_cache = self.self_attn(
            normed,
            positions=positions,
            attention_mask=attention_mask,
            use_kv_cache=use_kv_cache,
            past_kv=past_kv,
        )
        hidden_states = hidden_states + attn_output

        # FFN with pre-norm
        normed = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed)
        hidden_states = hidden_states + mlp_output

        return hidden_states, kv_cache


class LlamaModel(nn.Module):
    """Llama transformer model (embeddings + layers + final norm)."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through all layers.

        Args:
            input_ids: Token IDs of shape (batch, new_tokens)
            positions: Absolute positions of the new tokens
            attention_mask: Additive mask (see LlamaAttention.forward)
            use_kv_cache: Whether to return per-layer K,V caches
            past_key_values: Per-layer caches from a previous call

        Returns:
            Hidden states of shape (batch, new_tokens, hidden_size)
            List of per-layer (K, V) caches, or None
        """
        hidden_states = self.embed_tokens(input_ids)
        kv_caches = [] if use_kv_cache else None

        for i, layer in enumerate(self.layers):
            hidden_states, kv_cache = layer(
                hidden_states,
                positions=positions,
                attention_mask=attention_mask,
                use_kv_cache=use_kv_cache,
                past_kv=None if past_key_values is None else past_key_values[i],
            )
            if use_kv_cache:
                kv_caches.append(kv_cache)

        return self.norm(hidden_states), kv_caches


class LlamaForCausalLM(nn.Module):
    """Llama model with language modeling head for autoregressive generation."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie input embeddings and output projection (as in SmolLM2 / Llama-3.2 1B).
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        """
        Forward pass for language modeling.

        Args:
            input_ids: Token IDs of shape (batch, new_tokens)
            positions: Absolute positions of the new tokens
            attention_mask: Additive attention mask
            past_key_values: Per-layer K,V caches from earlier steps
            use_cache: If True, also return the updated caches

        Returns:
            Logits of shape (batch, new_tokens, vocab_size), plus caches if use_cache
        """
        hidden_states, caches = self.model(
            input_ids,
            positions=positions,
            attention_mask=attention_mask,
            use_kv_cache=use_cache,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)
        return (logits, caches) if use_cache else logits

    @torch.no_grad()
    def generate_iter(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        do_sample: bool = True,
        token_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Iterator[torch.Tensor]:
        """
        Yield one (batch,) tensor of new token ids per step.

        With use_cache=True the prompt is processed once (prefill) and each later
        step feeds only the newest token, reusing cached keys and values (decode).
        With use_cache=False every step recomputes the whole sequence; this is the
        slow reference path the cached path is tested against.
        """
        if token_mask is None:
            token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        dtype = self.lm_head.weight.dtype
        caches = None
        step_input = input_ids

        for _ in range(max_new_tokens):
            positions = positions_from_mask(token_mask)
            mask = build_attention_mask(token_mask, dtype=dtype)
            if use_cache and caches is not None:
                positions = positions[:, -step_input.shape[1]:]
                mask = mask[:, :, -step_input.shape[1]:, :]
                logits, caches = self.forward(step_input, positions=positions, attention_mask=mask,
                                              past_key_values=caches, use_cache=True)
            elif use_cache:
                logits, caches = self.forward(step_input, positions=positions, attention_mask=mask, use_cache=True)
            else:
                logits = self.forward(input_ids, positions=positions, attention_mask=mask)

            next_tokens = sample_next(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p, do_sample=do_sample
            )
            # Sequences that already finished keep emitting EOS.
            next_tokens = torch.where(finished, torch.full_like(next_tokens, self.config.eos_token_id), next_tokens)
            yield next_tokens

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=1)
            token_mask = torch.cat([token_mask, torch.ones_like(next_tokens[:, None], dtype=torch.bool)], dim=1)
            step_input = next_tokens[:, None]
            finished |= next_tokens == self.config.eos_token_id
            if finished.all():
                return

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        do_sample: bool = True,
        token_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch, prompt_len). Batched
                prompts of different lengths should be left-padded.
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 means greedy)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)
            token_mask: Optional (batch, prompt_len) bool mask, False on padding
            use_cache: Reuse keys/values between steps (prefill + decode)

        Returns:
            Token IDs of shape (batch, prompt_len + generated), stopping early
            once every sequence has produced EOS.
        """
        new = list(self.generate_iter(input_ids, max_new_tokens, temperature, top_k, top_p,
                                      do_sample, token_mask, use_cache))
        if not new:
            return input_ids
        return torch.cat([input_ids, torch.stack(new, dim=1)], dim=1)

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Count total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
