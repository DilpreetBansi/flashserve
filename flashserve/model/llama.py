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
from typing import Optional, Tuple
import math

from flashserve.model.config import LlamaConfig
from flashserve.attention.flash_attention import FlashAttention, GroupedQueryAttention
from flashserve.attention.attention_utils import apply_rotary_pos_emb, get_causal_mask


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
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV cache for autoregressive generation.

        Args:
            x: Input of shape (batch, seq_len, hidden_size)
            positions: Position indices for RoPE
            attention_mask: Causal mask for autoregressive attention
            use_kv_cache: Whether to cache K,V for generation

        Returns:
            Output of shape (batch, seq_len, hidden_size)
            KV cache for next iteration (if use_kv_cache=True)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)

        q = apply_rotary_pos_emb(q, positions, rope_theta=self.config.rope_theta)
        k = apply_rotary_pos_emb(k, positions, rope_theta=self.config.rope_theta)

        # Prepare causal mask for autoregressive attention
        if attention_mask is None:
            causal_mask = get_causal_mask(seq_len, device=x.device, dtype=x.dtype)
        else:
            causal_mask = attention_mask

        # Attention
        if self.num_kv_heads < self.num_heads:
            # Use GQA
            attn_output = self.gqa(q, k, v, causal_mask=causal_mask)
        else:
            # Standard attention
            attn_output, _ = self.flash_attn(q, k, v, causal_mask=causal_mask)

        # Reshape output: (batch, seq_len, num_heads * head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        output = self.o_proj(attn_output)

        # Return KV cache for autoregressive generation
        kv_cache = None
        if use_kv_cache:
            kv_cache = (k, v)

        return output, kv_cache


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
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through all layers.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            positions: Position indices
            attention_mask: Causal mask
            use_kv_cache: Whether to return KV cache

        Returns:
            Hidden states of shape (batch, seq_len, hidden_size)
            KV cache list (one per layer) or None
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Get positions
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)

        # Pass through all layers
        kv_caches = [] if use_kv_cache else None

        for layer in self.layers:
            hidden_states, kv_cache = layer(
                hidden_states,
                positions=positions,
                attention_mask=attention_mask,
                use_kv_cache=use_kv_cache,
            )
            if use_kv_cache:
                kv_caches.append(kv_cache)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states, kv_caches


class LlamaForCausalLM(nn.Module):
    """Llama model with language modeling head for autoregressive generation."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings and output weights (optional but common)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for language modeling.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            positions: Position indices
            attention_mask: Causal mask

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        hidden_states, _ = self.model(
            input_ids,
            positions=positions,
            attention_mask=attention_mask,
            use_kv_cache=False,
        )
        logits = self.lm_head(hidden_states)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)

        Returns:
            Generated token IDs of shape (batch, prompt_len + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        for _ in range(max_new_tokens):
            # Get logits for last token
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = torch.finfo(next_token_logits.dtype).min

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 0] = 0  # Keep at least one token
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = torch.finfo(next_token_logits.dtype).min

            # Sample or greedy
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)

            # Early stopping if all sequences generated EOS
            if (next_tokens == self.config.eos_token_id).all():
                break

        return input_ids

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Count total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
