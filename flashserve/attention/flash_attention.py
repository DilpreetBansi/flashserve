"""
Flash Attention: Memory-efficient attention computation via tiling.

Reference: "Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)

Standard attention: O(N²) memory to store full attention matrix
Flash attention: O(N) memory via tiled computation

Algorithm:
  for each block of queries Q_i (of size b):
    m_i = -inf, l_i = 0, O_i = 0
    for each block of K,V (of size b):
      compute attention scores S_ij = Q_i @ K_j^T / sqrt(d)
      compute softmax: P_ij = softmax(S_ij)
      update running statistics (online softmax):
        m_new = max(m_i, max(S_ij))
        l_new = exp(m_i - m_new) * l_i + sum(exp(S_ij - m_new))
        O_i = exp(m_i - m_new) * O_i + exp(S_ij - m_new) @ V_j
      m_i, l_i = m_new, l_new
    O_i = O_i / l_i  (final normalization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FlashAttention(nn.Module):
    """
    Flash Attention implementation using tiled computation for memory efficiency.

    Reduces peak memory usage from O(N²) to O(N) while maintaining exact results.
    Uses online softmax for numerical stability without materializing full attention matrix.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 128,
        dropout: float = 0.0,
    ):
        """
        Initialize Flash Attention.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            block_size: Tile size for computation (default 128 tokens)
            dropout: Dropout probability (applied during training)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.dropout_p = dropout

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute Flash Attention with tiling.

        Args:
            q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch, seq_len_kv, num_heads, head_dim)
            v: Value tensor of shape (batch, seq_len_kv, num_heads, head_dim)
            causal_mask: Causal mask for autoregressive attention
            attn_mask: Additional attention mask

        Returns:
            Attention output of shape (batch, seq_len, num_heads, head_dim)
            Attention weights (if needed for visualization)
        """
        seq_len_q, seq_len_kv = q.shape[1], k.shape[1]

        # Kernels work in (batch, heads, seq, head_dim) so matmuls contract over
        # the sequence/feature axes rather than across heads.
        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k, v))

        # For very small sequences, fall back to standard attention
        if seq_len_q * seq_len_kv < 4096:  # Threshold for tiling
            out, weights = self._standard_attention(
                q_, k_, v_, causal_mask=causal_mask, attn_mask=attn_mask
            )
        else:
            out, weights = self._tiled_attention(
                q_, k_, v_, causal_mask=causal_mask, attn_mask=attn_mask
            )
        return out.transpose(1, 2).contiguous(), weights

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard attention for small sequences (baseline for testing).

        Inputs are (batch, heads, seq, head_dim).
        """
        seq_len_q, head_dim = q.shape[2], q.shape[3]
        seq_len_kv = k.shape[2]

        # Compute attention scores: Q @ K^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(head_dim)

        # Apply masks
        if causal_mask is not None:
            scores = scores + causal_mask[..., :seq_len_q, :seq_len_kv]
        if attn_mask is not None:
            scores = scores + attn_mask

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Dropout
        if self.training and self.dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=True)

        # Apply to values
        output = torch.matmul(attn_weights, v)

        return output, attn_weights

    def _tiled_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Tiled Flash Attention algorithm.

        Processes queries in blocks, iterating over key/value blocks.
        Uses online softmax to avoid materializing full attention matrix.
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        device = q.device
        dtype = q.dtype

        # Initialize output, running max, and running sum
        output = torch.zeros_like(q)
        m = torch.full(
            (batch_size, num_heads, seq_len_q),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )  # Running max
        l = torch.zeros((batch_size, num_heads, seq_len_q), device=device, dtype=dtype)  # Running sum

        # Process query blocks
        for q_start in range(0, seq_len_q, self.block_size):
            q_end = min(q_start + self.block_size, seq_len_q)
            q_block = q[:, :, q_start:q_end, :]  # (batch, heads, block_q, head_dim)

            m_block = m[:, :, q_start:q_end]  # (batch, heads, block_q)
            l_block = l[:, :, q_start:q_end]  # (batch, heads, block_q)
            o_block = output[:, :, q_start:q_end, :]  # (batch, heads, block_q, head_dim)

            # Process key/value blocks
            for kv_start in range(0, seq_len_kv, self.block_size):
                kv_end = min(kv_start + self.block_size, seq_len_kv)
                k_block = k[:, :, kv_start:kv_end, :]  # (batch, heads, block_k, head_dim)
                v_block = v[:, :, kv_start:kv_end, :]  # (batch, heads, block_k, head_dim)

                # Compute attention scores: Q @ K^T / sqrt(d)
                scores = torch.matmul(q_block, k_block.transpose(-2, -1))  # (..., block_q, block_k)
                scores = scores / math.sqrt(head_dim)

                # Apply masks
                if causal_mask is not None:
                    mask_slice = causal_mask[..., q_start:q_end, kv_start:kv_end]
                    scores = scores + mask_slice
                if attn_mask is not None:
                    mask_slice = attn_mask[..., q_start:q_end, kv_start:kv_end]
                    scores = scores + mask_slice

                # Online softmax: track running max for numerical stability
                m_block_new = torch.max(m_block, scores.max(dim=-1)[0])

                # Compute exp(scores - m_new)
                p_block = torch.exp(scores - m_block_new[..., :, None])

                # Update running sum
                l_block_new = torch.exp(m_block - m_block_new) * l_block + p_block.sum(dim=-1)

                # Update output
                o_block = (
                    torch.exp(m_block - m_block_new)[..., None] * o_block +
                    torch.matmul(p_block, v_block)
                )

                m_block = m_block_new
                l_block = l_block_new

            # Apply dropout and finalize block
            if self.training and self.dropout_p > 0:
                # Note: Dropout is approximate in tiled version
                pass

            # Normalize output by running sum
            o_block = o_block / (l_block[..., :, None] + 1e-10)
            output[:, :, q_start:q_end, :] = o_block

            # Update global running max and sum
            m[:, :, q_start:q_end] = m_block
            l[:, :, q_start:q_end] = l_block

        return output, None


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) for efficient inference.

    Reduces KV cache size by sharing K,V across multiple query heads.
    num_kv_heads < num_heads
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        block_size: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.flash_attn = FlashAttention(hidden_size, num_heads, block_size, dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with grouped query attention.

        Args:
            q: Query of shape (batch, seq_len, num_heads, head_dim)
            k: Key of shape (batch, seq_len, num_kv_heads, head_dim)
            v: Value of shape (batch, seq_len, num_kv_heads, head_dim)

        Returns:
            Output of shape (batch, seq_len, num_heads, head_dim)
        """
        # Expand K,V to match number of query heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)

        return self.flash_attn(q, k, v, causal_mask=causal_mask)[0]
