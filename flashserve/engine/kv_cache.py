"""
KV Cache Manager: High-level interface for managing KV caches across requests.
"""

import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Statistics about KV cache usage."""
    total_allocated_tokens: int
    total_free_tokens: int
    num_active_requests: int
    cache_utilization_percent: float


class KVCacheManager:
    """
    Manages KV cache for multiple concurrent requests.

    Wraps paged KV cache and provides per-request allocation/deallocation.
    """

    def __init__(
        self,
        max_cache_tokens: int = 1_000_000,
        hidden_size: int = 4096,
        num_heads: int = 32,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize KV cache manager.

        Args:
            max_cache_tokens: Total tokens the cache can store
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            dtype: Data type for cache
            device: Device to allocate on
        """
        self.max_cache_tokens = max_cache_tokens
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dtype = dtype
        self.device = device

        # Pre-allocate K,V cache: (max_tokens, num_heads, head_dim)
        self.k_cache = torch.zeros(
            (max_cache_tokens, num_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (max_cache_tokens, num_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )

        # Track allocation: request_id -> (start_idx, length)
        self.allocations: Dict[int, Tuple[int, int]] = {}
        self.next_free_idx = 0
        self.request_lengths: Dict[int, int] = {}

    def allocate(self, request_id: int, initial_tokens: int) -> None:
        """
        Pre-allocate cache space for a request.

        Args:
            request_id: Unique request identifier
            initial_tokens: Expected initial token count (prompt)
        """
        if self.next_free_idx + initial_tokens > self.max_cache_tokens:
            raise RuntimeError(
                f"Insufficient cache space: need {initial_tokens}, "
                f"have {self.max_cache_tokens - self.next_free_idx}"
            )

        self.allocations[request_id] = (self.next_free_idx, 0)
        self.request_lengths[request_id] = 0
        self.next_free_idx += initial_tokens

    def free(self, request_id: int) -> None:
        """Free cache allocation for a request."""
        if request_id in self.allocations:
            del self.allocations[request_id]
        if request_id in self.request_lengths:
            del self.request_lengths[request_id]

    def append(
        self,
        request_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Append K,V values to cache for a request.

        Args:
            request_id: Request identifier
            k: Key tensor of shape (batch, new_tokens, num_heads, head_dim)
            v: Value tensor of shape (batch, new_tokens, num_heads, head_dim)
        """
        if request_id not in self.allocations:
            raise ValueError(f"Request {request_id} not allocated")

        start_idx, _ = self.allocations[request_id]
        current_len = self.request_lengths[request_id]
        new_tokens = k.shape[1]

        # Write to cache
        k = k.squeeze(0)  # (new_tokens, num_heads, head_dim)
        v = v.squeeze(0)

        end_idx = start_idx + current_len + new_tokens

        self.k_cache[start_idx + current_len:end_idx] = k
        self.v_cache[start_idx + current_len:end_idx] = v

        self.request_lengths[request_id] = current_len + new_tokens

    def read(
        self,
        request_id: int,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read K,V cache for a request.

        Args:
            request_id: Request identifier
            positions: Positions to read (default: all)

        Returns:
            K,V tensors of shape (seq_len, num_heads, head_dim)
        """
        if request_id not in self.allocations:
            raise ValueError(f"Request {request_id} not found")

        start_idx, _ = self.allocations[request_id]
        seq_len = self.request_lengths[request_id]

        if positions is None:
            positions = torch.arange(seq_len, device=self.device)

        k = self.k_cache[start_idx:start_idx + seq_len]
        v = self.v_cache[start_idx:start_idx + seq_len]

        # Select specific positions
        k = k[positions]
        v = v[positions]

        return k, v

    def get_cache_utilization(self) -> float:
        """Get cache utilization as percentage."""
        total_allocated = sum(
            length for _, length in self.allocations.values()
        )
        return 100 * total_allocated / self.max_cache_tokens

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_allocated = sum(
            self.request_lengths.get(req_id, 0)
            for req_id in self.allocations.keys()
        )

        return CacheStats(
            total_allocated_tokens=total_allocated,
            total_free_tokens=self.max_cache_tokens - total_allocated,
            num_active_requests=len(self.allocations),
            cache_utilization_percent=self.get_cache_utilization(),
        )
