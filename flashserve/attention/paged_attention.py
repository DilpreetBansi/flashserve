"""
Paged Attention: Efficient KV cache management with dynamic page allocation.

Reference: "Efficient Memory Management for Large Language Model Serving" (Kwon et al., 2023)

Key idea: Divide KV cache into fixed-size pages instead of per-request allocation.
- Reduces fragmentation (pages can be reused across requests)
- Enables copy-on-write for parallel sampling
- Dynamic memory allocation matches actual sequence length
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PageTableEntry:
    """Entry in a request's page table mapping logical to physical blocks."""
    logical_block_id: int
    physical_block_id: int
    is_copy_on_write: bool = False


@dataclass
class PageStats:
    """Statistics about page allocation and usage."""
    total_pages: int
    allocated_pages: int
    free_pages: int
    num_requests: int
    total_tokens_cached: int
    fragmentation_ratio: float = 0.0


class PagedKVCache:
    """
    Efficient paged KV cache manager for serving multiple requests.

    Allocates fixed-size pages (e.g., 16 tokens) that can be shared across requests.
    Supports dynamic memory allocation, copy-on-write for sampling, and page reuse.
    """

    def __init__(
        self,
        page_size: int = 16,
        max_pages: int = 10000,
        hidden_size: int = 4096,
        num_heads: int = 32,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize paged KV cache.

        Args:
            page_size: Number of tokens per page (typically 16)
            max_pages: Maximum number of pages to allocate
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            dtype: Data type for KV tensors
            device: Device to allocate memory on
        """
        self.page_size = page_size
        self.max_pages = max_pages
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dtype = dtype
        self.device = device

        # Physical KV cache storage: (max_pages, page_size, num_heads, head_dim)
        self.k_cache = torch.zeros(
            (max_pages, page_size, num_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (max_pages, page_size, num_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )

        # Track page ownership: page_id -> request_id (or -1 if free)
        self.page_to_request: Dict[int, int] = {}
        self.free_pages: List[int] = list(range(max_pages))

        # Track request page tables: request_id -> list of (logical_block, physical_page)
        self.request_page_tables: Dict[int, List[Tuple[int, int]]] = {}
        self.request_seq_lengths: Dict[int, int] = {}

    def allocate(self, request_id: int, num_pages: int) -> List[int]:
        """
        Allocate pages for a new request.

        Args:
            request_id: Unique request identifier
            num_pages: Number of pages to allocate

        Returns:
            List of allocated physical page IDs
        """
        if len(self.free_pages) < num_pages:
            raise RuntimeError(
                f"Insufficient free pages: need {num_pages}, have {len(self.free_pages)}"
            )

        allocated_pages = []
        for _ in range(num_pages):
            phys_page_id = self.free_pages.pop()
            allocated_pages.append(phys_page_id)
            self.page_to_request[phys_page_id] = request_id

        # Initialize page table for this request
        self.request_page_tables[request_id] = [
            (logical_id, phys_id) for logical_id, phys_id in enumerate(allocated_pages)
        ]
        self.request_seq_lengths[request_id] = 0

        return allocated_pages

    def free(self, request_id: int) -> int:
        """
        Free all pages allocated to a request.

        Args:
            request_id: Request to free

        Returns:
            Number of pages freed
        """
        if request_id not in self.request_page_tables:
            return 0

        pages_to_free = self.request_page_tables[request_id]
        num_freed = 0

        for _, phys_page_id in pages_to_free:
            if phys_page_id in self.page_to_request:
                del self.page_to_request[phys_page_id]
                self.free_pages.append(phys_page_id)
                num_freed += 1

        del self.request_page_tables[request_id]
        del self.request_seq_lengths[request_id]

        return num_freed

    def append_kv(
        self,
        request_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Append K,V tokens to cache for a request.

        Args:
            request_id: Request identifier
            k: Key tensor of shape (batch, new_tokens, num_heads, head_dim)
            v: Value tensor of shape (batch, new_tokens, num_heads, head_dim)
        """
        if request_id not in self.request_page_tables:
            raise ValueError(f"Request {request_id} not allocated")

        seq_len = self.request_seq_lengths[request_id]
        new_tokens = k.shape[1]
        page_table = self.request_page_tables[request_id]

        k = k.squeeze(0)  # (new_tokens, num_heads, head_dim)
        v = v.squeeze(0)

        # Write tokens to appropriate pages
        for token_idx in range(new_tokens):
            global_pos = seq_len + token_idx
            page_idx = global_pos // self.page_size
            pos_in_page = global_pos % self.page_size

            if page_idx >= len(page_table):
                raise RuntimeError(f"Exceeded allocated pages for request {request_id}")

            _, phys_page_id = page_table[page_idx]

            self.k_cache[phys_page_id, pos_in_page] = k[token_idx]
            self.v_cache[phys_page_id, pos_in_page] = v[token_idx]

        self.request_seq_lengths[request_id] = seq_len + new_tokens

    def read_kv_paged(
        self,
        request_id: int,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read K,V cache for a request using page table lookup.

        Args:
            request_id: Request identifier
            positions: Specific positions to read (default: all)

        Returns:
            K,V tensors of shape (num_heads, seq_len, head_dim)
        """
        if request_id not in self.request_page_tables:
            raise ValueError(f"Request {request_id} not found")

        seq_len = self.request_seq_lengths[request_id]
        page_table = self.request_page_tables[request_id]

        if positions is None:
            positions = torch.arange(seq_len, device=self.device)

        # Read from physical pages
        k_out = torch.zeros(
            (len(positions), self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        v_out = torch.zeros(
            (len(positions), self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        for out_idx, pos in enumerate(positions):
            page_idx = pos // self.page_size
            pos_in_page = pos % self.page_size

            if page_idx >= len(page_table):
                continue

            _, phys_page_id = page_table[page_idx]
            k_out[out_idx] = self.k_cache[phys_page_id, pos_in_page]
            v_out[out_idx] = self.v_cache[phys_page_id, pos_in_page]

        return k_out, v_out

    def copy_on_write(
        self,
        source_request_id: int,
        new_request_id: int,
    ) -> None:
        """
        Create a copy-on-write copy of a request's KV cache (for parallel sampling).

        Instead of immediately copying data, we create logical page references
        that become independent when modified.

        Args:
            source_request_id: Request to copy from
            new_request_id: New request identifier
        """
        if source_request_id not in self.request_page_tables:
            raise ValueError(f"Source request {source_request_id} not found")

        source_page_table = self.request_page_tables[source_request_id]
        source_seq_len = self.request_seq_lengths[source_request_id]

        # Create reference to same physical pages
        self.request_page_tables[new_request_id] = list(source_page_table)
        self.request_seq_lengths[new_request_id] = source_seq_len

        # Mark pages as CoW (in practice, would implement proper CoW logic)
        for _, phys_page_id in source_page_table:
            if phys_page_id not in self.page_to_request:
                self.page_to_request[phys_page_id] = source_request_id

    def get_stats(self) -> PageStats:
        """
        Get current cache statistics.

        Returns:
            PageStats object with allocation and utilization info
        """
        allocated_pages = len(self.page_to_request)
        total_tokens_cached = sum(self.request_seq_lengths.values())
        fragmentation_ratio = len(self.free_pages) / self.max_pages

        return PageStats(
            total_pages=self.max_pages,
            allocated_pages=allocated_pages,
            free_pages=len(self.free_pages),
            num_requests=len(self.request_page_tables),
            total_tokens_cached=total_tokens_cached,
            fragmentation_ratio=fragmentation_ratio,
        )

    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage in MB.

        Returns:
            Dictionary with breakdown of memory usage
        """
        bytes_per_element = 4 if self.dtype == torch.float32 else 2
        page_size_bytes = (
            self.page_size * self.num_heads * self.head_dim * bytes_per_element * 2  # K and V
        )

        stats = self.get_stats()
        allocated_memory = stats.allocated_pages * page_size_bytes / (1024 * 1024)
        total_memory = self.max_pages * page_size_bytes / (1024 * 1024)

        return {
            "allocated_mb": allocated_memory,
            "total_mb": total_memory,
            "utilization_percent": 100 * stats.allocated_pages / self.max_pages,
        }
