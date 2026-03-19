"""
Batch manager for collecting requests into micro-batches with timeouts.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class BatchedRequest:
    """Request in a batch."""
    request_id: int
    data: str
    added_at: float = field(default_factory=time.time)


class BatchManager:
    """
    Collects incoming requests into batches with size and timeout constraints.

    When max_batch_size is reached or max_wait_time is exceeded, a batch is ready.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time_ms: int = 50,
    ):
        """
        Initialize batch manager.

        Args:
            max_batch_size: Maximum requests per batch
            max_wait_time_ms: Maximum wait time in milliseconds
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms / 1000.0

        self.pending_requests: List[BatchedRequest] = []
        self.batch_start_time = None
        self.request_counter = 0

    def add_request(self, data: str) -> int:
        """
        Add request to batch.

        Args:
            data: Request data

        Returns:
            Request ID
        """
        request_id = self.request_counter
        self.request_counter += 1

        request = BatchedRequest(request_id=request_id, data=data)
        self.pending_requests.append(request)

        if self.batch_start_time is None:
            self.batch_start_time = time.time()

        return request_id

    def get_next_batch(self) -> Optional[List[BatchedRequest]]:
        """
        Get next batch if ready.

        Returns:
            List of requests, or None if batch not ready
        """
        if not self.pending_requests:
            return None

        # Check size threshold
        if len(self.pending_requests) >= self.max_batch_size:
            batch = self.pending_requests[:self.max_batch_size]
            self.pending_requests = self.pending_requests[self.max_batch_size:]
            self.batch_start_time = None
            return batch

        # Check time threshold
        if self.batch_start_time is not None:
            elapsed = time.time() - self.batch_start_time
            if elapsed >= self.max_wait_time_ms:
                batch = self.pending_requests
                self.pending_requests = []
                self.batch_start_time = None
                return batch

        return None

    def batch_ready(self) -> bool:
        """Check if a batch is ready."""
        return self.get_next_batch() is not None

    def queue_size(self) -> int:
        """Get current queue size."""
        return len(self.pending_requests)
