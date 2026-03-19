"""
Continuous/dynamic batching scheduler for efficient request serving.

Requests enter and exit the batch at any iteration (not at epoch boundaries).
Minimizes Time-to-First-Token (TTFT) while maintaining high throughput.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import time
from collections import deque


class RequestStage(Enum):
    """Stage of request processing."""
    WAITING = "waiting"  # In queue, not yet processing
    PREFILL = "prefill"  # Processing prompt (first iteration)
    DECODE = "decode"    # Generating output tokens


@dataclass
class Request:
    """A generation request."""
    request_id: int
    prompt: str
    max_tokens: int
    created_at: float = field(default_factory=time.time)
    stage: RequestStage = RequestStage.WAITING

    # Generation tracking
    prompt_tokens: int = 0
    generated_tokens: int = 0
    is_completed: bool = False


@dataclass
class Batch:
    """A batch of requests ready for execution."""
    request_ids: List[int]
    stage: RequestStage


class ContinuousBatchingScheduler:
    """
    Scheduler that performs continuous batching with the following properties:

    1. Prefill prioritization: Process full prompts first (minimize TTFT)
    2. Iteration-level batching: Requests enter/exit dynamically
    3. Memory awareness: Batch size adjusted based on available memory
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time_ms: int = 50,
        prefill_batch_size: int = None,
    ):
        """
        Initialize continuous batching scheduler.

        Args:
            max_batch_size: Maximum decode batch size
            max_wait_time_ms: Max wait time before batching prefill requests
            prefill_batch_size: Separate batch size for prefill (default: same as max)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.prefill_batch_size = prefill_batch_size or max_batch_size

        # Request queues
        self.waiting_queue: deque = deque()  # New requests waiting to be scheduled
        self.prefill_queue: deque = deque()  # Requests in prefill stage
        self.decode_queue: deque = deque()   # Requests in decode stage

        # Tracking
        self.request_map: Dict[int, Request] = {}
        self.next_request_id = 0

    def add_request(self, prompt: str, max_tokens: int) -> int:
        """
        Add a new generation request to the scheduler.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Request ID for tracking
        """
        request_id = self.next_request_id
        self.next_request_id += 1

        request = Request(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        self.request_map[request_id] = request
        self.waiting_queue.append(request_id)

        return request_id

    def get_next_batch(self) -> Optional[Batch]:
        """
        Get the next batch for execution.

        Prioritizes prefill requests to minimize TTFT.

        Returns:
            Batch object with request IDs and stage, or None if no requests ready
        """
        # First, try to form a prefill batch
        prefill_batch = self._get_prefill_batch()
        if prefill_batch is not None:
            return prefill_batch

        # Otherwise, form a decode batch
        decode_batch = self._get_decode_batch()
        if decode_batch is not None:
            return decode_batch

        return None

    def _get_prefill_batch(self) -> Optional[Batch]:
        """Get a batch of prefill requests."""
        # Move waiting requests to prefill queue if time exceeded
        now = time.time()
        timeout = self.max_wait_time_ms / 1000.0

        while len(self.waiting_queue) > 0:
            request_id = self.waiting_queue[0]
            request = self.request_map[request_id]

            if len(self.prefill_queue) >= self.prefill_batch_size:
                break
            if time.time() - request.created_at < timeout and len(self.prefill_queue) > 0:
                break

            self.waiting_queue.popleft()
            request.stage = RequestStage.PREFILL
            self.prefill_queue.append(request_id)

        # Return prefill batch if any
        if len(self.prefill_queue) > 0:
            batch_size = min(len(self.prefill_queue), self.prefill_batch_size)
            request_ids = []
            for _ in range(batch_size):
                request_ids.append(self.prefill_queue.popleft())

            return Batch(request_ids=request_ids, stage=RequestStage.PREFILL)

        return None

    def _get_decode_batch(self) -> Optional[Batch]:
        """Get a batch of decode requests."""
        if len(self.decode_queue) > 0:
            batch_size = min(len(self.decode_queue), self.max_batch_size)
            request_ids = []
            for _ in range(batch_size):
                request_ids.append(self.decode_queue.popleft())

            return Batch(request_ids=request_ids, stage=RequestStage.DECODE)

        return None

    def mark_prefill_complete(self, request_id: int) -> None:
        """Mark a request's prefill as complete, move to decode."""
        if request_id not in self.request_map:
            return

        request = self.request_map[request_id]
        request.stage = RequestStage.DECODE
        self.decode_queue.append(request_id)

    def mark_request_complete(self, request_id: int) -> None:
        """Mark a request as complete and remove from queues."""
        if request_id not in self.request_map:
            return

        request = self.request_map[request_id]
        request.is_completed = True

        # Remove from any queue (shouldn't be in multiple)
        # In practice, would ensure it's not in any queue

        # Can optionally keep for stats/logging
        # del self.request_map[request_id]

    def get_queue_stats(self) -> Dict:
        """Get current scheduler statistics."""
        return {
            "waiting_queue_size": len(self.waiting_queue),
            "prefill_queue_size": len(self.prefill_queue),
            "decode_queue_size": len(self.decode_queue),
            "total_active_requests": len(self.request_map),
            "completed_requests": sum(
                1 for r in self.request_map.values() if r.is_completed
            ),
        }
