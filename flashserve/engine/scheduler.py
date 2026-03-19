"""
Request scheduler with support for FCFS and priority-based scheduling.
"""

from typing import List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import time


class SchedulingPolicy(Enum):
    """Request scheduling policies."""
    FCFS = "fcfs"  # First-Come, First-Served
    PRIORITY = "priority"  # Custom priority function


@dataclass
class ScheduledRequest:
    """Request with scheduling metadata."""
    request_id: int
    priority: float
    created_at: float
    preemptible: bool = True


class RequestScheduler:
    """
    Scheduler for managing request execution order.

    Supports:
    - FCFS scheduling
    - Priority-based scheduling
    - Request preemption
    """

    def __init__(
        self,
        policy: SchedulingPolicy = SchedulingPolicy.FCFS,
        priority_fn: Optional[Callable] = None,
    ):
        """
        Initialize scheduler.

        Args:
            policy: Scheduling policy (FCFS or PRIORITY)
            priority_fn: Function to compute priority (for PRIORITY policy)
        """
        self.policy = policy
        self.priority_fn = priority_fn or (lambda x: 0)
        self.requests: List[ScheduledRequest] = []

    def add_request(self, request_id: int, priority: float = 0.0) -> None:
        """
        Add request to scheduler.

        Args:
            request_id: Unique request identifier
            priority: Priority score (higher = more important)
        """
        request = ScheduledRequest(
            request_id=request_id,
            priority=priority,
            created_at=time.time(),
        )
        self.requests.append(request)

    def get_next_request(self) -> Optional[int]:
        """
        Get next request to execute.

        Returns:
            Request ID, or None if queue is empty
        """
        if not self.requests:
            return None

        if self.policy == SchedulingPolicy.FCFS:
            # Return oldest request
            next_req = self.requests.pop(0)
            return next_req.request_id

        elif self.policy == SchedulingPolicy.PRIORITY:
            # Return highest priority request
            next_req = max(self.requests, key=lambda r: r.priority)
            self.requests.remove(next_req)
            return next_req.request_id

        return None

    def preempt_request(self, request_id: int) -> bool:
        """
        Preempt a running request.

        Args:
            request_id: Request to preempt

        Returns:
            True if preempted, False if not preemptible
        """
        for req in self.requests:
            if req.request_id == request_id:
                if req.preemptible:
                    self.requests.remove(req)
                    return True
                return False
        return False

    def queue_length(self) -> int:
        """Get current queue length."""
        return len(self.requests)

    def promote_request(self, request_id: int, priority_delta: float) -> None:
        """Increase priority of a request."""
        for req in self.requests:
            if req.request_id == request_id:
                req.priority += priority_delta
                break


class FCFSScheduler(RequestScheduler):
    """First-Come, First-Served scheduler."""

    def __init__(self):
        super().__init__(policy=SchedulingPolicy.FCFS)


class PriorityScheduler(RequestScheduler):
    """Priority-based scheduler."""

    def __init__(self, priority_fn: Callable):
        super().__init__(policy=SchedulingPolicy.PRIORITY, priority_fn=priority_fn)
