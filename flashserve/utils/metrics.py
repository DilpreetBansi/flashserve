"""
Performance metrics tracking.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class MetricsTracker:
    """Track generation metrics: TTFT, TPOT, throughput."""

    # TTFT: Time-To-First-Token (ms)
    ttft_list: List[float] = field(default_factory=list)

    # TPOT: Time-Per-Output-Token (ms)
    tpot_list: List[float] = field(default_factory=list)

    # Token counts
    total_tokens_generated: int = 0
    total_tokens_input: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)
    first_token_time: float = None

    def record_first_token(self) -> None:
        """Record time of first token generation."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            ttft = (self.first_token_time - self.start_time) * 1000
            self.ttft_list.append(ttft)

    def record_token(self) -> None:
        """Record time of each subsequent token."""
        self.total_tokens_generated += 1

    def finalize(self) -> None:
        """Finalize metrics for a generation."""
        if self.total_tokens_generated > 0 and self.first_token_time is not None:
            end_time = time.time()
            total_time = end_time - self.first_token_time
            remaining_tokens = self.total_tokens_generated - 1
            if remaining_tokens > 0:
                tpot = (total_time / remaining_tokens) * 1000
                self.tpot_list.append(tpot)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        import statistics

        stats = {
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_input": self.total_tokens_input,
        }

        if self.ttft_list:
            stats["ttft_mean_ms"] = statistics.mean(self.ttft_list)
            stats["ttft_median_ms"] = statistics.median(self.ttft_list)
            stats["ttft_p99_ms"] = sorted(self.ttft_list)[int(0.99 * len(self.ttft_list))]

        if self.tpot_list:
            stats["tpot_mean_ms"] = statistics.mean(self.tpot_list)
            stats["tpot_median_ms"] = statistics.median(self.tpot_list)

        elapsed = time.time() - self.start_time
        if elapsed > 0:
            stats["throughput_tokens_per_sec"] = self.total_tokens_generated / elapsed

        return stats
