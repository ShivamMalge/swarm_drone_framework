"""
Drop rate tracker — counts sent / dropped / delivered per comm round.

Read-only logger. Agents do NOT access this.
"""

from __future__ import annotations


class DropRateTracker:
    """Tracks communication drop statistics."""

    def __init__(self) -> None:
        self.total_sent: int = 0
        self.total_dropped: int = 0
        self.total_delivered: int = 0

    def record_sent(self, count: int = 1) -> None:
        """Record *count* packets sent."""
        self.total_sent += count

    def record_dropped(self, count: int = 1) -> None:
        """Record *count* packets dropped."""
        self.total_dropped += count

    def record_delivered(self, count: int = 1) -> None:
        """Record *count* packets delivered."""
        self.total_delivered += count

    @property
    def drop_rate(self) -> float:
        """Empirical drop rate."""
        if self.total_sent == 0:
            return 0.0
        return self.total_dropped / self.total_sent

    @property
    def delivery_rate(self) -> float:
        """Empirical delivery rate."""
        if self.total_sent == 0:
            return 0.0
        return self.total_delivered / self.total_sent

    def reset(self) -> None:
        """Reset all counters."""
        self.total_sent = 0
        self.total_dropped = 0
        self.total_delivered = 0
