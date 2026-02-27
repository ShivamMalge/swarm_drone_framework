"""
Simulation event types, base Event class, and priority ordering.

Events are the sole mechanism for state mutation in the simulation.
Ordering: (timestamp, priority, global_sequence_id) — fully deterministic.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class EventType(enum.IntEnum):
    """
    Event categories ordered by dispatch priority (lower = higher priority).
    When two events share the same timestamp, priority breaks the tie.
    """

    ENV_UPDATE = 0
    KINEMATIC_UPDATE = 1
    ENERGY_UPDATE = 2
    MSG_TRANSMIT = 3
    MSG_DELIVER = 4
    AUCTION_TIMEOUT = 5
    METRICS_LOG = 6
    CONSENSUS_UPDATE = 7


# Global auto-incrementing counter for deterministic tie-breaking.
_global_sequence_counter: int = 0


def _next_sequence_id() -> int:
    """Return the next globally unique sequence ID."""
    global _global_sequence_counter
    sid = _global_sequence_counter
    _global_sequence_counter += 1
    return sid


def reset_sequence_counter() -> None:
    """Reset the global sequence counter (for testing / new episodes)."""
    global _global_sequence_counter
    _global_sequence_counter = 0


@dataclass(order=False)
class Event:
    """
    A single discrete simulation event.

    Sorting key: (timestamp, priority, sequence_id).
    This guarantees fully deterministic execution order with no reliance
    on Python's default object comparison.

    Attributes
    ----------
    timestamp : float
        Simulation time at which the event fires.
    event_type : EventType
        Category determining dispatch handler and priority.
    agent_id : int | None
        Target agent (None for global events like ENV_UPDATE / METRICS_LOG).
    payload : Any
        Arbitrary data carried by the event.
    sequence_id : int
        Auto-assigned global counter for deterministic tie-breaking.
    """

    timestamp: float
    event_type: EventType
    agent_id: int | None = None
    payload: Any = field(default=None, repr=False)
    sequence_id: int = field(default_factory=_next_sequence_id, init=False)

    # -- Comparison operators for heapq (min-heap) ordering --

    def _sort_key(self) -> tuple[float, int, int]:
        return (self.timestamp, int(self.event_type), self.sequence_id)

    def __lt__(self, other: Event) -> bool:
        return self._sort_key() < other._sort_key()

    def __le__(self, other: Event) -> bool:
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other: Event) -> bool:
        return self._sort_key() > other._sort_key()

    def __ge__(self, other: Event) -> bool:
        return self._sort_key() >= other._sort_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return self._sort_key() == other._sort_key()

    def __hash__(self) -> int:
        return hash(self._sort_key())
