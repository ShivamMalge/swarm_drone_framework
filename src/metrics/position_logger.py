"""
Position logger — records (time, agent_id, x, y) tuples.

Read-only logger. Agents do NOT access this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PositionSnapshot:
    """Single position measurement."""
    time: float
    agent_id: int
    x: float
    y: float


class PositionLogger:
    """Collects position traces across all agents over time."""

    def __init__(self) -> None:
        self._records: list[PositionSnapshot] = []

    def record(self, time: float, agent_id: int, position: np.ndarray) -> None:
        """Append a position snapshot."""
        self._records.append(
            PositionSnapshot(time, agent_id, float(position[0]), float(position[1]))
        )

    @property
    def records(self) -> list[PositionSnapshot]:
        """All recorded snapshots."""
        return self._records

    def clear(self) -> None:
        """Wipe all records."""
        self._records.clear()
