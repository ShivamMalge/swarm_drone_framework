"""
Energy profiler — records (time, agent_id, energy) tuples.

Read-only logger. Agents do NOT access this.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnergySnapshot:
    """Single energy measurement."""
    time: float
    agent_id: int
    energy: float


class EnergyProfiler:
    """Collects energy traces across all agents over time."""

    def __init__(self) -> None:
        self._records: list[EnergySnapshot] = []

    def record(self, time: float, agent_id: int, energy: float) -> None:
        """Append an energy snapshot."""
        self._records.append(EnergySnapshot(time, agent_id, energy))

    @property
    def records(self) -> list[EnergySnapshot]:
        """All recorded snapshots."""
        return self._records

    def get_agent_trace(self, agent_id: int) -> list[EnergySnapshot]:
        """Return energy trace for a specific agent."""
        return [r for r in self._records if r.agent_id == agent_id]

    def clear(self) -> None:
        """Wipe all records."""
        self._records.clear()
