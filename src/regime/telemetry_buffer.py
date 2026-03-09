"""
Rolling window telemetry buffer for Phase 2D.

Stores historical time-series proxy metrics entirely separated from
the core execution loops to enforce passive observation properties.
"""

from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass, field


@dataclass
class TelemetrySnapshot:
    """A single frame of local proxy metrics."""
    time: float
    neighbor_count: int
    mean_neighbor_age: float
    local_consensus_variance: float
    local_energy: float


class TelemetryBuffer:
    """
    Rolling fixed-capacity window of telemetry snapshots.
    Used exclusively for passive regime detection heuristics.
    """

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("Telemetry window size must be > 0")
        
        self.window_size = window_size
        self._history: collections.deque[TelemetrySnapshot] = collections.deque(maxlen=window_size)

    def append(self, snapshot: TelemetrySnapshot) -> None:
        """Push a new snapshot into the rolling window."""
        self._history.append(snapshot)

    def is_ready(self) -> bool:
        """Return True only if the buffer has accumulated enough history over time."""
        # Need at least two points to derive slopes/trends, ideally more
        return len(self._history) >= min(3, self.window_size)

    def get_recent_history(self) -> list[TelemetrySnapshot]:
        """Return a chronological list of recent snapshots."""
        return list(self._history)

    def get_smoothed_metrics(self) -> dict[str, float]:
        """
        Compute moving averages for scalar heuristic thresholds.

        Returns
        -------
        dict[str, float]
            Aggregated metric mapping.
        """
        if not self._history:
            return {
                "mean_neighbor_count": 0.0,
                "mean_staleness": 0.0,
                "mean_variance": 0.0,
                "energy_slope": 0.0,
            }

        counts = [s.neighbor_count for s in self._history]
        ages = [s.mean_neighbor_age for s in self._history]
        variances = [s.local_consensus_variance for s in self._history]
        energies = [s.local_energy for s in self._history]
        times = [s.time for s in self._history]

        metrics = {
            "mean_neighbor_count": statistics.mean(counts),
            "mean_staleness": statistics.mean(ages),
            "mean_variance": statistics.mean(variances),
        }

        # Compute energy consumption rate (slope via simple finite difference over window)
        if len(self._history) > 1:
            dt = times[-1] - times[0]
            de = energies[-1] - energies[0]
            metrics["energy_slope"] = de / dt if dt > 0 else 0.0
        else:
            metrics["energy_slope"] = 0.0

        return metrics
