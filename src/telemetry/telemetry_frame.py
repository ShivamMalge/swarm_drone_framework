"""
TelemetryFrame — Structured read-only snapshot of complete simulation state.

This dataclass is the sole data contract between the DES kernel observer
layer and all downstream consumers (buffer, bridge, dashboard, export).

It captures:
  - Kinematic state (positions, velocities)
  - Energy state
  - Communication graph (adjacency, components, spectral gap)
  - Coordination state (consensus variance)
  - Communication statistics (drop rate, latency)
  - Regime classification per agent
  - Adaptive control parameters
  - Drone failure flags

All arrays use NumPy for vectorized downstream processing and
serialization readiness. Shapes are documented and strictly enforced.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(slots=True)
class TelemetryFrame:
    """
    Immutable-by-convention snapshot emitted by TelemetryEmitter.

    Attributes
    ----------
    time : float
        Simulation clock value at extraction.
    positions : np.ndarray
        Agent positions, shape (N, 2), dtype float64.
    energies : np.ndarray
        Agent energies, shape (N,), dtype float64.
    adjacency : np.ndarray
        Symmetric binary adjacency matrix, shape (N, N), dtype uint8.
        Built from the RGG communication graph at extraction time.
    connected_components : list[list[int]]
        Each inner list contains agent IDs belonging to one connected
        component. Computed from ``adjacency``.
    spectral_gap : float
        Algebraic connectivity λ₂ of the graph Laplacian.
    consensus_variance : float
        Variance of agent consensus states (global analytic metric;
        agents are blind to this value).
    packet_drop_rate : float
        Cumulative ratio of dropped packets to total sent.
    latency : float
        Configured mean exponential latency parameter (μ_τ).
    regime_state : dict[int, str]
        Mapping of agent_id → current Regime enum name.
    adaptive_parameters : dict[str, Any]
        System-wide adaptive control snapshot (per-agent Θ vectors,
        projection event counts, strategy distribution).
    drone_failure_flags : np.ndarray
        Boolean failure mask, shape (N,), dtype bool.
        True where energy ≤ 0 (permanently dead).
    """

    time: float
    positions: np.ndarray           # (N, 2)
    energies: np.ndarray            # (N,)
    adjacency: np.ndarray           # (N, N) uint8
    connected_components: list[list[int]]
    spectral_gap: float
    consensus_variance: float
    packet_drop_rate: float
    latency: float
    regime_state: dict[int, str]
    adaptive_parameters: dict[str, Any]
    drone_failure_flags: np.ndarray  # (N,) bool
    agent_states: np.ndarray         # (N,) float64

    # ── Factory helpers ──────────────────────────────────────

    @classmethod
    def empty(cls, n: int) -> TelemetryFrame:
        """Return a zero-initialized frame for *n* agents."""
        return cls(
            time=0.0,
            positions=np.zeros((n, 2), dtype=np.float64),
            energies=np.zeros(n, dtype=np.float64),
            adjacency=np.zeros((n, n), dtype=np.uint8),
            connected_components=[list(range(n))],
            spectral_gap=0.0,
            consensus_variance=0.0,
            packet_drop_rate=0.0,
            latency=0.0,
            regime_state={i: "STABLE" for i in range(n)},
            adaptive_parameters={},
            drone_failure_flags=np.ones(n, dtype=bool),
            agent_states=np.zeros(n, dtype=np.float64),
        )
