"""
percolation_shock_analyzer.py — Phase 2AA.1 Percolation Shock Detector.

Detects the onset of fragmentation collapse and tracks how disconnection
propagates outward from the original LCC boundary as a "shock wave".
All computation is vectorized via adjacency-matrix multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analytics.percolation_analyzer import PercolationMetrics
from src.telemetry.telemetry_frame import TelemetryFrame


@dataclass
class ShockMetrics:
    shock_active: bool
    shock_distance: np.ndarray     # (N,) float64 — raw BFS distance from LCC
    shock_normalized: np.ndarray   # (N,) float64 — [0, 1] normalised distance
    front_size: int                # number of nodes on the current wavefront
    max_distance: float            # current maximum propagation depth
    velocity: float                # Δmax_distance / frame


class PercolationShockAnalyzer:
    """
    Observer-only analyzer.  Seeds a distance field from the LCC at the
    moment of fragmentation and expands it each frame using vectorised
    adjacency-matrix operations.
    """

    TRIGGER_THRESHOLD = 0.6   # LCC ratio below which shock begins

    def __init__(self) -> None:
        self._shock_active: bool = False
        self._shock_distance: np.ndarray | None = None
        self._prev_max: float = 0.0
        self._last_hash: int = -1
        self._last_metrics: ShockMetrics | None = None

    # ── Public API ──────────────────────────────────────────

    def reset(self) -> None:
        self._shock_active = False
        self._shock_distance = None
        self._prev_max = 0.0
        self._last_hash = -1
        self._last_metrics = None

    def analyze(
        self,
        frame: TelemetryFrame,
        perc: PercolationMetrics,
    ) -> ShockMetrics:
        N = len(frame.positions)
        if N == 0:
            return ShockMetrics(False, np.array([]), np.array([]), 0, 0.0, 0.0)

        alive_mask = ~frame.drone_failure_flags

        # Hash-skip (topology + positions — positions change busts cache for propagation)
        h = hash(frame.adjacency.data.tobytes()
                 + alive_mask.tobytes()
                 + frame.positions.tobytes())
        if h == self._last_hash and self._last_metrics is not None:
            return self._last_metrics
        self._last_hash = h

        # ── Trigger detection ───────────────────────────────
        if not self._shock_active:
            if perc.connectivity_ratio < self.TRIGGER_THRESHOLD:
                self._shock_active = True
                # Seed distance: LCC nodes get 0, everyone else gets inf
                self._shock_distance = np.full(N, np.inf, dtype=np.float64)
                if perc.lcc_nodes is not None and len(perc.lcc_nodes) > 0:
                    self._shock_distance[perc.lcc_nodes] = 0.0
                # Dead agents stay inf
                self._shock_distance[~alive_mask] = np.inf
                self._prev_max = 0.0

        if not self._shock_active:
            m = ShockMetrics(False, np.zeros(N), np.zeros(N), 0, 0.0, 0.0)
            self._last_metrics = m
            return m

        # ── Propagation (vectorised BFS relaxation) ─────────
        # One relaxation step:  d[i] = min(d[i], min(d[neighbors]) + 1)
        # Using adjacency matrix: for each alive node, the minimum
        # neighbour distance is  min over j where adj[i,j]==1 of d[j].
        dist = self._shock_distance
        adj = frame.adjacency.astype(np.float64)

        # Mask dead columns/rows so they can't propagate
        adj[~alive_mask, :] = 0
        adj[:, ~alive_mask] = 0

        # Replace zeros in adj with inf so np.min picks only actual neighbours
        # Build a (N, N) matrix of neighbour distances; non-neighbours → inf
        INF = 1e18
        # For each (i,j) where adj[i,j]==1 → dist[j]; else → INF
        neigh_dist = np.where(adj > 0, dist[np.newaxis, :], INF)  # (N, N)
        min_neigh = np.min(neigh_dist, axis=1)  # (N,)

        new_dist = np.minimum(dist, min_neigh + 1.0)

        # Keep dead agents at inf, LCC origin at 0
        new_dist[~alive_mask] = np.inf
        # Nodes that were originally 0 stay 0
        new_dist[dist == 0.0] = 0.0

        self._shock_distance = new_dist

        # ── Metrics ─────────────────────────────────────────
        finite = np.isfinite(new_dist) & alive_mask
        max_d = float(new_dist[finite].max()) if np.any(finite) else 0.0
        velocity = max_d - self._prev_max
        self._prev_max = max_d

        # Wavefront: nodes whose distance == max_d (the outermost ring)
        front_size = int(np.sum(new_dist[finite] == max_d)) if max_d > 0 else 0

        # Normalise to [0, 1]
        norm = np.zeros(N, dtype=np.float64)
        if max_d > 0:
            norm[finite] = new_dist[finite] / max_d
        # Non-finite stays 0 (dead / unreached)

        m = ShockMetrics(
            shock_active=True,
            shock_distance=new_dist,
            shock_normalized=norm,
            front_size=front_size,
            max_distance=max_d,
            velocity=velocity,
        )
        self._last_metrics = m
        return m
