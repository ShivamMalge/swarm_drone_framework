"""
percolation_analyzer.py — Phase 2Q Connectivity Percolation Analyzer.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass

import numpy as np

from PySide6.QtCore import QObject, Signal

from src.telemetry.telemetry_frame import TelemetryFrame


@dataclass
class PercolationMetrics:
    lcc_size: int
    total_agents: int
    num_components: int
    connectivity_ratio: float
    normalized_ratio: float
    connectivity_margin: float
    d_ratio_dt: float
    state: str
    lcc_nodes: np.ndarray | None = None  # indices of nodes in the LCC


class PercolationAnalyzer(QObject):
    """
    Detects network fragmentation and percolation collapse.
    Emits signals when the connectivity ratio drops below threshold
    and Recovers.
    """

    percolation_collapse_detected = Signal(float)  # timestamp
    percolation_recovered = Signal(float)  # timestamp

    def __init__(
        self,
        collapse_threshold: float = 0.5,
        recovery_threshold: float = 0.6,
        history_size: int = 300,
    ) -> None:
        super().__init__()
        self._collapse_threshold = collapse_threshold
        self._recovery_threshold = recovery_threshold
        self._history_size = history_size

        self._in_collapse_state: bool = False
        self._last_adj_hash: int = -1
        self._last_metrics: PercolationMetrics | None = None

        # Temporal Tracking
        self.time_history = collections.deque(maxlen=history_size)
        self.ratio_history = collections.deque(maxlen=history_size)
        self.components_history = collections.deque(maxlen=history_size)

    def analyze_frame(self, frame: TelemetryFrame) -> PercolationMetrics:
        """Process a frame, maintain history, and emit collapse events if bounds crossed."""
        # 1. Efficient Computation: Avoid full recompute if adjacency unchanged
        adj_hash = hash(frame.adjacency.data.tobytes())
        alive_mask = ~frame.drone_failure_flags
        total_agents = int(len(alive_mask))  # Base ratio on absolute total swarm size

        if adj_hash != self._last_adj_hash or self._last_metrics is None:
            self._last_adj_hash = adj_hash

            # BFS / Union-Find isn't strictly needed as TelemetryFrame
            # already provides connected components, but we compute it locally per the prompt,
            # using adjacency directly to ensure correct logic if components array is outdated.
            
            # Simple BFS to find components
            visited = np.zeros(total_agents, dtype=bool)
            components = []
            
            # Use raw adjacency
            adj = frame.adjacency
            
            for i in range(total_agents):
                if not visited[i] and alive_mask[i]:
                    comp_members = []
                    queue = collections.deque([i])
                    visited[i] = True
                    
                    while queue:
                        curr = queue.popleft()
                        comp_members.append(curr)
                        
                        # Find unvisited neighbors
                        neighbors = np.where(adj[curr])[0]
                        unvisited = neighbors[~visited[neighbors]]
                        
                        # Filter to only alive unvisited out of caution
                        unvisited = unvisited[alive_mask[unvisited]]
                        
                        visited[unvisited] = True
                        queue.extend(unvisited)
                        
                    components.append(comp_members)

            comp_sizes = [len(c) for c in components]
            lcc_idx = int(np.argmax(comp_sizes)) if comp_sizes else 0
            lcc_size = comp_sizes[lcc_idx] if comp_sizes else 0
            lcc_nodes = np.array(components[lcc_idx], dtype=np.intp) if components else np.array([], dtype=np.intp)
            num_components = len(components)
            alive_count = int(alive_mask.sum())
            ratio = lcc_size / max(1, alive_count)

            self._last_metrics = PercolationMetrics(
                lcc_size=lcc_size,
                total_agents=alive_count,
                num_components=num_components,
                connectivity_ratio=ratio,
                normalized_ratio=ratio,
                connectivity_margin=ratio - self._collapse_threshold,
                d_ratio_dt=0.0,
                state="STABLE",
                lcc_nodes=lcc_nodes,
            )

        metrics = self._last_metrics

        # Rate of Change calculation
        dt = frame.time - self._last_time if hasattr(self, '_last_time') and self._last_time >= 0 else 1.0
        if dt <= 0:
            dt = 1e-6
        self._last_time = frame.time
        
        if len(self.ratio_history) > 0:
            metrics.d_ratio_dt = (metrics.connectivity_ratio - self.ratio_history[-1]) / dt
        else:
            metrics.d_ratio_dt = 0.0

        # Temporal Tracking
        self.time_history.append(frame.time)
        self.ratio_history.append(metrics.connectivity_ratio)
        self.components_history.append(metrics.num_components)

        # Collapse Detection (Hysteresis Band) using RAW metric
        ratio = metrics.connectivity_ratio

        if not self._in_collapse_state and ratio < self._collapse_threshold:
            self._in_collapse_state = True
            # In live, frame.time is our logical time
            self.percolation_collapse_detected.emit(frame.time)
        elif self._in_collapse_state and ratio >= self._recovery_threshold:
            self._in_collapse_state = False
            self.percolation_recovered.emit(frame.time)

        if self._in_collapse_state:
            metrics.state = "COLLAPSE"
        elif ratio < self._recovery_threshold:
            metrics.state = "DEGRADING"
        else:
            metrics.state = "STABLE"

        return metrics
