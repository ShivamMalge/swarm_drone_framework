"""
TelemetryEmitter — Read-only extraction bridge from DES kernel to telemetry pipeline.

Strict behavioural contract:
  - NEVER modifies simulation state.
  - NEVER schedules events.
  - NEVER interacts with kernel logic.
  - Reads agent / orchestrator values by direct attribute access (O(N)).
  - Builds adjacency and spectral metrics from the communication layer's
    global position array (global view is PERMITTED for the observer layer
    per architecture rules; agents remain blind).

This module feeds Phase 2B (TelemetryBuffer) and Phase 2C (SimulationWorker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

from src.telemetry.telemetry_frame import TelemetryFrame

if TYPE_CHECKING:
    from src.core.config import SimConfig
    from src.simulation import Phase1Simulation


class TelemetryEmitter:
    """
    Extracts a complete TelemetryFrame from a live Phase1Simulation instance.

    Parameters
    ----------
    simulation_ref : Phase1Simulation
        A *reference* to the running simulation orchestrator. The emitter
        never copies the orchestrator — it reads values in-place to
        minimise allocation overhead.
    config : SimConfig
        Frozen simulation configuration (for comm_radius, latency_mean, etc.).
    """

    def __init__(self, simulation_ref: Phase1Simulation, config: SimConfig) -> None:
        self._sim = simulation_ref
        self._cfg = config
        self._n = config.num_agents

    # ── Public API ───────────────────────────────────────────

    def extract_frame(self) -> TelemetryFrame:
        """
        Pull a full snapshot from the simulation.

        Complexity: O(N) for agent iteration + O(N log N) for KDTree.
        """
        sim = self._sim
        cfg = self._cfg
        agents = sim.agents
        n = self._n

        # ── 1. Positions  (N, 2) ────────────────────────────
        # Use internal _position directly to avoid N copy() calls
        positions = np.array(
            [a._position for a in agents], dtype=np.float64
        )

        # ── 2. Energies   (N,) ──────────────────────────────
        energies = np.array(
            [a._energy.energy for a in agents], dtype=np.float64
        )

        # ── 3. Drone failure flags  (N,) bool ───────────────
        drone_failure_flags = energies <= 0.0

        # ── 4. Adjacency  (N, N) uint8  — from RGG ─────────
        adjacency = np.zeros((n, n), dtype=np.uint8)
        tree = KDTree(positions)
        pairs = tree.query_pairs(cfg.comm_radius)
        for u, v in pairs:
            adjacency[u, v] = 1
            adjacency[v, u] = 1

        # ── 5. Connected components (BFS on adjacency) ──────
        connected_components = self._compute_components(adjacency, n)

        # ── 6. Spectral gap λ₂ ──────────────────────────────
        spectral_gap = self._compute_spectral_gap(adjacency, n)

        # ── 7. Consensus variance (global analytic) ─────────
        consensus_states = np.array(
            [a.consensus_state for a in agents], dtype=np.float64
        )
        alive_mask = sim.alive_mask
        alive_consensus = consensus_states[alive_mask]
        if len(alive_consensus) > 1:
            consensus_variance = float(np.var(alive_consensus))
        else:
            consensus_variance = 0.0

        # ── 8. Packet drop rate ─────────────────────────────
        total_sent = sim.comm_engine.total_sent
        total_dropped = sim.comm_engine.total_dropped
        if total_sent > 0:
            packet_drop_rate = total_dropped / total_sent
        else:
            packet_drop_rate = 0.0

        # ── 9. Latency (configured mean) ────────────────────
        latency = cfg.latency_mean

        # ── 10. Regime state per agent ──────────────────────
        regime_state: dict[int, str] = {}
        for a in agents:
            if a.is_alive:
                regime_state[a.agent_id] = a.current_regime.name
            else:
                regime_state[a.agent_id] = "DEAD"

        # ── 11. Adaptive parameters snapshot ────────────────
        adaptive_parameters = self._extract_adaptive_parameters(agents)

        # ── 12. Assemble frame ──────────────────────────────
        return TelemetryFrame(
            time=sim.kernel.now,
            positions=positions,
            energies=energies,
            adjacency=adjacency,
            connected_components=connected_components,
            spectral_gap=spectral_gap,
            consensus_variance=consensus_variance,
            packet_drop_rate=packet_drop_rate,
            latency=latency,
            regime_state=regime_state,
            adaptive_parameters=adaptive_parameters,
            drone_failure_flags=drone_failure_flags,
            agent_states=consensus_states,
        )

    def emit(self) -> TelemetryFrame:
        """Alias kept for interface symmetry with downstream consumers."""
        return self.extract_frame()

    # ── Private vectorised helpers ───────────────────────────

    @staticmethod
    def _compute_components(adj: np.ndarray, n: int) -> list[list[int]]:
        """BFS-based connected-component extraction from an adjacency matrix."""
        visited = np.zeros(n, dtype=bool)
        components: list[list[int]] = []

        for start in range(n):
            if visited[start]:
                continue
            component: list[int] = []
            stack = [start]
            visited[start] = True
            while stack:
                node = stack.pop()
                component.append(node)
                neighbors = np.nonzero(adj[node])[0]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(component)

        return components

    @staticmethod
    def _compute_spectral_gap(adj: np.ndarray, n: int) -> float:
        """Compute algebraic connectivity λ₂ from the adjacency matrix."""
        if n <= 1:
            return 0.0

        degree = adj.sum(axis=1).astype(np.float64)
        laplacian = np.diag(degree) - adj.astype(np.float64)

        try:
            eigvals = np.linalg.eigvalsh(laplacian)
            # eigvalsh returns sorted ascending; λ₂ is index 1
            return float(eigvals[1])
        except np.linalg.LinAlgError:
            return 0.0

    @staticmethod
    def _extract_adaptive_parameters(agents) -> dict:
        """
        Build a system-wide adaptive parameter snapshot.
        Iterates once over agents (O(N)).
        """
        n = len(agents)
        coverage_gains = np.empty(n, dtype=np.float64)
        gossip_epsilons = np.empty(n, dtype=np.float64)
        broadcast_rates = np.empty(n, dtype=np.float64)
        auction_participations = np.empty(n, dtype=np.float64)
        velocity_scales = np.empty(n, dtype=np.float64)
        strategies = [""] * n
        projection_events_total = 0
        tuning_updates_total = 0

        for i, a in enumerate(agents):
            coverage_gains[i] = a.coverage_gain
            gossip_epsilons[i] = a.gossip_epsilon
            broadcast_rates[i] = a.broadcast_rate
            auction_participations[i] = a.auction_participation
            velocity_scales[i] = a.velocity_scale
            strategies[i] = a.current_strategy.name
            projection_events_total += a.projection_events
            tuning_updates_total += a.total_tuning_updates

        return {
            "coverage_gains": coverage_gains,
            "gossip_epsilons": gossip_epsilons,
            "broadcast_rates": broadcast_rates,
            "auction_participations": auction_participations,
            "velocity_scales": velocity_scales,
            "strategies": strategies,
            "projection_events_total": projection_events_total,
            "tuning_updates_total": tuning_updates_total,
        }
