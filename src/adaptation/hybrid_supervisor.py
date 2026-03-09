"""
Hybrid Supervisor - Adaptive Control Layer (Phase 3C).

Maps passive regime classification into actionable behavioral strategies
that toggle agent operations to adapt to environmental constraints.
Operates purely within the AgentCore utilizing only LocalMap logic.
"""

from __future__ import annotations

from enum import Enum

from src.regime.classifier import Regime


class Strategy(Enum):
    """
    Adaptive behavioral policies dictating local agent routines.
    """
    NORMAL_OPERATION = "NORMAL_OPERATION"
    CONSENSUS_PRIORITY = "CONSENSUS_PRIORITY"
    CONNECTIVITY_RECOVERY = "CONNECTIVITY_RECOVERY"
    ENERGY_CONSERVATION = "ENERGY_CONSERVATION"
    LATENCY_STABILIZATION = "LATENCY_STABILIZATION"


class HybridSupervisor:
    """
    Decentralized control mechanism selecting strategies dynamically.
    Reads current regime from classification and enforces strategy mappings.
    """

    def __init__(self) -> None:
        """Initialize the default deterministic supervisor."""
        self._strategy_map: dict[Regime, Strategy] = {
            Regime.STABLE: Strategy.NORMAL_OPERATION,
            Regime.INTERMITTENT: Strategy.CONSENSUS_PRIORITY,
            Regime.MARGINAL: Strategy.CONSENSUS_PRIORITY,
            Regime.FRAGMENTED: Strategy.CONNECTIVITY_RECOVERY,
            Regime.ENERGY_CASCADE: Strategy.ENERGY_CONSERVATION,
            Regime.LATENCY_OSCILLATION: Strategy.LATENCY_STABILIZATION,
        }

    def select_strategy(self, regime: Regime) -> Strategy:
        """
        Map current passive regime to dominant active strategy.
        Deterministic translation preserving fog-of-war constraints.
        
        Parameters
        ----------
        regime : Regime
            The classification yielded by the passive TelemetryBuffer.
            
        Returns
        -------
        Strategy
            The corresponding behavioral policy to enable.
        """
        return self._strategy_map.get(regime, Strategy.NORMAL_OPERATION)

    def propose_parameters(self, strategy: Strategy, base_epsilon: float) -> dict[str, float]:
        """
        Propose parameter adjustments based on the active strategy.
        Returns theta_proposed to be routed through the safety projector.
        """
        # Default nominal parameters
        theta = {
            "coverage_gain": 1.0,
            "gossip_epsilon": base_epsilon,
            "broadcast_rate": 1.0,
            "auction_participation": 1.0,
            "velocity_scale": 1.0,
        }
        
        if strategy == Strategy.NORMAL_OPERATION:
            pass
            
        elif strategy == Strategy.CONSENSUS_PRIORITY:
            theta["gossip_epsilon"] = base_epsilon * 1.5
            theta["auction_participation"] = 0.0
            
        elif strategy == Strategy.CONNECTIVITY_RECOVERY:
            theta["coverage_gain"] = 0.0  # Intent: Stop scattering. Will be clipped to 0.5 by projector.
            theta["gossip_epsilon"] = base_epsilon * 2.0
            theta["auction_participation"] = 0.0
            
        elif strategy == Strategy.ENERGY_CONSERVATION:
            theta["broadcast_rate"] = 0.5
            theta["auction_participation"] = 0.0
            theta["velocity_scale"] = 0.0 # Intent: Freeze. Clamped to 0.5
            theta["coverage_gain"] = 0.0
            
        elif strategy == Strategy.LATENCY_STABILIZATION:
            theta["broadcast_rate"] = 0.5
            theta["gossip_epsilon"] = base_epsilon * 0.5
            
        return theta
