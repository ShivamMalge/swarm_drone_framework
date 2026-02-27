"""
Rule-Based Regime Classification Layer.

Classifies the agent's behavioral environment based solely on
historical fog-of-war local metrics tracking.
"""

from __future__ import annotations

import enum

from src.core.config import RegimeConfig


class Regime(enum.Enum):
    STABLE = "STABLE"
    INTERMITTENT = "INTERMITTENT"
    MARGINAL = "MARGINAL"
    FRAGMENTED = "FRAGMENTED"
    ENERGY_CASCADE = "ENERGY_CASCADE"
    LATENCY_OSCILLATION = "LATENCY_OSCILLATION"


class RegimeClassifier:
    """
    Passively evaluates a smoothed metric dictionary against the configured bounds.
    Does NOT affect execution or adapt parameters. Detection only.
    """

    def __init__(self, config: RegimeConfig) -> None:
        self.config = config

    def classify(self, metrics: dict[str, float]) -> Regime:
        """
        Evaluate smoothed telemetry dictionary to categorize the current environmental regime.

        Parameters
        ----------
        metrics : dict[str, float]
            Dictionary containing 'mean_neighbor_count', 'mean_staleness',
            'mean_variance', and 'energy_slope'.

        Returns
        -------
        Regime
            The classified environmental state bounding the agent.
        """
        density = metrics.get("mean_neighbor_count", 0.0)
        staleness = metrics.get("mean_staleness", 0.0)
        variance = metrics.get("mean_variance", 0.0)
        e_slope = metrics.get("energy_slope", 0.0)

        # 1. FRAGMENTED (Isolation)
        # If deeply isolated or extreme staleness blocking consensus
        if density < self.config.neighbor_low or staleness > self.config.staleness_high * 3:
            return Regime.FRAGMENTED

        # 2. ENERGY_CASCADE (Rapid Terminal Depletion)
        # If the rate of energy consumption implies rapid swarm death
        if e_slope < self.config.energy_slope_critical:
            return Regime.ENERGY_CASCADE

        # 3. LATENCY_OSCILLATION (High Noise / Delayed Updates)
        # Bouncing states due to stale propagation masking the true gradient
        if variance > self.config.variance_high and staleness > self.config.staleness_high:
            return Regime.LATENCY_OSCILLATION

        # 4. MARGINAL (Borderline Stability)
        # Struggling to converge or maintain stable topology
        if variance > self.config.variance_high * 0.5 or staleness > self.config.staleness_high * 0.5:
            return Regime.MARGINAL

        # 5. INTERMITTENT (Occasional Drop / Normal Noise)
        # Mild disruptions, well within stability bounds
        if staleness > self.config.staleness_high * 0.2:
            return Regime.INTERMITTENT

        # Default fallback
        return Regime.STABLE
