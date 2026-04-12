"""
swarm_health.py — Phase 2W Swarm Health Index integrating analytical dimensions.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from PySide6.QtCore import QObject, Signal

from src.analytics.percolation_analyzer import PercolationMetrics
from src.analytics.spectral_analyzer import SpectralMetrics
from src.analytics.energy_cascade_analyzer import EnergyMetrics
from src.analytics.consensus_analyzer import ConsensusMetrics


@dataclass
class HealthMetrics:
    health_score: float
    raw_score: float
    state: str


class SwarmHealthAnalyzer(QObject):
    """
    Computes a composite swarm health index reflecting global stability
    from multiple orthogonal domains (connectivity, energy, consensus).
    """

    health_updated = Signal(float)  # timestamp

    def __init__(self) -> None:
        super().__init__()
        
        # Weights (sum = 1.0)
        self.w_perc = 0.3
        self.w_spec = 0.2
        self.w_eng = 0.3
        self.w_cns = 0.2

        self._last_hash = -1
        self._last_metrics: HealthMetrics | None = None
        self._ema_health = 1.0

    def analyze(self, 
                perc: PercolationMetrics, 
                spec: SpectralMetrics, 
                eng: EnergyMetrics, 
                cns: ConsensusMetrics, 
                time_t: float) -> HealthMetrics:
        
        # 1. Hash-based caching boundary
        # Using pure Python tuples hash for speed mapping primitives
        state_tuple = (
            perc.normalized_ratio, perc.d_ratio_dt, perc.state,
            spec.lambda2_normalized, spec.dl2_dt, spec.state,
            eng.normalized_energy, eng.d_mean_energy_dt, eng.state,
            cns.normalized_variance, cns.global_convergence_rate, cns.state
        )
        
        current_hash = hash(state_tuple)
        if current_hash == self._last_hash and self._last_metrics is not None:
            return self._last_metrics
            
        self._last_hash = current_hash

        # 2. Base Health Score
        n_var = np.clip(cns.normalized_variance, 0.0, 1.0) if not np.isnan(cns.normalized_variance) else 0.0
        n_eng = np.clip(eng.normalized_energy, 0.0, 1.0) if not np.isnan(eng.normalized_energy) else 0.0
        n_perc = np.clip(perc.normalized_ratio, 0.0, 1.0) if not np.isnan(perc.normalized_ratio) else 0.0
        n_spec = np.clip(spec.lambda2_normalized, 0.0, 1.0) if not np.isnan(spec.lambda2_normalized) else 0.0
        
        raw_health = (
            self.w_perc * n_perc +
            self.w_spec * n_spec +
            self.w_eng * n_eng +
            self.w_cns * (1.0 - n_var)
        )

        # 3. Rate Penalty
        # Punish dropping values or rising variance
        # d_ratio_dt < 0 => penalty. d_var_dt > 0 => penalty
        p_perc = min(0.0, perc.d_ratio_dt) * -0.5
        p_spec = min(0.0, spec.dl2_dt) * -0.5
        p_eng = min(0.0, eng.d_mean_energy_dt) * -0.1
        p_cns = max(0.0, cns.global_convergence_rate) * 0.1
        
        # Clamp penalties loosely to prevent infinite explosion
        penalty = float(np.clip(p_perc + p_spec + p_eng + p_cns, 0.0, 0.5))
        
        score = raw_health - penalty

        # 4. Margin penalties
        # If margins are extremely close to failure (< 0.05), subtract small safety factor
        if perc.connectivity_margin < 0.05: score -= 0.05
        if spec.spectral_margin < 0.05: score -= 0.05
        if eng.cascade_margin > -0.05: score -= 0.05 # Cascade happens when intensity exceeds threshold, margin > 0.
        
        # 5. Event Override
        if perc.state == "COLLAPSE":
            score = min(score, 0.15)
        if eng.state == "CASCADE":
            score *= 0.5
        if cns.state == "DIVERGING":
            score *= 0.9

        # Ensure bounds strictly
        if np.isnan(score):
            score = 0.0
        
        score = float(np.clip(score, 0.0, 1.0))

        # UI Smoothing
        self._ema_health = 0.8 * self._ema_health + 0.2 * score
        
        # State logic
        if score > 0.8:
            state = "HEALTHY"
        elif score > 0.5:
            state = "DEGRADING"
        elif score > 0.2:
            state = "CRITICAL"
        else:
            state = "COLLAPSE"

        metrics = HealthMetrics(
            health_score=self._ema_health,
            raw_score=score,
            state=state
        )
        self._last_metrics = metrics
        self.health_updated.emit(time_t)
        return metrics
