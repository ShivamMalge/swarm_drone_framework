"""
research_metrics.py — Phase 2Y Research Metrics Engine for experiment-level aggregation.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field, asdict
from typing import Any

from PySide6.QtCore import QObject, Signal

from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.percolation_analyzer import PercolationMetrics
from src.analytics.spectral_analyzer import SpectralMetrics
from src.analytics.energy_cascade_analyzer import EnergyMetrics
from src.analytics.consensus_analyzer import ConsensusMetrics
from src.analytics.anomaly_detector import AnomalyMetrics
from src.analytics.swarm_health import HealthMetrics


@dataclass
class StabilityMetricsData:
    time_to_stability: float = 0.0
    time_in_stable: float = 0.0
    regime_transition_count: int = 0
    _last_regime: str = "UNKNOWN"
    _stable_start: float = -1.0


@dataclass
class ConnectivityMetricsData:
    sum_components: float = 0.0
    min_ratio: float = 1.0
    collapse_duration: float = 0.0


@dataclass
class SpectralMetricsData:
    sum_lambda2: float = 0.0
    min_lambda2: float = float('inf')
    instability_duration: float = 0.0


@dataclass
class EnergyMetricsData:
    sum_energy: float = 0.0
    max_dE_dt: float = float('-inf')
    cascade_frequency: int = 0
    _last_cascade_state: bool = False


@dataclass
class ConsensusMetricsData:
    convergence_time: float = -1.0
    sum_variance_decay: float = 0.0
    oscillation_count: int = 0
    _last_osc: bool = False


@dataclass
class AnomalyMetricsData:
    total_anomalies: int = 0
    peak_anomalies: int = 0
    anomaly_duration: float = 0.0


@dataclass
class HealthMetricsData:
    sum_health: float = 0.0
    min_health: float = 1.0
    time_in_critical: float = 0.0


@dataclass
class RunMetrics:
    scenario_name: str
    seed: int
    total_frames: int
    total_time: float
    
    stability: StabilityMetricsData = field(default_factory=StabilityMetricsData)
    connectivity: ConnectivityMetricsData = field(default_factory=ConnectivityMetricsData)
    spectral: SpectralMetricsData = field(default_factory=SpectralMetricsData)
    energy: EnergyMetricsData = field(default_factory=EnergyMetricsData)
    consensus: ConsensusMetricsData = field(default_factory=ConsensusMetricsData)
    anomaly: AnomalyMetricsData = field(default_factory=AnomalyMetricsData)
    health: HealthMetricsData = field(default_factory=HealthMetricsData)
    
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchMetricsEngine(QObject):
    """
    Observer-only streaming aggregator performing O(1) aggregations
    across live simulations or replay pipelines dynamically.
    """
    
    metrics_updated = Signal(object) # emits RunMetrics
    
    def __init__(self, scenario_name: str = "default", seed: int = 0) -> None:
        super().__init__()
        self.scenario_name = scenario_name
        self.seed = seed
        self.reset()
        
    def reset(self) -> None:
        self.metrics = RunMetrics(self.scenario_name, self.seed, 0, 0.0)
        self._last_time = -1.0
        
    def add_frame_metrics(
        self,
        frame: TelemetryFrame,
        perc: PercolationMetrics,
        spec: SpectralMetrics,
        eng: EnergyMetrics,
        cns: ConsensusMetrics,
        anom: AnomalyMetrics,
        health: HealthMetrics
    ) -> RunMetrics:
        
        t = frame.time
        dt = t - self._last_time if self._last_time >= 0 else 0.0
        self._last_time = t
        
        m = self.metrics
        m.total_frames += 1
        m.total_time = t
        
        # Stability (Regimes approximate using Health states or Regime string)
        # Using regime_state from frame if exists
        regime = "STABLE"
        if frame.regime_state:
            counter = collections.Counter(frame.regime_state.values())
            counter.pop("DEAD", None)
            if counter:
                regime = counter.most_common(1)[0][0]
                
        if regime != m.stability._last_regime:
            if m.total_frames > 1:
                m.stability.regime_transition_count += 1
            m.stability._last_regime = regime
            
        if regime == "STABLE" or health.state == "HEALTHY":
            m.stability.time_in_stable += dt
            if m.stability._stable_start < 0:
                m.stability._stable_start = t
        else:
            if m.stability._stable_start >= 0 and m.stability.time_to_stability == 0:
                m.stability.time_to_stability = t
                
        # Connectivity
        m.connectivity.sum_components += perc.num_components
        if perc.connectivity_ratio < m.connectivity.min_ratio:
            m.connectivity.min_ratio = perc.connectivity_ratio
        if perc.state == "COLLAPSE":
            m.connectivity.collapse_duration += dt
            
        # Spectral
        m.spectral.sum_lambda2 += spec.lambda2_normalized
        if spec.lambda2_normalized < m.spectral.min_lambda2:
            m.spectral.min_lambda2 = spec.lambda2_normalized
        if spec.state == "UNSTABLE":
            m.spectral.instability_duration += dt
            
        # Energy
        m.energy.sum_energy += eng.normalized_energy
        # rate is negative when dropping. We want max DROP rate, i.e., min dE/dt inverted.
        drop_rate = -eng.d_mean_energy_dt
        if drop_rate > m.energy.max_dE_dt:
            m.energy.max_dE_dt = drop_rate
            
        cascade_active = (eng.state == "CASCADE")
        if cascade_active and not m.energy._last_cascade_state:
            m.energy.cascade_frequency += 1
        m.energy._last_cascade_state = cascade_active
        
        # Consensus
        if cns.state == "CONVERGED" and m.consensus.convergence_time < 0:
            m.consensus.convergence_time = t
        m.consensus.sum_variance_decay += cns.global_convergence_rate # usually negative 
        
        osc_active = (cns.state == "OSCILLATING")
        if osc_active and not m.consensus._last_osc:
            m.consensus.oscillation_count += 1
        m.consensus._last_osc = osc_active
        
        # Anomaly
        m.anomaly.total_anomalies += anom.anomaly_count # Actually we want unique anomalies? "total" could be integral
        if anom.anomaly_count > m.anomaly.peak_anomalies:
            m.anomaly.peak_anomalies = anom.anomaly_count
        if anom.anomaly_count > 0:
            m.anomaly.anomaly_duration += dt
            
        # Health
        m.health.sum_health += health.health_score
        if health.health_score < m.health.min_health:
            m.health.min_health = health.health_score
        if health.state in ["CRITICAL", "COLLAPSE"]:
            m.health.time_in_critical += dt
            
        self.metrics_updated.emit(m)
        return m

    def export_json(self, filepath: str) -> None:
        data = self.metrics.as_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
