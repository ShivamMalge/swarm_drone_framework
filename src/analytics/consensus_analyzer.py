"""
consensus_analyzer.py — Phase 2T Consensus Convergence Analyzer.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from PySide6.QtCore import QObject, Signal

from src.telemetry.telemetry_frame import TelemetryFrame


@dataclass
class ConsensusMetrics:
    consensus_variance: float
    consensus_error: float
    normalized_variance: float
    smoothed_variance: float
    smoothed_rate: float
    consensus_margin: float
    global_convergence_rate: float
    hotspot_indices: list[int]
    state: str


class ConsensusAnalyzer(QObject):
    """
    Analyzes swarm consensus convergence, tracking variance and oscillation states.
    """

    consensus_converged = Signal(float)
    consensus_diverging = Signal(float)
    consensus_oscillating = Signal(float)
    consensus_stalled = Signal(float)

    def __init__(
        self,
        convergence_threshold: float = 1e-4,
        history_size: int = 100,
    ) -> None:
        super().__init__()
        self._convergence_threshold = convergence_threshold
        self._history_size = history_size

        self._last_state_hash: int = -1
        self._last_time: float = -1.0
        self._last_metrics: ConsensusMetrics | None = None

        self._initial_variance: float | None = None
        
        self.rate_history = collections.deque(maxlen=history_size)
        
        self._ema_variance: float = -1.0
        self._ema_rate: float = 0.0

        self._current_state_enum = "STABLE"

    def analyze_frame(self, frame: TelemetryFrame) -> ConsensusMetrics:
        # byte-level hashing of input state exactly matching 2Q/2R/2S caching consistency
        state_hash = hash(frame.adjacency.data.tobytes() + 
                          frame.agent_states.tobytes() + 
                          frame.drone_failure_flags.tobytes())

        if state_hash == self._last_state_hash and self._last_time >= 0 and hasattr(self, '_cached_heavy_metrics'):
            cm = self._cached_heavy_metrics
        else:
            self._last_state_hash = state_hash
            
            # Heavy compute block
            N = len(frame.agent_states)
            alive_mask = ~frame.drone_failure_flags
            alive_count = int(alive_mask.sum())
            
            if alive_count > 0:
                x_alive = frame.agent_states[alive_mask]
                mean_x = float(np.mean(x_alive))
                var_x = float(np.var(x_alive))
                # consensus_error = mean(|x_i - mean(x)|)
                err_x = float(np.mean(np.abs(x_alive - mean_x)))
            else:
                mean_x = 0.0
                var_x = 0.0
                err_x = 0.0
                
            if self._initial_variance is None:
                self._initial_variance = var_x if var_x > 0 else 1.0

            # Local Disagreement and Spatial Hotspots
            hotspots = []
            if alive_count > 0:
                adj = frame.adjacency.astype(np.float64)
                degrees = np.sum(adj, axis=1)
                
                neighbor_sum = adj @ frame.agent_states
                safe_deg = np.where(degrees > 0, degrees, 1.0)
                neighbor_mean = neighbor_sum / safe_deg
                neighbor_mean[degrees == 0] = frame.agent_states[degrees == 0]
                
                local_disagreement = np.abs(frame.agent_states - neighbor_mean)
                
                # Detect max |x_i - neighbor_mean|
                # Only consider alive agents for hotspots
                masked_disagreement = np.where(alive_mask, local_disagreement, -1.0)
                max_disagreement = np.max(masked_disagreement)
                
                # We identify hotspots if their disagreement is significantly large
                # For instance, > 0 and basically at or near the max
                if max_disagreement > 1e-5:
                    hotspots = np.where((masked_disagreement >= max_disagreement * 0.9))[0].tolist()

            self._cached_heavy_metrics = {
                'var_x': var_x,
                'err_x': err_x,
                'hotspots': hotspots
            }
            cm = self._cached_heavy_metrics

        var_x = cm['var_x']
        err_x = cm['err_x']
        hotspots = cm['hotspots']

        # Temporal computation
        dt = frame.time - self._last_time if self._last_time >= 0 else 1.0
        if dt <= 0:
            dt = 1e-6
            
        rate = 0.0
        if self._last_metrics is not None:
            rate = (var_x - self._last_metrics.consensus_variance) / dt
            
        self.rate_history.append(rate)
        
        self._last_time = frame.time

        # UI Smoothing
        if self._ema_variance < 0:
            self._ema_variance = var_x
        else:
            self._ema_variance = 0.8 * self._ema_variance + 0.2 * var_x
            
        self._ema_rate = 0.8 * self._ema_rate + 0.2 * rate

        # Detection Logic (Using RAW var_x and rate)
        # Sign flips detection for oscillation
        oscillation_detected = False
        if len(self.rate_history) >= 5:
            # check recent window for sign flips
            rates = np.array(list(self.rate_history)[-5:])
            # Ignore Near-zero noise
            rates[np.abs(rates) < 1e-8] = 0.0
            signs = np.sign(rates)
            # Remove zeros
            signs = signs[signs != 0]
            if len(signs) > 1:
                # Count flips
                flips = np.sum(signs[:-1] != signs[1:])
                if flips >= 2:
                    oscillation_detected = True

        stalled = False
        if var_x > self._convergence_threshold * 10 and abs(rate) < 1e-6:
            stalled = True

        new_state = self._current_state_enum
        
        if oscillation_detected:
            new_state = "OSCILLATING"
            self.consensus_oscillating.emit(frame.time)
        elif var_x < self._convergence_threshold:
            if new_state != "CONVERGED":
                self.consensus_converged.emit(frame.time)
            new_state = "CONVERGED"
        elif rate > 1e-5:
            if new_state != "DIVERGING":
                self.consensus_diverging.emit(frame.time)
            new_state = "DIVERGING"
        elif stalled:
            if new_state != "STALLED":
                self.consensus_stalled.emit(frame.time)
            new_state = "STALLED"
        elif rate < -1e-5:
            new_state = "SLOW_CONVERGENCE"
        
        self._current_state_enum = new_state

        metrics = ConsensusMetrics(
            consensus_variance=var_x,
            consensus_error=err_x,
            normalized_variance=var_x / self._initial_variance,
            smoothed_variance=self._ema_variance,
            smoothed_rate=self._ema_rate,
            consensus_margin=var_x - self._convergence_threshold,
            global_convergence_rate=rate,
            hotspot_indices=hotspots,
            state=new_state
        )
        self._last_metrics = metrics
        return metrics
