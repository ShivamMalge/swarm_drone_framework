"""
energy_cascade_analyzer.py — Phase 2S Energy Cascade Analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from PySide6.QtCore import QObject, Signal

from src.telemetry.telemetry_frame import TelemetryFrame


@dataclass
class EnergyMetrics:
    mean_energy: float
    smoothed_mean_energy: float
    min_energy: float
    energy_variance: float
    alive_count: int
    failure_count: int
    new_failures: int
    cascade_intensity: float
    smoothed_intensity: float
    normalized_energy: float
    cascade_margin: float
    d_mean_energy_dt: float
    energy_gradients: np.ndarray  # E_i - mean(E_neighbors)
    dead_agents: np.ndarray
    hotspot_indices: list[int]    # Agents in critical energy cluster
    state: str                    # "STABLE", "DRAINING", "CASCADE"


class _RingBuffer:
    """Fixed-size rolling NumPy buffer."""
    def __init__(self, size: int) -> None:
        self.buffer = np.zeros(size, dtype=np.float64)
        self.size = size
        self.index = 0
        self.count = 0

    def append(self, val: float) -> None:
        self.buffer[self.index] = val
        self.index = (self.index + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def as_array(self) -> np.ndarray:
        if self.count < self.size:
            return self.buffer[:self.count]
        return np.roll(self.buffer, -self.index)

    def sum(self) -> float:
        if self.count == 0:
            return 0.0
        return float(np.sum(self.buffer[:self.count]))


class EnergyCascadeAnalyzer(QObject):
    """
    Detects and tracks energy depletion and cascading failures.
    """

    energy_cascade_detected = Signal(float)
    energy_cascade_recovered = Signal(float)
    energy_stress_warning = Signal(float)

    def __init__(
        self,
        cascade_threshold: float = 0.1,
        recovery_threshold: float = 0.05,
        stress_drop_rate: float = 2.0,  # Joules/sec threshold
        window_frames: int = 10,
    ) -> None:
        super().__init__()
        self._cascade_threshold = cascade_threshold
        self._recovery_threshold = recovery_threshold
        self._stress_drop_rate = stress_drop_rate
        self._window_frames = window_frames

        self._in_cascade: bool = False
        self._stress_warned: bool = False

        self._last_energies: np.ndarray | None = None
        self._last_failures: np.ndarray | None = None
        self._last_time: float = -1.0
        self._last_mean_energy: float = -1.0

        self._failure_window = _RingBuffer(window_frames)

        # UI Smoothing buffers
        self._ema_mean_energy: float = -1.0
        self._ema_intensity: float = 0.0

    def analyze_frame(self, frame: TelemetryFrame) -> EnergyMetrics:
        # 1. Byte-level hashing
        state_hash = hash(frame.adjacency.data.tobytes() + 
                          frame.energies.tobytes() + 
                          frame.drone_failure_flags.tobytes())
                          
        if not hasattr(self, '_last_state_hash') or state_hash != self._last_state_hash or not hasattr(self, '_cached_heavy_metrics'):
            self._last_state_hash = state_hash
            
            # Heavy compute block
            N = len(frame.energies)
            alive_mask = ~frame.drone_failure_flags
            alive_count = int(alive_mask.sum())
            failure_count = N - alive_count

            # Prevent empty array issues
            if alive_count > 0:
                e_alive = frame.energies[alive_mask]
                mean_e = float(np.mean(e_alive))
                min_e = float(np.min(e_alive))
                var_e = float(np.var(e_alive))
            else:
                mean_e = 0.0
                min_e = 0.0
                var_e = 0.0
                
            # Initial energy tracking for normalization
            if not hasattr(self, '_initial_mean_energy'):
                self._initial_mean_energy = mean_e if mean_e > 0 else 1.0

            # Energy Gradient Vector computation
            gradients = np.zeros(N, dtype=np.float64)
            hotspots = []
            
            # Efficient neighborhood gradient (NumPy broadcast)
            if alive_count > 0:
                adj = frame.adjacency.astype(np.float64)
                degrees = np.sum(adj, axis=1)
                neighbor_sum = adj @ frame.energies
                safe_deg = np.where(degrees > 0, degrees, 1.0)
                neighbor_mean = neighbor_sum / safe_deg
                neighbor_mean[degrees == 0] = frame.energies[degrees == 0]
                
                gradients = frame.energies - neighbor_mean
                
                max_e = frame.energies.max()
                if max_e > 0:
                    stress_mask = alive_mask & (frame.energies < 0.2 * max_e) & (gradients < -2.0)
                    hotspots = np.where(stress_mask)[0].tolist()
                    
            self._cached_heavy_metrics = {
                'mean_e': mean_e,
                'min_e': min_e,
                'var_e': var_e,
                'alive_count': alive_count,
                'failure_count': failure_count,
                'gradients': gradients,
                'hotspots': hotspots
            }

        cm = self._cached_heavy_metrics
        mean_e = cm['mean_e']
        min_e = cm['min_e']
        var_e = cm['var_e']
        alive_count = cm['alive_count']
        failure_count = cm['failure_count']
        gradients = cm['gradients']
        hotspots = cm['hotspots']
        
        dt = frame.time - self._last_time if self._last_time >= 0 else 1.0
        if dt <= 0:
            dt = 1e-6

        # Failure delta mapping
        new_failures = 0
        if self._last_failures is not None:
            # new failure = was False (alive), now is True (dead)
            new_fails_arr = frame.drone_failure_flags & (~self._last_failures)
            new_failures = int(new_fails_arr.sum())
            
        self._failure_window.append(new_failures)
        
        # intensity = new_failures in window / alive
        if alive_count > 0:
            cascade_intensity = self._failure_window.sum() / alive_count
        elif failure_count > 0 and alive_count == 0:
            cascade_intensity = 1.0 # Terminal cascade
        else:
            cascade_intensity = 0.0

        # Mean drop calculation (energy rate of change)
        mean_drop_rate = 0.0
        if self._last_mean_energy >= 0:
            mean_drop_rate = (self._last_mean_energy - mean_e) / dt


        # Update persistent state
        self._last_failures = frame.drone_failure_flags.copy()
        self._last_energies = frame.energies.copy()
        self._last_time = frame.time
        self._last_mean_energy = mean_e

        # Smoothing for UI
        if self._ema_mean_energy < 0:
            self._ema_mean_energy = mean_e
        else:
            self._ema_mean_energy = 0.8 * self._ema_mean_energy + 0.2 * mean_e
            
        self._ema_intensity = 0.8 * self._ema_intensity + 0.2 * cascade_intensity

        # Cascade Evaluation (Hysteresis) using RAW cascade_intensity
        if not self._in_cascade and cascade_intensity > self._cascade_threshold and mean_drop_rate > 0:
            self._in_cascade = True
            self.energy_cascade_detected.emit(frame.time)
        elif self._in_cascade and cascade_intensity <= self._recovery_threshold:
            self._in_cascade = False
            self.energy_cascade_recovered.emit(frame.time)

        # Early Warning
        if mean_drop_rate > self._stress_drop_rate and new_failures == 0:
            if not self._stress_warned:
                self._stress_warned = True
                self.energy_stress_warning.emit(frame.time)
        else:
            self._stress_warned = False # Reset if drop recovers or failures start

        # General State derivation
        if self._in_cascade:
            state = "CASCADE"
        elif mean_drop_rate > self._stress_drop_rate or cascade_intensity > 0:
            state = "DRAINING"
        else:
            state = "STABLE"

        self._last_metrics = EnergyMetrics(
            mean_energy=mean_e,
            smoothed_mean_energy=self._ema_mean_energy,
            min_energy=min_e,
            energy_variance=var_e,
            alive_count=alive_count,
            failure_count=failure_count,
            new_failures=new_failures,
            cascade_intensity=cascade_intensity,
            smoothed_intensity=self._ema_intensity,
            normalized_energy=mean_e / self._initial_mean_energy,
            cascade_margin=cascade_intensity - self._cascade_threshold,
            d_mean_energy_dt=-mean_drop_rate,  # Using standard math sign convention
            energy_gradients=gradients,
            dead_agents=np.where(frame.drone_failure_flags)[0],
            hotspot_indices=hotspots,
            state=state
        )
        
        return self._last_metrics
