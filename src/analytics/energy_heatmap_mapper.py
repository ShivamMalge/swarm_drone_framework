"""
energy_heatmap_mapper.py — Phase 2AC Energy Heatmap Mapper.

Computes spatial energy distribution, local gradients, and hotspot masks
for real-time visualization of cascade propagation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.telemetry.telemetry_frame import TelemetryFrame


@dataclass
class EnergyHeatmapResult:
    energy_norm: np.ndarray        # (N,) float64 [0, 1]
    gradient_norm: np.ndarray      # (N,) float64 [0, 1]
    cascade_intensity: np.ndarray  # (N,) float64 [0, 1]
    hotspots: np.ndarray           # (N,) bool


class EnergyHeatmapMapper:
    """
    Computes normalised topological energy gradients to drive the heatmap.
    """

    def __init__(self) -> None:
        self._last_hash = -1
        self._last_result = None

    def map_energies(self, frame: TelemetryFrame) -> EnergyHeatmapResult:
        N = len(frame.positions)
        alive_mask = ~frame.drone_failure_flags

        # Hash-skip based on topology, alive list, and actual energies
        h = hash(frame.adjacency.data.tobytes() + alive_mask.tobytes() + frame.energies.tobytes())
        if h == self._last_hash and self._last_result is not None:
            return self._last_result
        self._last_hash = h

        # Initialize outputs
        e_norm = np.zeros(N, dtype=np.float64)
        g_norm = np.zeros(N, dtype=np.float64)
        intensity = np.zeros(N, dtype=np.float64)
        hotspots = np.zeros(N, dtype=bool)

        if N == 0 or not np.any(alive_mask):
            res = EnergyHeatmapResult(e_norm, g_norm, intensity, hotspots)
            self._last_result = res
            return res

        # 1. Normalise active energies
        active_e = frame.energies[alive_mask]
        e_min = np.min(active_e)
        e_max = np.max(active_e)
        
        range_e = max(e_max - e_min, 1e-6)
        normalized_e = (frame.energies - e_min) / range_e
        
        # Clamp active agents to [0, 1]; dead stay 0
        e_norm[alive_mask] = np.clip(normalized_e[alive_mask], 0.0, 1.0)
        
        # 2. Compute local gradients
        adj = frame.adjacency.astype(np.float64)
        # Prevent dead agents from contributing to neighbor mean
        adj[~alive_mask, :] = 0
        adj[:, ~alive_mask] = 0
        
        degree = np.sum(adj, axis=1)
        # Avoid division by zero
        degree_safe = np.maximum(degree, 1.0)
        
        # Mean energy of neighbors
        neighbor_mean = (adj @ frame.energies) / degree_safe
        
        # Absolute gradient
        # For isolated nodes (degree 0), G = 0 since neighbor_mean is 0
        # Wait, if neighbor_mean is 0, G = abs(E - 0) = E, which is a false gradient!
        # Fix: For isolated nodes, set neighbor_mean to E so G = 0.
        is_isolated = (degree == 0) & alive_mask
        neighbor_mean[is_isolated] = frame.energies[is_isolated]
        
        g_raw = np.abs(frame.energies - neighbor_mean)
        g_raw[~alive_mask] = 0.0
        
        g_max = np.max(g_raw)
        g_norm_active = g_raw / max(g_max, 1e-6)
        g_norm[alive_mask] = np.clip(g_norm_active[alive_mask], 0.0, 1.0)
        
        # 3. Cascade intensity = 0.7 * E_norm + 0.3 * G_norm
        intensity_active = 0.7 * e_norm + 0.3 * g_norm
        intensity[alive_mask] = np.clip(intensity_active[alive_mask], 0.0, 1.0)
        
        # 4. Hotspots detector: G_norm > 0.6 AND E_norm > 0.5
        hotspot_mask = (g_norm > 0.6) & (e_norm > 0.5) & alive_mask
        hotspots[hotspot_mask] = True

        res = EnergyHeatmapResult(
            energy_norm=e_norm,
            gradient_norm=g_norm,
            cascade_intensity=intensity,
            hotspots=hotspots,
        )
        self._last_result = res
        return res
