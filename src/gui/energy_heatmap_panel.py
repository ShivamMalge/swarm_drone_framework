"""
energy_heatmap_panel.py — Phase 2AC Energy Heatmap Overlay Provider.

Renders node-based continuous colormaps based on energy cascade intensity
and highlights hotspots. Preallocates buffers to achieve <1ms per frame.
"""

from __future__ import annotations

import numpy as np

from src.analytics.energy_heatmap_mapper import EnergyHeatmapResult


class EnergyHeatmapOverlay:
    """
    Maintains preallocated buffers to provide continuous color mapping
    and dynamic size adjustments without per-frame allocations.
    """

    def __init__(self, max_agents: int = 2000) -> None:
        self.colors = np.zeros((max_agents, 4), dtype=np.ubyte)
        self.sizes = np.zeros(max_agents, dtype=np.float64)
        
        # Color palette for interpolation: Blue -> Yellow -> Red
        self._c_low = np.array([31, 119, 180, 200], dtype=np.float64)   # Blue
        self._c_mid = np.array([210, 153, 34, 200], dtype=np.float64)   # Yellow
        self._c_high = np.array([248, 81, 73, 200], dtype=np.float64)   # Red
        
        self._c_dead = np.array([100, 100, 100, 80], dtype=np.ubyte)
        self._c_hotspot = np.array([255, 100, 0, 255], dtype=np.float64) # Bright orange glow

    def update_overlay(
        self,
        result: EnergyHeatmapResult,
        base_sizes: np.ndarray,
        alive_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Maps intensity to colours and hotspots to sizes.
        
        Returns:
            (colors (N, 4) uint8, sizes (N,) float64)
        """
        N = len(alive_mask)
        if N > len(self.colors):
            # Grow buffers if necessary
            self.colors = np.zeros((N * 2, 4), dtype=np.ubyte)
            self.sizes = np.zeros(N * 2, dtype=np.float64)

        # Slice active buffers
        out_c = self.colors[:N]
        out_s = self.sizes[:N]

        # Base sizes
        out_s[:] = base_sizes

        # Dead agents
        out_c[~alive_mask] = self._c_dead

        # Alive agents continuous interpolation
        intensity = result.cascade_intensity

        # Expand to shape (N, 1) for broadcasting
        val = intensity[:, np.newaxis]

        # Mix colors: 0..0.5 maps Blue -> Yellow; 0.5..1.0 maps Yellow -> Red
        # Vectorized interpolation
        mask_low = (intensity <= 0.5) & alive_mask
        if np.any(mask_low):
            norm_low = val[mask_low] * 2.0
            out_c[mask_low] = (self._c_low * (1.0 - norm_low) + self._c_mid * norm_low).astype(np.ubyte)

        mask_high = (intensity > 0.5) & alive_mask
        if np.any(mask_high):
            norm_high = (val[mask_high] - 0.5) * 2.0
            out_c[mask_high] = (self._c_mid * (1.0 - norm_high) + self._c_high * norm_high).astype(np.ubyte)

        # Hotspots overwrite color and boost size
        active_hotspots = result.hotspots & alive_mask
        if np.any(active_hotspots):
            out_c[active_hotspots] = self._c_hotspot.astype(np.ubyte)
            out_s[active_hotspots] *= 1.5

        return out_c, out_s
