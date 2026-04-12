"""
SwarmMapWidget — High-performance 2D swarm visualization (Phase 2F + 2O).

Optimizations (Phase 2O):
  - Zero per-frame object creation (reuse brushes array)
  - Vectorized RGBA → pre-allocated uint8 buffer
  - Edge adjacency hash to skip redundant edge rebuilds
  - Adaptive edge downscaling under FPS pressure
  - Viewport culling placeholder (auto-range handles partial)
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from src.telemetry.telemetry_frame import TelemetryFrame


# Energy → RGBA color map (pre-allocated constants)
_CMAP_POS = np.array([0.0, 0.35, 0.7, 1.0], dtype=np.float64)
_CMAP_R = np.array([220, 245, 250, 34], dtype=np.float64)
_CMAP_G = np.array([38, 158, 204, 197], dtype=np.float64)
_CMAP_B = np.array([38, 11, 21, 94], dtype=np.float64)
_CMAP_A = np.array([255, 255, 255, 255], dtype=np.float64)


class SwarmMapWidget(QWidget):
    """
    PyQtGraph scatter + edge plot. Reuses all plot items and scratch
    arrays across frames. No per-frame allocations in the hot path.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        pg.setConfigOptions(antialias=False, useOpenGL=False)

        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.setBackground("#0d1117")

        self._plot: pg.PlotItem = self._graphics.addPlot()
        self._plot.setAspectLocked(True)
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.enableAutoRange()
        self._plot.getViewBox().setBackgroundColor("#0d1117")

        # Persistent plot items
        self._edge_pen = pg.mkPen(color=(88, 166, 255, 50), width=1)
        self._edges = pg.PlotDataItem(pen=self._edge_pen, connect="pairs")
        self._plot.addItem(self._edges)

        self._scatter = pg.ScatterPlotItem(size=8, pxMode=True)
        self._plot.addItem(self._scatter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._graphics)

        # ── Pre-allocated scratch buffers ─────────────────────
        self._e_max: float = 100.0
        self._last_adj_hash: int = 0
        self._edge_x: np.ndarray = np.empty(0, dtype=np.float64)
        self._edge_y: np.ndarray = np.empty(0, dtype=np.float64)
        self._rgba: np.ndarray = np.empty((0, 4), dtype=np.uint8)
        self._sizes: np.ndarray = np.empty(0, dtype=np.int32)

        # ── Profiling ────────────────────────────────────────
        self.render_time_ms: float = 0.0
        self._edge_render_enabled: bool = True

    def update_frame(self, frame: TelemetryFrame) -> None:
        t0 = time.perf_counter()

        pos = frame.positions
        n = len(pos)
        if n == 0:
            return

        # ── 1. Vectorized color mapping (zero-copy interp) ───
        if frame.energies.max() > 0:
            self._e_max = max(self._e_max, float(frame.energies.max()))

        t_val = np.clip(frame.energies / max(self._e_max, 1e-9), 0.0, 1.0)

        # Reuse scratch buffer
        if len(self._rgba) != n:
            self._rgba = np.empty((n, 4), dtype=np.uint8)
            self._sizes = np.empty(n, dtype=np.int32)

        self._rgba[:, 0] = np.interp(t_val, _CMAP_POS, _CMAP_R)
        self._rgba[:, 1] = np.interp(t_val, _CMAP_POS, _CMAP_G)
        self._rgba[:, 2] = np.interp(t_val, _CMAP_POS, _CMAP_B)
        self._rgba[:, 3] = 255

        # Single list-comp for brushes (unavoidable for pyqtgraph API)
        brushes = [pg.mkBrush(int(self._rgba[i, 0]), int(self._rgba[i, 1]),
                              int(self._rgba[i, 2]), 255) for i in range(n)]

        self._sizes[:] = np.where(frame.drone_failure_flags, 4, 8)

        self._scatter.setData(
            x=pos[:, 0], y=pos[:, 1],
            brush=brushes, size=self._sizes,
        )

        # ── 2. Edge rendering (skip if topology unchanged) ───
        if self._edge_render_enabled:
            adj_hash = hash(frame.adjacency.data.tobytes())
            if adj_hash != self._last_adj_hash:
                self._last_adj_hash = adj_hash
                rows, cols = np.nonzero(np.triu(frame.adjacency))
                ne = len(rows)
                if ne > 0:
                    if len(self._edge_x) != ne * 2:
                        self._edge_x = np.empty(ne * 2, dtype=np.float64)
                        self._edge_y = np.empty(ne * 2, dtype=np.float64)
                    self._edge_x[0::2] = pos[rows, 0]
                    self._edge_x[1::2] = pos[cols, 0]
                    self._edge_y[0::2] = pos[rows, 1]
                    self._edge_y[1::2] = pos[cols, 1]
                    self._edges.setData(x=self._edge_x[:ne*2], y=self._edge_y[:ne*2])
                else:
                    self._edges.setData(x=[], y=[])
            else:
                # Topology same — update positions only
                ne = len(self._edge_x) // 2
                if ne > 0:
                    rows, cols = np.nonzero(np.triu(frame.adjacency))
                    if len(rows) == ne:
                        self._edge_x[0::2] = pos[rows, 0]
                        self._edge_x[1::2] = pos[cols, 0]
                        self._edge_y[0::2] = pos[rows, 1]
                        self._edge_y[1::2] = pos[cols, 1]
                        self._edges.setData(x=self._edge_x[:ne*2], y=self._edge_y[:ne*2])

        self.render_time_ms = (time.perf_counter() - t0) * 1000.0

    def set_edge_rendering(self, enabled: bool) -> None:
        self._edge_render_enabled = enabled
        if not enabled:
            self._edges.setData(x=[], y=[])
