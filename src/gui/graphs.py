"""
TelemetryGraphsWidget — Optimized rolling time-series (Phase 2G + 2O).

Optimizations:
  - Pre-allocated NumPy ring buffers (no deque → array copy)
  - Single setData call per curve
  - No per-frame allocation
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget, QTabWidget

from src.telemetry.telemetry_frame import TelemetryFrame


class _RingBuffer:
    """Fixed-size ring buffer backed by a pre-allocated NumPy array."""

    __slots__ = ("_buf", "_max", "_count", "_idx")

    def __init__(self, maxlen: int) -> None:
        self._buf = np.zeros(maxlen, dtype=np.float64)
        self._max = maxlen
        self._count = 0
        self._idx = 0

    def append(self, value: float) -> None:
        self._buf[self._idx] = value
        self._idx = (self._idx + 1) % self._max
        if self._count < self._max:
            self._count += 1

    def as_array(self) -> np.ndarray:
        """Return data in chronological order (zero-copy view when full)."""
        if self._count < self._max:
            return self._buf[:self._count]
        return np.roll(self._buf, -self._idx)


class TelemetryGraphsWidget(QWidget):
    """
    Three rolling time-series plots with pre-allocated ring buffers.
    """

    def __init__(self, parent: QWidget | None = None, max_points: int = 300) -> None:
        super().__init__(parent)
        self._max_points = max_points

        # Ring buffers
        self._time_buffer = _RingBuffer(max_points)
        self._spectral_buffer = _RingBuffer(max_points)
        self._energy_buffer = _RingBuffer(max_points)
        self._consensus_buffer = _RingBuffer(max_points)

        pg.setConfigOptions(antialias=False, useOpenGL=False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #30363d; border-radius: 4px; }")
        layout.addWidget(self.tabs)

        self._p_spectral, self._c_spectral = self._create_plot(
            "Spectral Gap (λ₂)", "#58a6ff"
        )
        self.tabs.addTab(self._p_spectral, "Spectral Gap")

        self._p_energy, self._c_energy = self._create_plot(
            "Average Energy", "#3fb950"
        )
        self.tabs.addTab(self._p_energy, "Average Energy")

        self._p_consensus, self._c_consensus = self._create_plot(
            "Consensus Variance", "#d2a8ff"
        )
        self.tabs.addTab(self._p_consensus, "Consensus State")

    def _create_plot(self, title: str, color: str) -> tuple[pg.PlotWidget, pg.PlotDataItem]:
        plot = pg.PlotWidget(title=title)
        plot.setBackground("#161b22")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideAxis('bottom')
        plot.getAxis('left').setPen("#8b949e")
        plot.getAxis('left').setTextPen("#c9d1d9")
        plot.setTitle(title, color="#c9d1d9", size="10pt")

        curve = plot.plot(pen=pg.mkPen(color=color, width=2))
        return plot, curve

    def update_frame(self, frame: TelemetryFrame) -> None:
        avg_energy = float(frame.energies.mean()) if len(frame.energies) > 0 else 0.0

        self._time_buffer.append(frame.time)
        self._spectral_buffer.append(frame.spectral_gap)
        self._energy_buffer.append(avg_energy)
        self._consensus_buffer.append(frame.consensus_variance)

        t_arr = self._time_buffer.as_array()
        self._c_spectral.setData(x=t_arr, y=self._spectral_buffer.as_array())
        self._c_energy.setData(x=t_arr, y=self._energy_buffer.as_array())
        self._c_consensus.setData(x=t_arr, y=self._consensus_buffer.as_array())
