"""
energy_panel.py — Display energy cascade metrics (Phase 2S)
"""

from __future__ import annotations

import collections

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget, QGridLayout
)

from src.analytics.energy_cascade_analyzer import EnergyMetrics


class EnergyPanel(QFrame):
    """
    Displays real-time energy stability and cascade metrics, 
    with dual-plot for trend and failure spikes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history_len = 100
        self._energy_data = collections.deque(maxlen=self._history_len)
        self._intensity_data = collections.deque(maxlen=self._history_len)
        self._fail_data = collections.deque(maxlen=self._history_len)
        
        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumHeight(120)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        # Header Row
        header_row = QHBoxLayout()
        header = QLabel("Energy Cascade Analyzer")
        header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header.setStyleSheet("color: #d2a8ff; border: none;")
        header_row.addWidget(header)
        
        self.lbl_state = QLabel("STABLE")
        self.lbl_state.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_state.setStyleSheet("color: #3fb950; border: none;")
        header_row.addWidget(self.lbl_state)
        
        layout.addLayout(header_row)

        # Metrics Grid
        grid = QGridLayout()
        grid.setSpacing(4)
        
        self.lbl_mean = QLabel("Mean E: 0.00")
        self.lbl_mean.setFont(QFont("Consolas", 9))
        self.lbl_mean.setStyleSheet("color: #c9d1d9; border: none;")
        
        self.lbl_alive = QLabel("Alive: 0")
        self.lbl_alive.setFont(QFont("Consolas", 9))
        self.lbl_alive.setStyleSheet("color: #79c0ff; border: none;")
        
        self.lbl_fail = QLabel("Failures: 0")
        self.lbl_fail.setFont(QFont("Consolas", 9))
        self.lbl_fail.setStyleSheet("color: #ff7b72; border: none;")
        
        self.lbl_intens = QLabel("Intensity: 0.00")
        self.lbl_intens.setFont(QFont("Consolas", 9))
        self.lbl_intens.setStyleSheet("color: #d2a8ff; border: none;")

        grid.addWidget(self.lbl_mean, 0, 0)
        grid.addWidget(self.lbl_alive, 0, 1)
        grid.addWidget(self.lbl_fail, 1, 0)
        grid.addWidget(self.lbl_intens, 1, 1)
        
        layout.addLayout(grid)

        # Graph Area
        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(50)
        self.plot.setBackground(None)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)
        
        # Dual axis (sort of) by mapping to generic Y scale, or just two curves
        self.curve_energy = self.plot.plot(pen=pg.mkPen(color="#3fb950", width=2))
        # Bar graph for failure spikes
        self.bar_fails = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#ff7b72")
        self.plot.addItem(self.bar_fails)
        
        layout.addWidget(self.plot)

    def update_metrics(self, metrics: EnergyMetrics) -> None:
        self.lbl_mean.setText(f"Mean E: {metrics.smoothed_mean_energy:.2f}")
        self.lbl_alive.setText(f"Alive: {metrics.alive_count}")
        self.lbl_fail.setText(f"Failures: {metrics.failure_count}")
        self.lbl_intens.setText(f"Intensity: {metrics.smoothed_intensity:.3f}")
        
        state = metrics.state
        self.lbl_state.setText(state)
        
        if state == "STABLE":
            color = "#3fb950"
        elif state == "DRAINING":
            color = "#d29922"
        else:
            color = "#f85149"
            
        self.lbl_state.setStyleSheet(f"color: {color}; border: none;")
        
        # Update charts
        self._energy_data.append(metrics.smoothed_mean_energy)
        self._fail_data.append(metrics.new_failures)
        
        y_energy = np.array(self._energy_data)
        
        # Normalize energy curve for UI if max>0
        if y_energy.max() > 0:
            y_energy = (y_energy / y_energy.max()) * 10.0
            
        self.curve_energy.setData(y=y_energy)
        
        fails = np.array(self._fail_data)
        x_vals = np.arange(len(fails))
        
        self.bar_fails.setOpts(x=x_vals, height=fails)
