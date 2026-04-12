"""
consensus_panel.py — Display consensus convergence metrics (Phase 2T)
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

from src.analytics.consensus_analyzer import ConsensusMetrics


class ConsensusPanel(QFrame):
    """
    Displays real-time consensus stability, tracking global variance
    and convergence rates via dual sparklines.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history_len = 100
        self._var_data = collections.deque(maxlen=self._history_len)
        self._rate_data = collections.deque(maxlen=self._history_len)
        
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
        header = QLabel("Consensus State Analyzer")
        header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header.setStyleSheet("color: #a371f7; border: none;")
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
        
        self.lbl_var = QLabel("Var: 0.000")
        self.lbl_var.setFont(QFont("Consolas", 9))
        self.lbl_var.setStyleSheet("color: #c9d1d9; border: none;")
        
        self.lbl_norm = QLabel("Norm: 0.000")
        self.lbl_norm.setFont(QFont("Consolas", 9))
        self.lbl_norm.setStyleSheet("color: #8b949e; border: none;")
        
        self.lbl_rate = QLabel("Rate: 0.000")
        self.lbl_rate.setFont(QFont("Consolas", 9))
        self.lbl_rate.setStyleSheet("color: #79c0ff; border: none;")
        
        self.lbl_err = QLabel("Err: 0.000")
        self.lbl_err.setFont(QFont("Consolas", 9))
        self.lbl_err.setStyleSheet("color: #8b949e; border: none;")

        grid.addWidget(self.lbl_var, 0, 0)
        grid.addWidget(self.lbl_norm, 0, 1)
        grid.addWidget(self.lbl_rate, 1, 0)
        grid.addWidget(self.lbl_err, 1, 1)
        
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
        
        # Dual curves: variance and rate
        self.curve_var = self.plot.plot(pen=pg.mkPen(color="#a371f7", width=2))
        self.curve_rate = self.plot.plot(pen=pg.mkPen(color="#79c0ff", width=1, style=Qt.PenStyle.DashLine))
        
        layout.addWidget(self.plot)

    def update_metrics(self, metrics: ConsensusMetrics) -> None:
        self.lbl_var.setText(f"Var: {metrics.smoothed_variance:.4f}")
        self.lbl_norm.setText(f"Norm: {metrics.normalized_variance:.4f}")
        self.lbl_rate.setText(f"Rate: {metrics.smoothed_rate:+.4f}")
        self.lbl_err.setText(f"Err: {metrics.consensus_error:.4f}")
        
        state = metrics.state
        self.lbl_state.setText(state)
        
        if state == "CONVERGED":
            color = "#3fb950"  # GREEN
        elif state == "SLOW_CONVERGENCE":
            color = "#d29922"  # YELLOW
        elif state == "DIVERGING":
            color = "#f85149"  # RED
        elif state == "OSCILLATING":
            color = "#ff7b72"  # ORANGE
        elif state == "STALLED":
            color = "#8b949e"  # GREY
        else:
            color = "#c9d1d9"
            
        self.lbl_state.setStyleSheet(f"color: {color}; border: none;")
        
        # Update charts
        self._var_data.append(metrics.smoothed_variance)
        self._rate_data.append(metrics.smoothed_rate)
        
        y_var = np.array(self._var_data)
        y_rate = np.array(self._rate_data)
        
        # Normalize visually so both fit in same plot space, just for trend matching
        if y_var.max() > 0:
            y_var = (y_var / y_var.max()) * 10.0
            
        rate_max = np.abs(y_rate).max()
        if rate_max > 0:
            y_rate = (y_rate / rate_max) * 5.0 + 5.0 # Center around 5.0
            
        self.curve_var.setData(y=y_var)
        self.curve_rate.setData(y=y_rate)
