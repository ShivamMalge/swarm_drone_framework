"""
spectral_panel.py — Display spectral stability metrics (Phase 2R)
"""

from __future__ import annotations

import collections

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget
)

from src.analytics.spectral_analyzer import SpectralMetrics


class SpectralPanel(QFrame):
    """
    Displays real-time algebraic connectivity (λ₂) alongside a small 
    sparkline trend graph to highlight proximity to fragmentation.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history_len = 100
        self._l2_data = collections.deque(maxlen=self._history_len)
        
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
        header = QLabel("Spectral Stability (λ₂)")
        header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header.setStyleSheet("color: #58a6ff; border: none;")
        header_row.addWidget(header)
        
        self.lbl_state = QLabel("UNKNOWN")
        self.lbl_state.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_state.setStyleSheet("color: #8b949e; border: none;")
        header_row.addWidget(self.lbl_state)
        
        layout.addLayout(header_row)

        # Metrics Row
        self.lbl_val = QLabel("0.000")
        self.lbl_val.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        self.lbl_val.setStyleSheet("color: #c9d1d9; border: none;")
        
        self.lbl_rate = QLabel("Δ: 0.000/s")
        self.lbl_rate.setFont(QFont("Consolas", 8))
        self.lbl_rate.setStyleSheet("color: #8b949e; border: none;")
        
        metrics_row = QHBoxLayout()
        metrics_row.addWidget(self.lbl_val)
        metrics_row.addStretch()
        metrics_row.addWidget(self.lbl_rate)
        metrics_row.setAlignment(self.lbl_rate, Qt.AlignmentFlag.AlignBottom)
        
        layout.addLayout(metrics_row)

        # Sparkline Graph (PyQtGraph)
        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(40)
        self.plot.setBackground(None)  # transparent to adopt QFrame background
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)
        self.plot.setYRange(0, 2.0) # generic maximum bounds, could dynamically adjust
        
        self.curve = self.plot.plot(pen=pg.mkPen(color="#58a6ff", width=2))
        layout.addWidget(self.plot)

    def update_metrics(self, metrics: SpectralMetrics) -> None:
        val = metrics.lambda2
        self.lbl_val.setText(f"{val:.3f}")
        
        rate_str = f"Δ: {metrics.dl2_dt:+.3f}/s"
        self.lbl_rate.setText(rate_str)
        
        state = metrics.state
        self.lbl_state.setText(state)
        
        if state == "STRONG":
            color = "#3fb950"
        elif state == "WEAKENING":
            color = "#d29922"
        else:
            color = "#f85149"
            
        self.lbl_state.setStyleSheet(f"color: {color}; border: none;")
        self.lbl_val.setStyleSheet(f"color: {color}; border: none;")
        self.curve.setPen(pg.mkPen(color=color, width=2))

        # Update sparkline
        self._l2_data.append(val)
        self.curve.setData(y=np.array(self._l2_data))
