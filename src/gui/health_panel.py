"""
health_panel.py — Displays swarm health metrics visually linking analytics mathematically to visual gauges.
"""

from __future__ import annotations

import collections
import numpy as np

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget, QProgressBar
)

from src.analytics.swarm_health import HealthMetrics


class HealthPanel(QFrame):
    """
    Shows composite swarm health with an EMA smoothed progression and visual safety boundaries.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history_len = 200
        self._data = collections.deque(maxlen=self._history_len)
        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumHeight(140)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header Row
        header_row = QHBoxLayout()
        header = QLabel("Swarm Health Index")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setStyleSheet("color: #e6edf3; border: none;")
        header_row.addWidget(header)
        
        self.lbl_state = QLabel("HEALTHY")
        self.lbl_state.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignRight)
        header_row.addWidget(self.lbl_state)
        
        layout.addLayout(header_row)

        # Gauge / Progress Bar
        self.gauge = QProgressBar()
        self.gauge.setMinimum(0)
        self.gauge.setMaximum(100)
        self.gauge.setValue(100)
        self.gauge.setTextVisible(True)
        self.gauge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gauge.setFixedHeight(18)
        self.gauge.setStyleSheet(
            "QProgressBar { border: 1px solid #30363d; border-radius: 4px; background-color: #0d1117; text-align: center; color: white;} "
            "QProgressBar::chunk { background-color: #3fb950; border-radius: 3px; }"
        )
        layout.addWidget(self.gauge)

        # Graph Area
        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(60)
        self.plot.setBackground(None)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)
        self.plot.setYRange(0, 1.05)
        
        self.curve = self.plot.plot(pen=pg.mkPen(color="#3fb950", width=2))
        # Add collapse limit line
        self.plot.addLine(y=0.2, pen=pg.mkPen(color="#f85149", width=1, style=Qt.PenStyle.DashLine))
        
        layout.addWidget(self.plot)

    def update_metrics(self, metrics: HealthMetrics) -> None:
        score = metrics.health_score
        
        self.lbl_state.setText(metrics.state)
        self.gauge.setValue(int(score * 100))
        self.gauge.setFormat(f"Health: {score:.2f}")

        # Coloring bounds
        if metrics.state == "HEALTHY":
            color = "#3fb950"
        elif metrics.state == "DEGRADING":
            color = "#d29922"
        elif metrics.state == "CRITICAL":
            color = "#f85149"
        else: # COLLAPSE
            color = "#ff7b72"
            
        # Update styles
        self.gauge.setStyleSheet(
            f"QProgressBar {{ border: 1px solid #30363d; border-radius: 4px; background-color: #0d1117; text-align: center; color: white;}} "
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}"
        )
        self.lbl_state.setStyleSheet(f"color: {color}; border: none;")
        self.curve.setPen(pg.mkPen(color=color, width=2))

        # Update chart
        self._data.append(score)
        self.curve.setData(y=np.array(self._data))
