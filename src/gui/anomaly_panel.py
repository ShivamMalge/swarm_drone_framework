"""
anomaly_panel.py — Displays anomaly detection counters and metrics dynamically tracking agent outliers.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget, QGridLayout
)

from src.analytics.anomaly_detector import AnomalyMetrics


class AnomalyPanel(QFrame):
    """
    Shows per-agent anomaly metrics identifying drifting, stressing, or disagreeing nodes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumHeight(100)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header Row
        header_row = QHBoxLayout()
        header = QLabel("Anomaly Intelligence")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setStyleSheet("color: #e6edf3; border: none;")
        header_row.addWidget(header)
        
        self.lbl_state = QLabel("NORMAL")
        self.lbl_state.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_state.setStyleSheet("color: #3fb950; border: none;")
        header_row.addWidget(self.lbl_state)
        
        layout.addLayout(header_row)

        # Metrics Grid
        grid = QGridLayout()
        grid.setSpacing(4)
        
        self.lbl_anomalies = QLabel("Anomalies: 0 (0 clusters)")
        self.lbl_anomalies.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.lbl_anomalies.setStyleSheet("color: #f85149; border: none;")
        
        self.lbl_suspicious = QLabel("Suspicious: 0")
        self.lbl_suspicious.setFont(QFont("Consolas", 10))
        self.lbl_suspicious.setStyleSheet("color: #d29922; border: none;")
        
        grid.addWidget(self.lbl_anomalies, 0, 0)
        grid.addWidget(self.lbl_suspicious, 0, 1)
        
        layout.addLayout(grid)
        layout.addStretch()

    def update_metrics(self, metrics: AnomalyMetrics) -> None:
        import numpy as np
        num_a = metrics.anomaly_count
        num_s = int(np.sum(metrics.class_labels == 1))
        
        c_count = metrics.cluster_count
        c_large = metrics.largest_cluster
        
        if c_count > 0:
            self.lbl_anomalies.setText(f"Anomalies: {num_a} ({c_count} clusters, max {c_large})")
        else:
            self.lbl_anomalies.setText(f"Anomalies: {num_a}")
            
        self.lbl_suspicious.setText(f"Suspicious: {num_s}")

        if num_a > 0:
            self.lbl_state.setText("ANOMALIES DETECTED")
            self.lbl_state.setStyleSheet("color: #f85149; border: none;")
        elif num_s > 0:
            self.lbl_state.setText("SUSPICIOUS")
            self.lbl_state.setStyleSheet("color: #d29922; border: none;")
        else:
            self.lbl_state.setText("NORMAL")
            self.lbl_state.setStyleSheet("color: #3fb950; border: none;")
