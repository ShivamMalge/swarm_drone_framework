"""
percolation_panel.py — Display connectivity metrics (Phase 2Q)
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget
)

from src.analytics.percolation_analyzer import PercolationMetrics


class PercolationPanel(QFrame):
    """
    Displays real-time connectivity ratio, components, and LCC size.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumHeight(80)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QLabel("Percolation Analyzer")
        header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header.setStyleSheet("color: #58a6ff; border: none;")
        layout.addWidget(header)

        # Metrics row
        row = QHBoxLayout()
        self.lbl_stats = QLabel("Ratio: 0.00 | Comp: 0 | LCC: 0")
        self.lbl_stats.setFont(QFont("Consolas", 8))
        self.lbl_stats.setStyleSheet("color: #c9d1d9; border: none;")
        row.addWidget(self.lbl_stats)
        row.addStretch()
        layout.addLayout(row)

        # Progress bar
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        self.bar.setStyleSheet(self._get_bar_style("#3fb950"))
        layout.addWidget(self.bar)

    def _get_bar_style(self, color: str) -> str:
        return f"""
            QProgressBar {{
                border: 1px solid #30363d;
                border-radius: 4px;
                background-color: #0d1117;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """

    def update_metrics(self, metrics: PercolationMetrics) -> None:
        ratio = metrics.connectivity_ratio
        
        # Color coding
        if ratio > 0.7:
            color = "#3fb950"  # GREEN -> stable
        elif ratio >= 0.5:
            color = "#d29922"  # YELLOW -> degrading
        else:
            color = "#f85149"  # RED -> fragmented
            
        self.bar.setValue(int(ratio * 100))
        self.bar.setStyleSheet(self._get_bar_style(color))
        
        self.lbl_stats.setText(
            f"Ratio: {ratio:.2f} | Comp: {metrics.num_components} | LCC: {metrics.lcc_size}/{metrics.total_agents}"
        )
