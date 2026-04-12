"""
research_panel.py — Displays experiment-level metrics aggregated dynamically.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget, QGridLayout, QPushButton
)

from src.analytics.research_metrics import RunMetrics


class ResearchPanel(QFrame):
    """
    Displays aggregated stability, connectivity, spectral, and energy metrics dynamically.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumHeight(150)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header Row
        header_row = QHBoxLayout()
        header = QLabel("Research Engine Aggregates")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setStyleSheet("color: #e6edf3; border: none;")
        header_row.addWidget(header)
        
        self.btn_export = QPushButton("Export JSON")
        self.btn_export.setStyleSheet("background-color: #21262d; color: #58a6ff; border: 1px solid #30363d; border-radius: 4px; padding: 2px 6px;")
        self.btn_export.setFixedWidth(80)
        header_row.addWidget(self.btn_export, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addLayout(header_row)

        # Metrics Grid
        grid = QGridLayout()
        grid.setSpacing(6)
        
        # Stability
        self.lbl_stable_pct = self._make_label("Stability: 0.0%")
        self.lbl_t_stable = self._make_label("T-to-stable: -")
        self.lbl_reg_trans = self._make_label("Reg-Trans: 0")
        
        # Connect & Spectral
        self.lbl_avg_comp = self._make_label("Avg Comp: 0.0")
        self.lbl_min_l2 = self._make_label("Min λ2: 0.00")
        self.lbl_col_dur = self._make_label("Collapse-T: 0.0s")
        
        # Energy & Consensus
        self.lbl_cascades = self._make_label("Cascades: 0")
        self.lbl_max_dedt = self._make_label("Drop: 0.0/s")
        self.lbl_conv_time = self._make_label("Conv-T: -")
        
        # Anomaly & Health
        self.lbl_peak_anom = self._make_label("Peak Anom: 0")
        self.lbl_min_health = self._make_label("Min Hlth: 1.00")
        self.lbl_crit_dur = self._make_label("Crit-T: 0.0s")

        grid.addWidget(self.lbl_stable_pct, 0, 0)
        grid.addWidget(self.lbl_t_stable, 0, 1)
        grid.addWidget(self.lbl_reg_trans, 0, 2)
        
        grid.addWidget(self.lbl_avg_comp, 1, 0)
        grid.addWidget(self.lbl_min_l2, 1, 1)
        grid.addWidget(self.lbl_col_dur, 1, 2)
        
        grid.addWidget(self.lbl_cascades, 2, 0)
        grid.addWidget(self.lbl_max_dedt, 2, 1)
        grid.addWidget(self.lbl_conv_time, 2, 2)
        
        grid.addWidget(self.lbl_peak_anom, 3, 0)
        grid.addWidget(self.lbl_min_health, 3, 1)
        grid.addWidget(self.lbl_crit_dur, 3, 2)
        
        layout.addLayout(grid)
        layout.addStretch()

    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Consolas", 8))
        lbl.setStyleSheet("color: #8b949e; border: none;")
        return lbl

    def update_metrics(self, m: RunMetrics) -> None:
        tt = m.total_time if m.total_time > 0 else 1.0
        tf = m.total_frames if m.total_frames > 0 else 1
        
        pct_stable = (m.stability.time_in_stable / tt) * 100.0
        self.lbl_stable_pct.setText(f"Stability: {pct_stable:.1f}%")
        self.lbl_t_stable.setText(f"T-to-stable: {m.stability.time_to_stability:.1f}s" if m.stability.time_to_stability > 0 else "T-to-stable: ---")
        self.lbl_reg_trans.setText(f"Reg-Trans: {m.stability.regime_transition_count}")
        
        self.lbl_avg_comp.setText(f"Avg Comp: {(m.connectivity.sum_components / tf):.1f}")
        self.lbl_min_l2.setText(f"Min λ2: {m.spectral.min_lambda2:.2f}")
        self.lbl_col_dur.setText(f"Collapse-T: {m.connectivity.collapse_duration:.1f}s")
        
        self.lbl_cascades.setText(f"Cascades: {m.energy.cascade_frequency}")
        if m.energy.max_dE_dt > -9999:
            self.lbl_max_dedt.setText(f"Drop: {m.energy.max_dE_dt:.2f}/s")
            
        self.lbl_conv_time.setText(f"Conv-T: {m.consensus.convergence_time:.1f}s" if m.consensus.convergence_time >= 0 else "Conv-T: ---")
        
        self.lbl_peak_anom.setText(f"Peak Anom: {m.anomaly.peak_anomalies}")
        self.lbl_min_health.setText(f"Min Hlth: {m.health.min_health:.2f}")
        self.lbl_crit_dur.setText(f"Crit-T: {m.health.time_in_critical:.1f}s")
