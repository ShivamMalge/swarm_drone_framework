"""
spectral_alarm_panel.py — GUI for the Phase 2AB Spectral Alarm.

Displays lambda-2 value, alarm state (colour-coded), normalised value,
margin, and a sparkline of recent lambda-2 history.

Uses a preallocated NumPy ring buffer (no per-frame allocation).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from src.analytics.spectral_alarm import SpectralAlarmResult

_STATE_COLORS = {
    "STABLE":    "#3fb950",
    "WEAKENING": "#d29922",
    "CRITICAL":  "#f85149",
}


class SpectralAlarmPanel(QFrame):
    """Renders spectral alarm state, lambda-2 value, and a sparkline."""

    HISTORY_LEN = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Preallocated ring buffer
        self._buf = np.zeros(self.HISTORY_LEN, dtype=np.float64)
        self._idx = 0
        self._count = 0

        self._pulse_on = False
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(400)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._init_ui()

    # ── UI setup ────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setMinimumHeight(130)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d;"
            " border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Header row
        hdr = QHBoxLayout()
        title = QLabel("Spectral Alarm")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #e6edf3; border: none;")
        hdr.addWidget(title)

        self.lbl_state = QLabel("STABLE")
        self.lbl_state.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_state.setStyleSheet("color: #3fb950; border: none;")
        hdr.addWidget(self.lbl_state)
        layout.addLayout(hdr)

        # Value row
        val_row = QHBoxLayout()
        self.lbl_lambda2 = QLabel("L2 = 0.000")
        self.lbl_lambda2.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self.lbl_lambda2.setStyleSheet("color: #e6edf3; border: none;")
        val_row.addWidget(self.lbl_lambda2)

        self.lbl_delta = QLabel("D = +0.000")
        self.lbl_delta.setFont(QFont("Consolas", 9))
        self.lbl_delta.setStyleSheet("color: #8b949e; border: none;")
        self.lbl_delta.setAlignment(Qt.AlignmentFlag.AlignRight)
        val_row.addWidget(self.lbl_delta)
        layout.addLayout(val_row)

        # Margin / normalised row
        extra = QHBoxLayout()
        self.lbl_margin = QLabel("Margin: 0.000")
        self.lbl_margin.setFont(QFont("Consolas", 8))
        self.lbl_margin.setStyleSheet("color: #8b949e; border: none;")
        extra.addWidget(self.lbl_margin)

        self.lbl_norm = QLabel("Norm: 0.000")
        self.lbl_norm.setFont(QFont("Consolas", 8))
        self.lbl_norm.setStyleSheet("color: #8b949e; border: none;")
        self.lbl_norm.setAlignment(Qt.AlignmentFlag.AlignRight)
        extra.addWidget(self.lbl_norm)
        layout.addLayout(extra)

        # Sparkline
        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(40)
        self.plot.setBackground(None)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)
        self.plot.setYRange(0, 1.2)

        self.curve = self.plot.plot(
            pen=pg.mkPen(color="#3fb950", width=2))
        self.plot.addLine(
            y=0.3, pen=pg.mkPen(color="#f85149", width=1,
                                style=Qt.PenStyle.DashLine))
        self.plot.addLine(
            y=0.6, pen=pg.mkPen(color="#d29922", width=1,
                                style=Qt.PenStyle.DashLine))
        layout.addWidget(self.plot)

    # ── Public API ──────────────────────────────────────────

    def update_alarm(self, r: SpectralAlarmResult) -> None:
        color = _STATE_COLORS.get(r.state, "#8b949e")

        self.lbl_state.setText(r.state)
        self.lbl_state.setStyleSheet(f"color: {color}; border: none;")
        self.lbl_lambda2.setText(f"L2 = {r.lambda2:.3f}")
        self.lbl_delta.setText(f"D = {r.delta_lambda2:+.4f}")
        self.lbl_margin.setText(f"Margin: {r.spectral_margin:+.3f}")
        self.lbl_norm.setText(f"Norm: {r.lambda2_normalized:.4f}")

        self.curve.setPen(pg.mkPen(color=color, width=2))

        # Pulse effect for CRITICAL
        if r.state == "CRITICAL":
            if not self._pulse_timer.isActive():
                self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
            self.lbl_state.setVisible(True)

        # Ring-buffer update (no allocation)
        self._buf[self._idx] = r.lambda2
        self._idx = (self._idx + 1) % self.HISTORY_LEN
        self._count = min(self._count + 1, self.HISTORY_LEN)

        # Build view without copy when possible
        if self._count < self.HISTORY_LEN:
            view = self._buf[:self._count]
        else:
            view = np.roll(self._buf, -self._idx)

        self.curve.setData(y=view)

    # ── Internal ────────────────────────────────────────────

    def _toggle_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        self.lbl_state.setVisible(self._pulse_on)
