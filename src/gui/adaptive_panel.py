"""
AdaptivePanel — Real-time display for adaptive control parameters (Phase 2I).

Observer layer. Renders the mean value of adaptive parameters across
the swarm. Employs lightweight color flashing to visually denote changes.
No computation beyond simple scalar aggregates.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.telemetry.telemetry_frame import TelemetryFrame


class AdaptivePanel(QWidget):
    """
    Displays current adaptive parameters (Θ).
    Briefly highlights values in yellow if they change significantly.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_values: dict[str, float] = {}

        # Reset timer for color flashing
        self._reset_timer = QTimer(self)
        self._reset_timer.setInterval(500)  # 500ms flash
        self._reset_timer.timeout.connect(self._reset_colors)

        self._labels: dict[str, QLabel] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header = QLabel("Adaptive Control (Θ_safe)")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #58a6ff;")
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(15)

        # Keys we care about from TelemetryFrame.adaptive_parameters
        keys = [
            "coverage_gains",
            "gossip_epsilons",
            "velocity_scales",
            "broadcast_rates",
            "auction_participations",
            "projection_events_total",
        ]

        # Display names
        self._display_names = {
            "coverage_gains": "Coverage Gain",
            "gossip_epsilons": "Gossip Epsilon",
            "velocity_scales": "Velocity Scale",
            "broadcast_rates": "Broadcast Rate",
            "auction_participations": "Auction Active",
            "projection_events_total": "Projections (Σ)",
        }

        for row, key in enumerate(keys):
            name_lbl = QLabel(self._display_names[key])
            name_lbl.setFont(QFont("Segoe UI", 9))
            name_lbl.setStyleSheet("color: #8b949e;")
            grid.addWidget(name_lbl, row, 0, Qt.AlignmentFlag.AlignLeft)

            val_lbl = QLabel("-")
            val_lbl.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            val_lbl.setStyleSheet("color: #c9d1d9;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(val_lbl, row, 1, Qt.AlignmentFlag.AlignRight)

            self._labels[key] = val_lbl

        layout.addLayout(grid)
        layout.addStretch()

    def update_frame(self, frame: TelemetryFrame) -> None:
        """Extract mean parameter values and update UI, flashing on changes."""
        params = frame.adaptive_parameters
        if not params:
            return

        changed = False

        for key, val_lbl in self._labels.items():
            if key not in params:
                continue

            # Compute scalar (either mean of array or raw value)
            raw_val = params[key]
            if isinstance(raw_val, np.ndarray):
                current_val = float(raw_val.mean()) if len(raw_val) > 0 else 0.0
            else:
                current_val = float(raw_val)

            # Compare to detect change
            last_val = self._last_values.get(key)
            
            # Format display
            if "total" in key:
                text = f"{int(current_val)}"
            else:
                text = f"{current_val:.3f}"

            val_lbl.setText(text)

            # Flash yellow if changed > 1e-4
            if last_val is not None and abs(current_val - last_val) > 1e-4:
                val_lbl.setStyleSheet("color: #d29922;")  # Warning Yellow
                changed = True

            self._last_values[key] = current_val

        if changed:
            self._reset_timer.start()

    def _reset_colors(self) -> None:
        """Reset label colors back to default after delay."""
        self._reset_timer.stop()
        for lbl in self._labels.values():
            lbl.setStyleSheet("color: #c9d1d9;")
