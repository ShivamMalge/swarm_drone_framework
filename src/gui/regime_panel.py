"""
RegimePanel — Real-time classification display (Phase 2H).

Observer layer. Renders smoothed global regime and per-agent distribution counts
from the TelemetryFrame's dictionary. No heavy processing.
"""

from __future__ import annotations

import collections

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.telemetry.telemetry_frame import TelemetryFrame


_REGIME_COLORS = {
    "STABLE": "#3fb950",               # Green
    "INTERMITTENT": "#d2a8ff",         # Purple (minor)
    "MARGINAL": "#d29922",             # Yellow/Orange
    "FRAGMENTED": "#f85149",           # Red
    "ENERGY_CASCADE": "#f85149",       # Red
    "LATENCY_OSCILLATION": "#f85149",  # Red
    "DEAD": "#484f58",                 # Gray
    "UNKNOWN": "#8b949e",              # Light Gray
}


class RegimePanel(QWidget):
    """
    Displays the globally dominant smoothed regime and per-agent counts.
    """

    def __init__(
        self, parent: QWidget | None = None, smoothing_window: int = 5
    ) -> None:
        super().__init__(parent)
        self._max_history = smoothing_window
        self._history = collections.deque(maxlen=smoothing_window)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ── 1. Global Dominant Status ────────────────────────
        header = QLabel("Global Regime")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #58a6ff;")
        layout.addWidget(header)

        self._global_label = QLabel("UNKNOWN")
        self._global_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._global_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self._global_label.setStyleSheet("color: #8b949e; background-color: #21262d; border-radius: 4px; padding: 10px;")
        layout.addWidget(self._global_label)

        # ── 2. Distribution Breakdown ────────────────────────
        dist_header = QLabel("Agent Distribution")
        dist_header.setFont(QFont("Segoe UI", 9))
        dist_header.setStyleSheet("color: #8b949e; margin-top: 10px;")
        layout.addWidget(dist_header)

        self._dist_labels: dict[str, QLabel] = {}
        
        # Pre-allocate labels for all known regimes to prevent resizing
        keys = list(_REGIME_COLORS.keys())
        keys.remove("UNKNOWN")
        
        for k in keys:
            row = QWidget()
            rlayout = QHBoxLayout(row)
            rlayout.setContentsMargins(0, 0, 0, 0)
            
            name_lbl = QLabel(k)
            name_lbl.setFont(QFont("Consolas", 9))
            name_lbl.setStyleSheet(f"color: {_REGIME_COLORS[k]};")
            
            val_lbl = QLabel("0")
            val_lbl.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            rlayout.addWidget(name_lbl)
            rlayout.addStretch()
            rlayout.addWidget(val_lbl)
            
            layout.addWidget(row)
            self._dist_labels[k] = val_lbl

        layout.addStretch()

    def update_frame(self, frame: TelemetryFrame) -> None:
        """Process counts and smooth the global regime classification."""
        if not frame.regime_state:
            return

        cnt = collections.Counter(frame.regime_state.values())
        
        # ── 1. Update Distribution text ───────────────────────
        for k, lbl in self._dist_labels.items():
            lbl.setText(str(cnt.get(k, 0)))

        # ── 2. Compute Dominant Smoothed Regime ───────────────
        # Exclude DEAD from global mode
        alive_cnt = cnt.copy()
        if "DEAD" in alive_cnt:
            del alive_cnt["DEAD"]
            
        current_dom = "UNKNOWN"
        if alive_cnt:
            current_dom = alive_cnt.most_common(1)[0][0]
            
        self._history.append(current_dom)
        
        smoothed_dom = collections.Counter(self._history).most_common(1)[0][0]
        
        # ── 3. Visual Update ──────────────────────────────────
        color = _REGIME_COLORS.get(smoothed_dom, "#8b949e")
        self._global_label.setText(smoothed_dom)
        self._global_label.setStyleSheet(
            f"color: {color}; background-color: #21262d; border-radius: 4px; padding: 10px;"
        )
