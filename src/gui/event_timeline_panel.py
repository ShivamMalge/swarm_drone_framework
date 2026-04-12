"""
event_timeline_panel.py — Horizontal timeline with colour-coded event markers.
"""

from __future__ import annotations

from collections import deque

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.analytics.system_event import SystemEvent

# Severity → colour mapping
_COLORS = {
    "collapse":        (248, 81, 73),     # red
    "instability":     (210, 153, 34),     # amber
    "cascade":         (255, 140, 0),      # orange
    "anomaly_spike":   (163, 113, 247),    # purple
    "health_critical": (248, 81, 73),      # red
    "recovery":        (63, 185, 80),      # green
}
_DEFAULT_COLOR = (88, 166, 255)  # blue fallback


class EventTimelinePanel(QFrame):
    """
    Renders a horizontal scatter-style timeline of SystemEvents.
    Bounded to 200 events maximum.
    """

    MAX_EVENTS = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._events: deque[SystemEvent] = deque(maxlen=self.MAX_EVENTS)
        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumHeight(100)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d;"
            " border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("Event Timeline (Fused)")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #e6edf3; border: none;")
        hdr.addWidget(title)

        self.lbl_count = QLabel("0 events")
        self.lbl_count.setFont(QFont("Consolas", 9))
        self.lbl_count.setStyleSheet("color: #8b949e; border: none;")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignRight)
        hdr.addWidget(self.lbl_count)
        layout.addLayout(hdr)

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(50)
        self.plot.setBackground(None)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("left")
        self.plot.getAxis("bottom").setStyle(showValues=True)
        self.plot.getAxis("bottom").setPen(pg.mkPen(color="#8b949e", width=1))
        self.plot.setMenuEnabled(False)
        self.plot.setYRange(-0.5, 1.5)

        self.scatter = pg.ScatterPlotItem(pxMode=True)
        self.plot.addItem(self.scatter)
        layout.addWidget(self.plot)

    # ── Public API ──────────────────────────────────────────

    def add_events(self, events: list[SystemEvent]) -> None:
        for e in events:
            self._events.append(e)
        self._redraw()

    def clear(self) -> None:
        self._events.clear()
        self.scatter.clear()
        self.lbl_count.setText("0 events")

    # ── Internal ────────────────────────────────────────────

    def _redraw(self) -> None:
        n = len(self._events)
        self.lbl_count.setText(f"{n} events")
        if n == 0:
            self.scatter.clear()
            return

        spots = []
        for e in self._events:
            r, g, b = _COLORS.get(e.type, _DEFAULT_COLOR)
            spots.append({
                "pos": (e.timestamp, e.severity),
                "size": 8 + int(e.severity * 6),
                "pen": pg.mkPen(None),
                "brush": pg.mkBrush(r, g, b, 200),
            })
        self.scatter.setData(spots)
