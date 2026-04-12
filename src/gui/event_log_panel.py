"""
event_log_panel.py — Scrollable, bounded textual event log.
"""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QTextCharFormat
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QTextEdit, QVBoxLayout, QWidget,
)

from src.analytics.system_event import SystemEvent

_SEVERITY_COLORS = {
    "collapse":        "#f85149",
    "instability":     "#d29922",
    "cascade":         "#ff8c00",
    "anomaly_spike":   "#a371f7",
    "health_critical": "#f85149",
    "recovery":        "#3fb950",
}
_DEFAULT = "#8b949e"


class EventLogPanel(QFrame):
    """
    Auto-scrolling textual event log with severity colouring.
    Bounded to 100 visible entries; older entries are discarded.
    """

    MAX_LINES = 100

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lines: deque[str] = deque(maxlen=self.MAX_LINES)
        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumHeight(120)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d;"
            " border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        title = QLabel("Event Log")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #e6edf3; border: none;")
        hdr.addWidget(title)

        self.lbl_count = QLabel("0 entries")
        self.lbl_count.setFont(QFont("Consolas", 9))
        self.lbl_count.setStyleSheet("color: #8b949e; border: none;")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignRight)
        hdr.addWidget(self.lbl_count)
        layout.addLayout(hdr)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Consolas", 9))
        self.text.setStyleSheet(
            "QTextEdit { background-color: #0d1117; color: #c9d1d9;"
            " border: 1px solid #21262d; border-radius: 4px; }"
        )
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.text)

    # ── Public API ──────────────────────────────────────────

    def add_events(self, events: list[SystemEvent]) -> None:
        for e in events:
            color = _SEVERITY_COLORS.get(e.type, _DEFAULT)
            line = (
                f'<span style="color:{color}">'
                f"[{e.timestamp:7.2f}s] "
                f"<b>{e.type.upper()}</b> "
                f"sev={e.severity:.2f}  conf={e.confidence:.2f}  "
                f"src={e.source}"
                f"</span>"
            )
            self._lines.append(line)

        self._refresh()

    def clear(self) -> None:
        self._lines.clear()
        self.text.clear()
        self.lbl_count.setText("0 entries")

    # ── Internal ────────────────────────────────────────────

    def _refresh(self) -> None:
        self.lbl_count.setText(f"{len(self._lines)} entries")
        self.text.setHtml("<br>".join(self._lines))
        # Auto-scroll to bottom
        sb = self.text.verticalScrollBar()
        sb.setValue(sb.maximum())
