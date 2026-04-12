"""
event_timeline.py — Phase 2V Event Visualization tracking system transitions
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget, QSizePolicy
)

from src.analytics.event_logger import SystemEvent

class EventTimeline(QFrame):
    """
    Renders timeline of system transitions safely displaying limited fixed windows.
    Automatically pushes recent events dropping oldest visually protecting UI threads.
    """

    def __init__(self, parent: QWidget | None = None, max_ui_events: int = 150) -> None:
        super().__init__(parent)
        self._max_ui_events = max_ui_events
        self._items: list[QWidget] = []
        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumHeight(120)
        self.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        # Header
        header = QLabel("Event Timeline")
        header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header.setStyleSheet("color: #e6edf3; border: none;")
        main_layout.addWidget(header)

        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background: transparent;")
        
        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.v_layout = QVBoxLayout(self.container)
        self.v_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.v_layout.setContentsMargins(0, 0, 0, 0)
        self.v_layout.setSpacing(4)
        
        self.scroll.setWidget(self.container)
        main_layout.addWidget(self.scroll)

    def add_event(self, event: SystemEvent) -> None:
        # Create item
        item = QWidget()
        item.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        item.setMinimumHeight(24)
        
        if event.severity == "CRITICAL":
            color = "#f85149" # Red
            bg = "#3a1d1e"
        elif event.severity == "WARNING":
            color = "#d29922" # Orange
            bg = "#3d2e15"
        elif event.severity == "RECOVERY":
            color = "#3fb950" # Green
            bg = "#1e3a23"
        else:
            color = "#58a6ff" # Blue Info
            bg = "#1f2a3a"
            
        item.setStyleSheet(f"background-color: {bg}; border-radius: 4px; border: 1px solid {color};")
        
        h_layout = QHBoxLayout(item)
        h_layout.setContentsMargins(6, 2, 6, 2)
        
        lbl_time = QLabel(f"[{event.timestamp:06.1f}s]")
        lbl_time.setFont(QFont("Consolas", 8))
        lbl_time.setStyleSheet(f"color: {color}; border: none;")
        h_layout.addWidget(lbl_time)
        
        lbl_type = QLabel(event.event_type)
        lbl_type.setFont(QFont("Segoe UI", 9))
        lbl_type.setStyleSheet(f"color: #e6edf3; border: none;")
        h_layout.addWidget(lbl_type)
        
        # Newest top
        self.v_layout.insertWidget(0, item)
        self._items.insert(0, item)
        
        # Maintain max items protecting UI
        while len(self._items) > self._max_ui_events:
            oldest = self._items.pop()
            self.v_layout.removeWidget(oldest)
            oldest.deleteLater()
