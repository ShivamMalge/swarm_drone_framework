"""
event_logger.py — Phase 2V Event Logger mapping system transitions chronologically.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, Signal


@dataclass
class SystemEvent:
    event_type: str
    timestamp: float
    module: str
    severity: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventLogger(QObject):
    """
    Observer-only chronological tracker maintaining fixed-size memory buffers,
    applying suppression logic mapping severity constraints implicitly.
    """

    event_logged = Signal(object)  # Emits SystemEvent

    def __init__(self, max_size: int = 1000, dedup_window: float = 1.0) -> None:
        super().__init__()
        self._max_size = max_size
        self._dedup_window = dedup_window
        
        self.events: collections.deque[SystemEvent] = collections.deque(maxlen=max_size)
        
        # Hash mapping mapping (event_type, module) -> timestamp
        self._last_occurrences: dict[tuple[str, str], float] = {}

    def log_event(self, event_type: str, module: str, timestamp: float, payload: dict | None = None) -> None:
        # Determine Severity
        severity = "INFO"
        et_lower = event_type.lower()
        if "collapse" in et_lower or "cascade" in et_lower and "recovered" not in et_lower:
            severity = "CRITICAL"
        elif "instability" in et_lower or "diverging" in et_lower or "stress" in et_lower or "oscillating" in et_lower or "stalled" in et_lower:
            severity = "WARNING"
        elif "recovered" in et_lower or "stabilized" in et_lower or "converged" in et_lower:
            severity = "RECOVERY"

        key = (event_type, module)
        last_t = self._last_occurrences.get(key, -1.0)
        
        # Deduplication temporal suppress window
        if last_t >= 0 and timestamp - last_t < self._dedup_window:
            return  # Suppressed

        # Record event
        self._last_occurrences[key] = timestamp

        event = SystemEvent(
            event_type=event_type,
            timestamp=timestamp,
            module=module,
            severity=severity,
            payload=payload or {}
        )
        
        self.events.append(event)
        self.event_logged.emit(event)
