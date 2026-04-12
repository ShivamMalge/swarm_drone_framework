"""
TelemetryBridge — Qt signal bridge from TelemetryBuffer to GUI layer.

Pipeline position: DES → Worker → Buffer → Bridge → GUI

Runs on the GUI main thread via QTimer at ~60 Hz.
Polls the single-slot buffer; emits the latest frame as a Qt signal.
No kernel access. No blocking. No accumulation.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.telemetry_frame import TelemetryFrame


class TelemetryBridge(QObject):
    """
    Polls TelemetryBuffer at ~60 FPS and emits frame_ready(TelemetryFrame)
    for GUI consumption via Qt signal/slot.

    Parameters
    ----------
    telemetry_buffer : TelemetryBuffer
        Single-slot buffer written by SimulationWorker.
    poll_interval_ms : int
        Timer interval in milliseconds (default 16 ≈ 60 Hz).
    parent : QObject | None
        Qt parent for ownership.
    """

    frame_ready = Signal(object)

    def __init__(
        self,
        telemetry_buffer: TelemetryBuffer,
        poll_interval_ms: int = 16,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._buffer = telemetry_buffer
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.poll)
        self._poll_interval = poll_interval_ms

    # ── Polling ──────────────────────────────────────────────

    def poll(self) -> None:
        """Read latest frame from buffer; emit if new data exists."""
        if not self._buffer.has_new_data:
            return
        frame = self._buffer.get_latest()
        if frame is not None:
            self.frame_ready.emit(frame)

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        """Begin polling at the configured interval."""
        self._timer.start(self._poll_interval)

    def stop(self) -> None:
        """Stop polling."""
        self._timer.stop()

    @property
    def is_active(self) -> bool:
        return self._timer.isActive()
