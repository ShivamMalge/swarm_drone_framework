"""
ReplayController — Qt bridge between ReplayEngine and GUI panels.

Emits frame_ready(TelemetryFrame) at a rate governed by the engine's
speed multiplier. Completely decoupled from SimulationWorker.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from src.replay.replay_engine import ReplayEngine, PlaybackState
from src.telemetry.telemetry_frame import TelemetryFrame


class ReplayController(QObject):
    """
    Bridges ReplayEngine → GUI via QTimer polling.
    No SimulationWorker interaction.
    """

    frame_ready = Signal(object)  # TelemetryFrame
    playback_finished = Signal()

    def __init__(
        self,
        engine: ReplayEngine,
        base_interval_ms: int = 16,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._base_interval = base_interval_ms

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)

    # ── Control ──────────────────────────────────────────────

    def start(self) -> None:
        self._engine.play()
        interval = max(1, int(self._base_interval / self._engine.speed))
        self._timer.start(interval)

    def pause(self) -> None:
        self._engine.pause()
        self._timer.stop()

    def stop(self) -> None:
        self._engine.stop()
        self._timer.stop()

    def set_speed(self, multiplier: float) -> None:
        self._engine.set_speed(multiplier)
        if self._timer.isActive():
            interval = max(1, int(self._base_interval / self._engine.speed))
            self._timer.setInterval(interval)

    def seek(self, frame_index: int) -> None:
        self._engine.seek(frame_index)
        frame = self._engine.current_frame()
        if frame is not None:
            self.frame_ready.emit(frame)

    # ── Internal ─────────────────────────────────────────────

    def _tick(self) -> None:
        frame = self._engine.next_frame()
        if frame is not None:
            self.frame_ready.emit(frame)
        else:
            self._timer.stop()
            self.playback_finished.emit()
