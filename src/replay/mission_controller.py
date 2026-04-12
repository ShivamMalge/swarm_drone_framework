"""
mission_controller.py — Top-level playback integration enforcing determinism.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from src.replay.replay_engine import ReplayEngine
from src.telemetry.telemetry_frame import TelemetryFrame


class MissionController(QObject):
    """
    Wraps ReplayEngine with deterministic integration overrides properly
    synchronizing analytical buffers via explicit reset hooks safely dropping
    asynchronous drifts logically maintaining sequence identical to live telemetry.
    """

    frame_ready = Signal(object)
    playback_finished = Signal()
    reset_state = Signal() # GUI hook to recreate/clear all analyzers

    def __init__(self, engine: ReplayEngine, base_interval_ms: int = 16, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._base_interval = base_interval_ms

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._engine.play()
        self._update_timer_interval()

    def pause(self) -> None:
        self._engine.pause()
        self._timer.stop()

    def stop(self) -> None:
        self._engine.stop()
        self._timer.stop()
        self.reset_state.emit()
        
        self._engine.seek(0)
        frame = self._engine.current_frame()
        if frame is not None:
            self.frame_ready.emit(frame)

    def set_speed(self, multiplier: float) -> None:
        self._engine.set_speed(multiplier)
        if self._timer.isActive():
            self._update_timer_interval()

    def seek(self, frame_index: int) -> None:
        # Prevent drift and reset analyzer states to rely purely on current telemetry bounds
        self.reset_state.emit()
        
        self._engine.seek(frame_index)
        frame = self._engine.current_frame()
        if frame is not None:
            self.frame_ready.emit(frame)

    def _update_timer_interval(self) -> None:
        # Enforce max 60 FPS (16ms limits)
        interval = max(16, int(self._base_interval / self._engine.speed))
        self._timer.start(interval)

    def _tick(self) -> None:
        frame = self._engine.next_frame()
        if frame is not None:
            self.frame_ready.emit(frame)
        else:
            self._timer.stop()
            self.playback_finished.emit()
