"""
TelemetryBuffer — Single-slot latest-frame priority buffer.

Thread-safe, non-blocking, O(1) push/read data bridge between
the SimulationWorker (producer) and the GUI main thread (consumer).

Behavioural guarantees:
  - GUI always receives the freshest available frame.
  - Stale frames are overwritten immediately on push.
  - Simulation is NEVER blocked by GUI consumption speed.
  - Zero queue buildup; zero memory growth beyond one frame.
"""

from __future__ import annotations

import threading

from src.telemetry.telemetry_frame import TelemetryFrame


class TelemetryBuffer:
    """
    Single-slot overwrite buffer holding the most recent TelemetryFrame.

    Producer (SimulationWorker) calls ``push()``; consumer (GUI QTimer)
    calls ``get_latest()``.  Both are O(1) and non-blocking under a
    minimal ``threading.Lock``.
    """

    __slots__ = ("_lock", "_latest_frame", "_has_new_data")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: TelemetryFrame | None = None
        self._has_new_data: bool = False

    # ── Producer API (SimulationWorker thread) ───────────────

    def push(self, frame: TelemetryFrame) -> None:
        """
        Overwrite the stored frame with *frame*.

        - O(1) — single pointer swap under lock.
        - Never blocks the producer; the lock is held for
          two attribute assignments only.
        - Previous frame is discarded (eligible for GC).
        """
        with self._lock:
            self._latest_frame = frame
            self._has_new_data = True

    # ── Consumer API (GUI main thread) ───────────────────────

    def get_latest(self) -> TelemetryFrame | None:
        """
        Return the most recent frame, or ``None`` if nothing has
        been pushed yet.

        - O(1) read under lock.
        - Resets the ``has_new_data`` flag so callers can skip
          redundant redraws when the frame has not changed.
        """
        with self._lock:
            self._has_new_data = False
            return self._latest_frame

    # ── Optional helpers ─────────────────────────────────────

    @property
    def has_new_data(self) -> bool:
        """Check (without consuming) whether a new frame is available."""
        with self._lock:
            return self._has_new_data

    def clear(self) -> None:
        """Discard the stored frame and reset the flag."""
        with self._lock:
            self._latest_frame = None
            self._has_new_data = False
