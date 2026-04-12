"""
ReplayEngine — Deterministic frame server with timeline navigation.

Completely decoupled from the DES kernel and SimulationWorker.
Operates exclusively on pre-loaded TelemetryFrame sequences.
"""

from __future__ import annotations

from enum import Enum, auto

from src.telemetry.telemetry_frame import TelemetryFrame


class PlaybackState(Enum):
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


class ReplayEngine:
    """
    Deterministic replay over an ordered frame sequence.

    Stepping is index-based (no interpolation by default).
    Speed multiplier affects only the external timer rate,
    not the frame data.
    """

    def __init__(self, frames: list[TelemetryFrame]) -> None:
        self._frames = frames
        self._index: int = 0
        self._speed: float = 1.0
        self._state = PlaybackState.STOPPED

    # ── Properties ───────────────────────────────────────────

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def index(self) -> int:
        return self._index

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def total_duration(self) -> float:
        if not self._frames:
            return 0.0
        return self._frames[-1].time - self._frames[0].time

    @property
    def current_time(self) -> float:
        if not self._frames or self._index >= len(self._frames):
            return 0.0
        return self._frames[self._index].time

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def is_finished(self) -> bool:
        return self._index >= len(self._frames)

    # ── Playback control ─────────────────────────────────────

    def play(self) -> None:
        if self._frames:
            self._state = PlaybackState.PLAYING

    def pause(self) -> None:
        self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        self._state = PlaybackState.STOPPED
        self._index = 0

    def set_speed(self, multiplier: float) -> None:
        self._speed = max(0.25, min(multiplier, 8.0))

    def seek(self, frame_index: int) -> None:
        """Jump to exact frame index."""
        self._index = max(0, min(frame_index, len(self._frames) - 1))

    def seek_time(self, t: float) -> None:
        """Jump to closest frame at or after time t."""
        for i, fr in enumerate(self._frames):
            if fr.time >= t:
                self._index = i
                return
        self._index = len(self._frames) - 1

    # ── Frame serving ────────────────────────────────────────

    def next_frame(self) -> TelemetryFrame | None:
        """
        Return the next frame if playing, advance index.
        Returns None if paused, stopped, or finished.
        """
        if self._state != PlaybackState.PLAYING:
            return None
        if self._index >= len(self._frames):
            self._state = PlaybackState.STOPPED
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def current_frame(self) -> TelemetryFrame | None:
        """Return frame at current index without advancing."""
        if not self._frames or self._index >= len(self._frames):
            return None
        return self._frames[self._index]

    # ── Event detection (derived from frame data only) ───────

    def detect_events(self) -> list[dict]:
        """
        Scan loaded frames for notable events.
        Returns list of {type, frame_index, time, details}.
        """
        events = []
        if len(self._frames) < 2:
            return events

        prev = self._frames[0]
        for i in range(1, len(self._frames)):
            curr = self._frames[i]

            # Agent deaths
            prev_alive = (~prev.drone_failure_flags).sum()
            curr_alive = (~curr.drone_failure_flags).sum()
            if curr_alive < prev_alive:
                events.append({
                    "type": "agent_death",
                    "frame_index": i,
                    "time": curr.time,
                    "details": f"alive: {int(prev_alive)} -> {int(curr_alive)}",
                })

            # Regime transitions (global dominant)
            if prev.regime_state and curr.regime_state:
                from collections import Counter
                prev_dom = Counter(prev.regime_state.values()).most_common(1)
                curr_dom = Counter(curr.regime_state.values()).most_common(1)
                if prev_dom and curr_dom and prev_dom[0][0] != curr_dom[0][0]:
                    events.append({
                        "type": "regime_transition",
                        "frame_index": i,
                        "time": curr.time,
                        "details": f"{prev_dom[0][0]} -> {curr_dom[0][0]}",
                    })

            # Connectivity collapse (spectral gap drops to ~0)
            if prev.spectral_gap > 0.01 and curr.spectral_gap <= 0.01:
                events.append({
                    "type": "connectivity_collapse",
                    "frame_index": i,
                    "time": curr.time,
                    "details": f"lambda2: {prev.spectral_gap:.4f} -> {curr.spectral_gap:.4f}",
                })

            prev = curr

        return events
