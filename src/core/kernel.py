"""
Simulation Kernel — the sole global event dispatcher.

The kernel owns the min-heap event queue and simulation clock.
It is the ONLY component permitted to:
  - Advance simulation time
  - Pop and dispatch events
  - Mutate global simulation state (agent removal on death)

All modules interact with the kernel only by scheduling events.
Agents must NEVER hold a reference to this object.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any

from src.core.clock import SimulationClock
from src.core.event import Event, EventType, reset_sequence_counter


class SimulationKernel:
    """
    Min-heap discrete-event simulation kernel.

    Attributes
    ----------
    clock : SimulationClock
        The sole authoritative time source.
    _queue : list[Event]
        Min-heap of pending events.
    _handlers : dict[EventType, Callable]
        Registered callbacks per event type.
    _running : bool
        Guard flag to prevent re-entrance.
    """

    def __init__(self) -> None:
        self.clock = SimulationClock()
        self._queue: list[Event] = []
        self._handlers: dict[EventType, Callable[[Event], None]] = {}
        self._running: bool = False
        # Read-only instrumentation (not accessible by agents)
        self.peak_queue_size: int = 0

    # ── Event registration ──────────────────────────────────

    def register_handler(
        self, event_type: EventType, handler: Callable[[Event], None]
    ) -> None:
        """
        Register a callback for *event_type*.

        Parameters
        ----------
        event_type : EventType
            The category of event to handle.
        handler : Callable[[Event], None]
            Function invoked when an event of this type is dispatched.
        """
        self._handlers[event_type] = handler

    # ── Event scheduling ────────────────────────────────────

    def schedule_event(self, event: Event) -> None:
        """
        Insert *event* into the min-heap queue.

        Raises
        ------
        ValueError
            If the event is scheduled in the past.
        """
        if event.timestamp < self.clock.now:
            raise ValueError(
                f"Cannot schedule event in the past: "
                f"event.t={event.timestamp:.6f}, clock.now={self.clock.now:.6f}"
            )
        heapq.heappush(self._queue, event)

    def schedule_batch(self, events: list[Event]) -> None:
        """Schedule multiple events at once."""
        for event in events:
            self.schedule_event(event)

    # ── Main simulation loop ────────────────────────────────

    def run(self, until: float) -> int:
        """
        Execute events until the simulation clock reaches *until*
        or the event queue is empty.

        Returns
        -------
        int
            Total number of events dispatched.
        """
        if self._running:
            raise RuntimeError("Kernel.run() is not re-entrant.")
        self._running = True
        dispatched = 0

        try:
            while self._queue:
                # Peek at the next event without popping.
                next_event = self._queue[0]
                if next_event.timestamp > until:
                    break

                # Pop and dispatch.
                event = heapq.heappop(self._queue)
                self.clock.advance_to(event.timestamp)
                self._dispatch(event)
                dispatched += 1
        finally:
            self._running = False

        return dispatched

    def _dispatch(self, event: Event) -> None:
        """Route *event* to its registered handler."""
        handler = self._handlers.get(event.event_type)
        if handler is not None:
            handler(event)

    # ── Inspection helpers ──────────────────────────────────

    @property
    def pending_count(self) -> int:
        """Number of events remaining in the queue."""
        return len(self._queue)

    @property
    def now(self) -> float:
        """Current simulation time (shorthand)."""
        return self.clock.now

    # ── Reset ───────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the queue, reset clock and sequence counter."""
        self._queue.clear()
        self.clock.reset()
        self._running = False
        reset_sequence_counter()
