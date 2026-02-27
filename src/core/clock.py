"""
Simulation clock — tracks the current simulation time.

The clock only advances forward; it never rewinds.
Only the SimulationKernel is permitted to call advance_to().
"""

from __future__ import annotations


class SimulationClock:
    """
    Monotonically advancing simulation clock.

    Attributes
    ----------
    _current_time : float
        The current simulation time (starts at 0.0).
    """

    def __init__(self) -> None:
        self._current_time: float = 0.0

    @property
    def now(self) -> float:
        """Return the current simulation time."""
        return self._current_time

    def advance_to(self, t: float) -> None:
        """
        Advance the clock to time *t*.

        Raises
        ------
        ValueError
            If *t* is strictly less than the current time (causality violation).
        """
        if t < self._current_time:
            raise ValueError(
                f"Causality violation: cannot rewind clock from "
                f"{self._current_time:.6f} to {t:.6f}"
            )
        self._current_time = t

    def reset(self) -> None:
        """Reset the clock to zero (for new episodes)."""
        self._current_time = 0.0
