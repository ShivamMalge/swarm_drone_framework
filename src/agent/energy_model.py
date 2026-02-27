"""
Agent energy model — strictly monotonically decreasing.

Energy is consumed ONLY through scheduled events:
  - KINEMATIC_UPDATE → movement cost
  - MSG_TRANSMIT → communication cost
  - ENERGY_UPDATE → idle cost

Energy can never increase. At E ≤ 0 the agent is permanently dead.
"""

from __future__ import annotations


class EnergyModel:
    """
    Tracks an agent's energy with strict monotonic decay.

    Parameters
    ----------
    initial_energy : float
        Starting energy E_i(0).
    """

    def __init__(self, initial_energy: float) -> None:
        if initial_energy < 0:
            raise ValueError("Initial energy must be non-negative.")
        self._energy: float = initial_energy
        self._alive: bool = True

    @property
    def energy(self) -> float:
        """Current energy level."""
        return self._energy

    @property
    def is_alive(self) -> bool:
        """True if the agent has not been permanently deactivated."""
        return self._alive

    def consume(self, amount: float) -> float:
        """
        Consume *amount* of energy.

        - If consumption would drive energy below zero, energy is
          clamped to 0 and the agent is marked permanently dead.
        - Returns the actual amount consumed.

        Raises
        ------
        RuntimeError
            If the agent is already dead.
        ValueError
            If *amount* is negative (energy can never increase).
        """
        if not self._alive:
            raise RuntimeError("Cannot consume energy: agent is dead.")
        if amount < 0:
            raise ValueError(
                f"Energy consumption must be non-negative, got {amount}"
            )

        actual = min(amount, self._energy)
        self._energy -= actual
        if self._energy <= 0.0:
            self._energy = 0.0
            self._alive = False
        return actual
