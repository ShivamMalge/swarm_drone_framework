"""
Exogenous adversarial interference field ψ(q, t).

The field is purely exogenous — it is independent of agent actions.
Agents do NOT have access to this field; only the Communication Engine
reads it to compute edge reliability.
"""

from __future__ import annotations

import enum

import numpy as np


class FieldMode(enum.Enum):
    """Supported interference field generation modes."""

    CONSTANT = "constant"
    GAUSSIAN_BLOB = "gaussian_blob"
    PULSE = "pulse"


class InterferenceField:
    """
    Generates the interference scalar ψ(q, t) ∈ [0, ψ_max].

    Parameters
    ----------
    mode : FieldMode
        Generation strategy.
    psi_max : float
        Upper bound on interference intensity.
    center : np.ndarray | None
        Spatial center for GAUSSIAN_BLOB mode.
    sigma : float
        Spatial spread for GAUSSIAN_BLOB mode.
    pulse_period : float
        Temporal period for PULSE mode.
    pulse_duty : float
        Duty cycle for PULSE mode (fraction of period that is active).
    """

    def __init__(
        self,
        mode: FieldMode = FieldMode.CONSTANT,
        psi_max: float = 0.3,
        center: np.ndarray | None = None,
        sigma: float = 20.0,
        pulse_period: float = 50.0,
        pulse_duty: float = 0.2,
    ) -> None:
        self.mode = mode
        self.psi_max = psi_max
        self.center = center if center is not None else np.array([50.0, 50.0])
        self.sigma = sigma
        self.pulse_period = pulse_period
        self.pulse_duty = pulse_duty

    def evaluate(self, position: np.ndarray, time: float) -> float:
        """
        Compute ψ(q, t) at the given position and time.

        ψ is an attenuation *fraction* and is clamped to [0, 1]. Callers use it
        as ``(1 - ψ)``, so an unclamped ψ > 1 makes the survival probability
        negative -- which happens to behave like total blackout, but only by
        accident, and it lets experiments ramp ψ to nonsense values (the
        percolation sweep reaches ψ = 2.55, i.e. "255% jamming"; audit F-39).
        Clamping makes total blackout explicit and bounds the quantity to its
        documented range.

        Returns
        -------
        float
            Interference intensity in [0, min(ψ_max, 1)].
        """
        if self.mode == FieldMode.CONSTANT:
            raw = self.psi_max

        elif self.mode == FieldMode.GAUSSIAN_BLOB:
            dist_sq = float(np.sum((position - self.center) ** 2))
            raw = self.psi_max * float(np.exp(-dist_sq / (2.0 * self.sigma**2)))

        elif self.mode == FieldMode.PULSE:
            phase = (time % self.pulse_period) / self.pulse_period
            raw = self.psi_max if phase < self.pulse_duty else 0.0

        else:
            raw = 0.0

        return float(min(max(raw, 0.0), 1.0))
