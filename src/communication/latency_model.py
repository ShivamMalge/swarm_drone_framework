"""
Exponential latency sampler.

Samples communication delay τ ~ Exp(1/mean) + τ_min.
"""

from __future__ import annotations

import numpy as np


class LatencyModel:
    """
    Samples per-message delivery latency.

    Parameters
    ----------
    mean : float
        Mean of the exponential distribution (1/rate).
    tau_min : float
        Hard minimum latency floor.
    rng : np.random.Generator
        Seeded random generator.
    """

    def __init__(
        self, mean: float, tau_min: float, rng: np.random.Generator
    ) -> None:
        if mean <= 0:
            raise ValueError("Latency mean must be positive.")
        if tau_min < 0:
            raise ValueError("Minimum latency must be non-negative.")
        self._mean = mean
        self._tau_min = tau_min
        self._rng = rng

    def sample(self) -> float:
        """
        Sample a latency value τ = Exp(1/mean) + τ_min.

        Returns
        -------
        float
            Sampled delivery delay (always > τ_min).
        """
        return float(self._rng.exponential(self._mean)) + self._tau_min
