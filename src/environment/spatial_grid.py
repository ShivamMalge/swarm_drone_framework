"""
Bounded 2D spatial domain.

Provides boundary clamping for agent positions and validity checks.
"""

from __future__ import annotations

import numpy as np


class SpatialGrid:
    """
    Rectangular bounded 2D spatial domain [0, width] × [0, height].

    Parameters
    ----------
    width : float
        Domain width.
    height : float
        Domain height.
    """

    def __init__(self, width: float, height: float) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive.")
        self.width = width
        self.height = height

    def clamp_position(self, pos: np.ndarray) -> np.ndarray:
        """Clamp *pos* to lie within the domain boundaries."""
        return np.clip(pos, [0.0, 0.0], [self.width, self.height])

    def is_valid_position(self, pos: np.ndarray) -> bool:
        """Return True if *pos* is within domain boundaries."""
        return bool(
            0.0 <= pos[0] <= self.width and 0.0 <= pos[1] <= self.height
        )

    def random_positions(
        self, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate *n* uniformly distributed random positions.

        Returns
        -------
        np.ndarray
            Shape (n, 2).
        """
        xs = rng.uniform(0.0, self.width, size=n)
        ys = rng.uniform(0.0, self.height, size=n)
        return np.column_stack([xs, ys])
