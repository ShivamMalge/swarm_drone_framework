"""
Bernoulli packet drop sampler.

Evaluates per-edge drop probability incorporating baseline p_drop
and the local interference field ψ.

Dropped packets are SILENTLY discarded — no ACK, no NACK.
The sender is never informed of a drop.
"""

from __future__ import annotations

import numpy as np


class PacketDropSampler:
    """
    Samples per-edge packet drops.

    Parameters
    ----------
    base_p_drop : float
        Baseline drop probability (independent of interference).
    rng : np.random.Generator
        Seeded random generator for reproducibility.
    """

    def __init__(self, base_p_drop: float, rng: np.random.Generator) -> None:
        if not 0.0 <= base_p_drop <= 1.0:
            raise ValueError("base_p_drop must be in [0, 1].")
        self._base_p_drop = base_p_drop
        self._rng = rng

    def should_drop(self, psi_value: float) -> bool:
        """
        Determine whether a packet should be dropped.

        The effective drop probability is:
            p_eff = 1 - (1 - p_drop) * (1 - ψ)

        Parameters
        ----------
        psi_value : float
            Local interference field intensity at the sender's position.

        Returns
        -------
        bool
            True if the packet is dropped (silent discard).
        """
        p_survive = (1.0 - self._base_p_drop) * (1.0 - psi_value)
        return bool(self._rng.random() >= p_survive)
