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

    def should_drop(self, psi_value: float, distance: float, tx_radius: float) -> bool:
        """
        Determine whether a packet should be dropped using a non-linear path loss model.

        The effective survival probability combines baseline reliability, 
        exogenous interference (ψ), and inverse-square path loss:
            p_survive = (1 - p_drop_base) * (1 - ψ) * max(0, 1 - (d / R_tx)^2)

        Parameters
        ----------
        psi_value : float
            Local interference field intensity at the sender's position.
        distance : float
            Physical Euclidean distance between sender and receiver.
        tx_radius : float
            The sender's current transmission power radius.

        Returns
        -------
        bool
            True if the packet is dropped (silent discard).
        """
        if tx_radius <= 0.0 or distance > tx_radius:
            return True
            
        # Inverse-square path loss attenuation
        path_loss_factor = max(0.0, 1.0 - (distance / tx_radius)**2)
        
        p_survive = (1.0 - self._base_p_drop) * (1.0 - psi_value) * path_loss_factor
        return bool(self._rng.random() >= p_survive)
