"""
Stability-Constrained Tuning Layer (Phase 4B).

Provides a recursive exponential moving average (EMA) smoother that asymptotically
drives active parameters toward proposed analytic safe bounds.
This suppresses discrete jump destabilization inside feedback control loops.
"""

from __future__ import annotations


def smooth_update(current_value: float, target_value: float, alpha: float) -> float:
    """
    Applies first-order lag filter (EMA) to smooth parameter transitions.

    Parameters
    ----------
    current_value : float
        The currently active parameter value held by the agent.
    target_value : float
        The desired target value (post-theta-safe projection).
    alpha : float
        The tuning coefficient in [0.0, 1.0]. 
        1.0 means instantaneous jump (no smoothing), 0.0 means frozen state.

    Returns
    -------
    float
        The smoothed parameter for the next tick.
    """
    # Defensive bound checking for the meta-parameter
    alpha = max(0.0, min(1.0, alpha))
    return current_value + alpha * (target_value - current_value)
