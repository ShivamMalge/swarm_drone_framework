"""
Phase 4: Stability Projection (theta_safe).

Provides the mathematical projection operator that guarantees local agent tuning 
parameters strictly remain within safe analytic bounds to prevent swarm destabilization.
"""

from __future__ import annotations

# Conservative safe ranges protecting swarm dynamics
THETA_SAFE_BOUNDS = {
    "coverage_gain": (0.5, 2.0),
    "gossip_epsilon": (0.01, 0.05),
    "broadcast_rate": (0.2, 1.5),
    "auction_participation": (0.0, 1.0),
    "velocity_scale": (0.5, 1.5),
}

def project_to_theta_safe(theta_proposed: dict[str, float]) -> tuple[dict[str, float], int]:
    """
    Clamps proposed adaptive parameters into the safe analytic manifold.

    Parameters
    ----------
    theta_proposed : dict[str, float]
        The dictionary of proposed parameter values from the supervisor.

    Returns
    -------
    tuple[dict[str, float], int]
        (theta_safe, projection_events)
        - theta_safe is the sanitized dictionary holding clamped values.
        - projection_events is the total count of times a bound was enforced.
    """
    theta_safe = {}
    projection_events = 0

    for key, val in theta_proposed.items():
        if key in THETA_SAFE_BOUNDS:
            lower, upper = THETA_SAFE_BOUNDS[key]
            if val < lower:
                theta_safe[key] = float(lower)
                projection_events += 1
            elif val > upper:
                theta_safe[key] = float(upper)
                projection_events += 1
            else:
                theta_safe[key] = float(val)
        else:
            # Pass unconstrained parameters identically
            theta_safe[key] = val

    return theta_safe, projection_events
