"""
Phase 4: Stability Projection (theta_safe).

Provides the mathematical projection operator that guarantees local agent tuning 
parameters strictly remain within safe analytic bounds to prevent swarm destabilization.

Theoretical Justification (Hybrid Automata & Switched Systems):
---------------------------------------------------------------
1. State Constraints (Invariant Sets): The THETA_SAFE_BOUNDS define the physical and 
   mathematical boundaries required for subsystem stability (e.g., ensuring coverage 
   gains are strictly positive for convergence).
2. Gain Scheduling (Lipschitz Continuity): In `agent_core.py`, the `tuning_alpha` 
   EMA filter ensures that parameter transitions remain continuous and slow-varying. 
   This satisfies the requirements for a Common Lyapunov Function in switched systems, 
   preventing energy-spiking and chattering during regime transitions.
"""

from __future__ import annotations

# Conservative safe ranges protecting swarm dynamics
THETA_SAFE_BOUNDS = {
    "coverage_gain": (0.5, 2.0),
    "gossip_epsilon": (0.01, 0.05),
    "broadcast_rate": (0.2, 1.5),
    "auction_participation": (0.0, 1.0),
    "velocity_scale": (0.5, 1.5),
    "tx_power_scale": (1.0, 2.0),
}

def project_to_theta_safe(
    theta_proposed: dict[str, float],
    theta_nominal: dict[str, float],
    dynamic_bounds: dict[str, float] = None
) -> tuple[dict[str, float], int]:
    """
    Clamps proposed adaptive parameters into the safe analytic manifold using
    Box Clamping and Spectral Verification via Bisection Loop.

    Parameters
    ----------
    theta_proposed : dict[str, float]
        The dictionary of proposed parameter values from the supervisor.
    theta_nominal : dict[str, float]
        The nominal base parameters to bisect toward if proposed is unsafe.
    dynamic_bounds : dict[str, float], optional
        Locally computed dynamic upper bounds (e.g., delay-tolerant epsilon).

    Returns
    -------
    tuple[dict[str, float], int]
        (theta_safe, projection_events)
        - theta_safe is the sanitized dictionary holding clamped values.
        - projection_events is the total count of times a bound was enforced.
    """
    theta_safe = {}
    projection_events = 0
    dynamic_bounds = dynamic_bounds or {}

    # Step 1: Box Clamping & Pass-through
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
            theta_safe[key] = val

    # Step 2: Spectral Verification & Bisection Loop
    # Enforces discrete-time closed loop stability rho(A_cl) < 1 via dynamic local proxies.
    for key, bound in dynamic_bounds.items():
        if key in theta_safe and theta_safe[key] >= bound:
            # The proposed parameter violates the spectral bound.
            # Execute Bisection Loop toward nominal to find highest safe value.
            low = theta_nominal.get(key, 0.0)
            # Ensure the nominal value itself is safe. If not, clamp it.
            if low >= bound:
                low = bound * 0.99
                
            high = theta_safe[key]
            safe_val = low
            
            # 5-step Bisection Search
            for _ in range(5):
                mid = (low + high) / 2.0
                if mid < bound:
                    safe_val = mid
                    low = mid
                else:
                    high = mid
                    
            theta_safe[key] = safe_val
            projection_events += 1

    return theta_safe, projection_events
