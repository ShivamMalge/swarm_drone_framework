"""
Local Fog-of-War Proxy Metric Extractors.

These are pure functions that distill the Agent's local state and
its `LocalMap` into scalar inputs for the Regime Classifier.

CRITICAL: As per architecture constraints, these functions must NEVER
import, access, or attempt to derive global information, adjacency matrices,
or true spectral bounds.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.agent_core import AgentCore
    from src.agent.local_map import LocalMap


def compute_neighbor_density(local_map: LocalMap) -> int:
    """
    Return the raw number of currently believed neighbors.
    Stale neighbors inherently persist in LocalMap until drop heuristics clean them,
    so this is an imperfect proxy by design.
    """
    return len(local_map.get_all_neighbors())


def compute_information_staleness(local_map: LocalMap, current_time: float) -> float:
    """
    Return the mean delay (staleness) of information residing in the local map.
    A proxy for packet drop severity and network fracturing.
    """
    neighbors = local_map.get_all_neighbors()
    if not neighbors:
        # Heavily penalized if completely isolated to trigger FRAGMENTED heuristics
        return 999.0
        
    ages = [current_time - n.timestamp for n in neighbors]
    return statistics.mean(ages)


# Saturation ceiling for the variance proxy. The regime classifier only ever
# THRESHOLDS this value (variance_high ~ 1.5), so beyond the largest threshold
# its magnitude carries no decision information; the cap exists purely so a
# divergent consensus reads as "maximal variance" instead of overflowing.
VARIANCE_PROXY_CEILING = 1e12


def compute_local_consensus_variance(agent: AgentCore) -> float:
    """
    Return the variance of consensus states among the local neighborhood + self.
    A surrogate metric for tracking system-wide consensus convergence (λ₂).

    Saturating: divergent consensus states (possible in the unconstrained
    baseline, where no stability bound protects epsilon) are reported as
    VARIANCE_PROXY_CEILING rather than crashing the monitor. The previous
    implementation used ``statistics.variance``, whose exact-fraction
    arithmetic raises OverflowError once states pass ~1e154 -- the divergence
    is real baseline behaviour and must stay observable; the observer dying
    on it is the defect.
    """
    import numpy as np

    neighbors = agent.local_map.get_all_neighbors()
    if not neighbors:
        return 0.0  # Zero variance if alone

    states = [agent.consensus_state] + [n.consensus_state for n in neighbors]
    if len(states) < 2:
        return 0.0

    with np.errstate(over="ignore", invalid="ignore"):
        # ddof=1: sample variance, matching the statistics.variance convention.
        var = float(np.var(np.asarray(states, dtype=np.float64), ddof=1))

    if not np.isfinite(var):
        return VARIANCE_PROXY_CEILING
    return min(var, VARIANCE_PROXY_CEILING)

def compute_distributed_spectral_proxy(
    current_variance: float, 
    prev_variance: float, 
    epsilon: float, 
    dt: float
) -> float:
    """
    Phase 5: Distributed Estimator for Algebraic Connectivity (lambda_2).
    
    Rather than relying on global centralized calculations, the agent estimates 
    the Fiedler value locally using the convergence rate of the discrete-time 
    gossip consensus protocol.
    
    According to established literature, the state error decays proportionally 
    to (1 - epsilon * lambda_2). By calculating the variance decay ratio:
        r(t) = variance(t) / variance(t-1)
    The agent extracts the local spectral proxy:
        lambda_2_proxy = (1 - r(t)) / (epsilon * dt)
        
    This provides the mathematical difference equations required to justify
    regime detection without violating the Fog-of-War constraint.
    """
    if prev_variance <= 1e-6 or current_variance <= 1e-6:
        # If consensus is already reached (or perfectly isolated), return a high/max proxy
        return 1.0
        
    # Calculate decay ratio (bounded to prevent negative lambda estimates from noise)
    decay_ratio = current_variance / prev_variance
    decay_ratio = min(decay_ratio, 1.0)
    
    if epsilon <= 0.0:
        return 0.0
        
    # Extract estimated Fiedler value proxy
    lambda_2_est = (1.0 - decay_ratio) / (epsilon * dt)
    return float(lambda_2_est)
