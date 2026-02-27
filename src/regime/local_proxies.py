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


def compute_local_consensus_variance(agent: AgentCore) -> float:
    """
    Return the variance of consensus states among the local neighborhood + self.
    A surrogate metric for tracking system-wide consensus convergence (λ₂).
    """
    neighbors = agent.local_map.get_all_neighbors()
    if not neighbors:
        return 0.0  # Zero variance if alone
        
    states = [agent.consensus_state] + [n.consensus_state for n in neighbors]
    if len(states) < 2:
        return 0.0
        
    return statistics.variance(states)
