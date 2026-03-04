"""
Global Connectivity Telemetry (Phase 3D).

Passive observer layer exclusively accessed by the simulation kernel 
to log communication graph percolation states over time.
AGENTS MUST NEVER ACCESS THESE FUNCTIONS.
"""

from __future__ import annotations

from enum import Enum
import numpy as np
from scipy.spatial import KDTree


class ConnectivityPhase(Enum):
    CONNECTED = "CONNECTED"
    PARTIAL_CONNECTIVITY = "PARTIAL_CONNECTIVITY"
    FRAGMENTED = "FRAGMENTED"


def classify_connectivity_phase(ratio: float) -> ConnectivityPhase:
    """Classify the swarm's connectivity based on LCC fraction."""
    if ratio > 0.9:
        return ConnectivityPhase.CONNECTED
    elif ratio > 0.5:
        return ConnectivityPhase.PARTIAL_CONNECTIVITY
    else:
        return ConnectivityPhase.FRAGMENTED


def compute_largest_connected_component(
    agent_positions: np.ndarray, comm_radius: float
) -> tuple[int, int, float]:
    """
    Compute global graph connectivity metrics.
    
    Parameters
    ----------
    agent_positions : np.ndarray
        Shape (N, 2) array of all agent true positions.
    comm_radius : float
        The maximum distance for a viable communication edge.
        
    Returns
    -------
    tuple[int, int, float]
        (largest_component_size, component_count, connectivity_ratio)
    """
    n = len(agent_positions)
    if n == 0:
        return 0, 0, 0.0

    tree = KDTree(agent_positions)
    # Undirected set of edges
    pairs = tree.query_pairs(comm_radius)

    adj = {i: [] for i in range(n)}
    for u, v in pairs:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    components = []

    for i in range(n):
        if i not in visited:
            comp_size = 0
            queue = [i]
            visited.add(i)
            
            while queue:
                u = queue.pop(0)
                comp_size += 1
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
                        
            components.append(comp_size)

    largest_component_size = max(components) if components else 0
    component_count = len(components)
    connectivity_ratio = largest_component_size / n

    return largest_component_size, component_count, connectivity_ratio

def compute_connectivity_metrics(agent_positions: np.ndarray, comm_radius: float) -> dict:
    """Wrapper that returns connectivity metrics in dictionary format."""
    lcc, count, ratio = compute_largest_connected_component(agent_positions, comm_radius)
    return {
        "largest_component": lcc,
        "component_count": count,
        "connectivity_ratio": ratio
    }
