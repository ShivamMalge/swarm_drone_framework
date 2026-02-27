"""
Decentralized Gossip Consensus Module.

Implements the standard Laplacian consensus update:
    x_i(t+1) = x_i(t) + epsilon * SUM(x_j - x_i)

Includes an architectural safeguard for the strictly decentralized environment:
Since agents cannot access the global adjacency matrix (and thus do not
know the actual degree bound or max eigenvalue), epsilon must be clamped locally
based on the agent's observable degree (number of neighbors in its LocalMap)
to guarantee stability: epsilon < 1 / d_i
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def compute_gossip_update(
    own_state: float,
    neighbor_states: list[float],
    base_epsilon: float
) -> float:
    """
    Compute the next state value using the discrete-time consensus protocol.
    
    If the base epsilon exceeds the local stability bound (1.0 / N_neighbors),
    it is automatically clamped to maintain asymptotic convergence properties.
    
    Parameters
    ----------
    own_state : float
        The agent's current state value x_i(t).
    neighbor_states : list[float]
        The state values of all known neighbors from the stale LocalMap.
    base_epsilon : float
        The user-configured step size.
        
    Returns
    -------
    float
        The updated consensus state x_i(t+1).
    """
    n_neighbors = len(neighbor_states)
    
    if n_neighbors == 0:
        return own_state
        
    # Local stability bound: epsilon must be strictly less than 1/d_i
    # We use 0.99 to give a tiny margin.
    safe_bound = 0.99 / n_neighbors
    
    actual_epsilon = base_epsilon
    if base_epsilon >= safe_bound:
        actual_epsilon = safe_bound

    # SUM(x_j - x_i)
    diff_sum = sum((x_j - own_state) for x_j in neighbor_states)
    
    new_state = own_state + actual_epsilon * diff_sum
    return new_state
