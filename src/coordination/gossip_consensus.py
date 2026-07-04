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


import math

def compute_gossip_update(
    own_state: float,
    neighbor_states: list[float],
    neighbor_delays: list[float],
    dt: float,
    base_epsilon: float
) -> float:
    """
    Compute the next state value using a delay-tolerant discrete-time consensus protocol.
    
    Based on established literature for discrete-time multi-agent systems with time-varying 
    delays (e.g., Xiao & Wang 2006), asymptotic convergence requires the step size to be bounded by:
        0 < epsilon <= 1 / (d_max * (tau_max + 1))
        
    To preserve the strictly decentralized Fog-of-War constraint, this agent calculates a 
    conservative local bound using its observable out-degree (d_i) and locally observed 
    maximum discrete delay (tau_i_max).
    
    Parameters
    ----------
    own_state : float
        The agent's current state value x_i(t).
    neighbor_states : list[float]
        The state values of all known neighbors from the stale LocalMap.
    neighbor_delays : list[float]
        The observed continuous time delays for each neighbor's message.
    dt : float
        The simulation discrete time step duration.
    base_epsilon : float
        The user-configured step size proposed by the supervisor.
        
    Returns
    -------
    float
        The updated consensus state x_i(t+1).
    """
    n_neighbors = len(neighbor_states)
    
    if n_neighbors == 0:
        return own_state
        
    # Calculate local maximum discrete delay (tau_max in steps)
    max_continuous_delay = max(neighbor_delays) if neighbor_delays else 0.0
    tau_max_discrete = math.ceil(max_continuous_delay / dt)
    
    # Local delay-tolerant stability bound
    # We use 0.99 to provide a strict inequality margin.
    safe_bound = 0.99 / (n_neighbors * (tau_max_discrete + 1))
    
    actual_epsilon = base_epsilon
    if base_epsilon >= safe_bound:
        actual_epsilon = safe_bound

    # SUM(x_j - x_i)
    diff_sum = sum((x_j - own_state) for x_j in neighbor_states)
    
    new_state = own_state + actual_epsilon * diff_sum
    return new_state
