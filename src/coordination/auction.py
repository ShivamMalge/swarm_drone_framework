"""
Phase 2C: Distributed Auction Task Allocation.

Provides logic for computing bids and resolving winners locally.
Follows Fog-of-War constraints: no global state, localized decision making.

Bid convention — COST, minimised:
    bid_i(tau) = omega_d * ||p_i - p_tau|| + omega_e / E_i
Lower is better. The task reward does not appear in the bid: for a given task
every bidder sees the same reward, so it is a constant offset that cannot
change the winner; it participates only in feasibility. An earlier revision
scored bids as ``reward - distance`` (maximised), which is winner-equivalent
to distance minimisation and carried NO energy dependence at all, despite the
manuscript crediting energy-aware allocation (audit F-14).

The energy weight omega_e defaults to the nominal initial energy E_0 = 100,
so the penalty is 1 unit of distance at full charge and grows hyperbolically
as the agent drains (10 at E=10, 50 at E=2). This is a principled scale
choice, not a tuned one: it makes the two bid terms commensurate at full
charge and lets depletion dominate only when depletion is severe.
"""

from typing import Optional

import numpy as np

# Default energy weight: the nominal initial energy of an agent.
OMEGA_E_DEFAULT = 100.0


def compute_bid(
    agent_position: np.ndarray,
    agent_energy: float,
    task_position: np.ndarray,
    task_reward: float,
    p_move: float = 0.1,
    omega_e: float = OMEGA_E_DEFAULT,
) -> float:
    """
    Compute a COST bid for a task from local state. Lower is better.

    Feasibility: if the movement energy to reach the task
    (``p_move * distance``) meets or exceeds the agent's remaining energy,
    the agent must not bid — returns ``float('inf')``, which can never win
    under the min convention. (The earlier revision compared raw distance
    against energy, mixing metres with energy units, and returned ``-inf``
    for infeasible bids — which under a min convention would have WON.)

    Parameters
    ----------
    agent_position, task_position : np.ndarray
        2D positions.
    agent_energy : float
        Remaining energy E_i.
    task_reward : float
        Unused in the bid value (constant per task across bidders); retained
        so call sites document the full task tuple.
    p_move : float
        The agent's own movement energy cost per unit distance.
    omega_e : float
        Energy-margin weight (see module docstring).
    """
    dist = float(np.linalg.norm(task_position - agent_position))

    if agent_energy <= 0.0 or p_move * dist >= agent_energy:
        return float("inf")

    return dist + omega_e / agent_energy


def resolve_local_winner(task_id: str, local_map) -> Optional[int]:
    """
    Determine the winner for a task based entirely on the agent's LocalMap.
    Returns the agent_id of the winner, or None if no bids exist.
    """
    if task_id not in local_map.active_auctions:
        return None

    _bid, winner_id, _ts = local_map.active_auctions[task_id]
    return winner_id
