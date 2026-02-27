"""
Phase 2C: Distributed Auction Task Allocation

Provides logic for computing bids and resolving winners locally.
Follows Fog-of-War constraints: no global state, localized decision making.
"""

from typing import Optional, Tuple
import numpy as np

def compute_bid(
    agent_position: np.ndarray,
    agent_energy: float,
    task_position: np.ndarray,
    task_reward: float
) -> float:
    """
    Compute a bid for a task based on local state.
    
    Cost to reach is Euclidean distance. Bid is reward - cost.
    If energy is insufficient to reach the task (plus a small buffer), 
    bid is fundamentally invalid (returns -inf).
    """
    dist = np.linalg.norm(task_position - agent_position)
    
    # If the agent doesn't even have enough energy to travel the distance, it shouldn't bid.
    # Buffer is optional, but taking cost = distance as a baseline (assuming P_move=... handled elsewhere)
    # For now, just a hard limit: if dist > energy, invalid.
    if dist > agent_energy:
        return float('-inf')
        
    return float(task_reward - dist)

def update_local_winner(
    current_best: Optional[Tuple[float, int]],
    incoming_bid_value: float,
    incoming_bidder_id: int
) -> Tuple[float, int]:
    """
    Returns the updated best bid tuple (highest_bid, winning_agent_id).
    Tie-breaking: lowest agent_id wins.
    """
    if current_best is None:
        return (incoming_bid_value, incoming_bidder_id)
        
    current_bid_value, current_bidder_id = current_best
    
    if incoming_bid_value > current_bid_value:
        return (incoming_bid_value, incoming_bidder_id)
    elif incoming_bid_value == current_bid_value and incoming_bidder_id < current_bidder_id:
        return (incoming_bid_value, incoming_bidder_id)
        
    return current_best

def resolve_local_winner(task_id: str, local_map) -> Optional[int]:
    """
    Determine the winner for a task based entirely on the agent's LocalMap.
    Returns the agent_id of the winner, or None if no bids exist.
    """
    if task_id not in local_map.active_auctions:
        return None
        
    _, winner_id = local_map.active_auctions[task_id]
    return winner_id
