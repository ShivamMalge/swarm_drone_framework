"""
Local neighbor belief buffer — the agent's fog-of-war view.

Updated ONLY when a MSG_DELIVER event is processed.
All neighbor data is inherently stale (timestamped at send time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NeighborBelief:
    """Stale snapshot of a neighbor's state."""

    agent_id: int
    position: np.ndarray
    energy: float
    consensus_state: float
    timestamp: float  # time at which the neighbor sent the data


class LocalMap:
    """
    Maintains a dictionary of stale neighbor beliefs.

    All data is updated exclusively via incoming message deliveries.
    There is no mechanism to query "current" neighbor states.
    """

    def __init__(self) -> None:
        self._beliefs: dict[int, NeighborBelief] = {}
        
        # Phase 2C: Active auctions buffer
        # Map: task_id -> (highest_bid_value, winning_agent_id)
        self.active_auctions: dict[str, tuple[float, int]] = {}

    def update_neighbor(
        self,
        agent_id: int,
        position: np.ndarray,
        energy: float,
        consensus_state: float,
        timestamp: float,
    ) -> None:
        """
        Insert or update a neighbor's belief.

        Only accepts data that is newer than what is currently held.
        """
        existing = self._beliefs.get(agent_id)
        if existing is None or timestamp > existing.timestamp:
            self._beliefs[agent_id] = NeighborBelief(
                agent_id=agent_id,
                position=position.copy(),
                energy=energy,
                consensus_state=consensus_state,
                timestamp=timestamp,
            )

    def get_neighbor(self, agent_id: int) -> NeighborBelief | None:
        """Return the belief for a specific neighbor, or None."""
        return self._beliefs.get(agent_id)

    def get_all_neighbors(self) -> list[NeighborBelief]:
        """Return all currently held neighbor beliefs."""
        return list(self._beliefs.values())

    def remove_neighbor(self, agent_id: int) -> None:
        """Remove a neighbor (e.g., on confirmed death)."""
        self._beliefs.pop(agent_id, None)

    @property
    def neighbor_count(self) -> int:
        """Number of known neighbors."""
        return len(self._beliefs)

    def clear(self) -> None:
        """Wipe all beliefs."""
        self._beliefs.clear()

    # Phase 2C
    def update_auction(self, task_id: str, bid_value: float, bidder_id: int) -> bool:
        """
        Updates the local best-known bid for a task.
        Returns True if the local belief was updated, False otherwise.
        """
        if task_id not in self.active_auctions:
            self.active_auctions[task_id] = (bid_value, bidder_id)
            return True
            
        current_best_bid, current_best_id = self.active_auctions[task_id]
        
        # Max bid wins
        if bid_value > current_best_bid:
            self.active_auctions[task_id] = (bid_value, bidder_id)
            return True
        # Tie-break on lowest agent_id
        elif bid_value == current_best_bid and bidder_id < current_best_id:
            self.active_auctions[task_id] = (bid_value, bidder_id)
            return True
            
        return False
