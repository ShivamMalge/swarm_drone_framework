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
        # Map: task_id -> (best_bid_value, winning_agent_id, last_seen_time)
        # Bids are COSTS: the MINIMUM bid wins (see coordination/auction.py).
        # last_seen_time supports expiry: without it, unresolved auctions
        # accumulated forever and the random gossip pick had a ~4% chance of
        # broadcasting a bid that still mattered (audit F-13).
        self.active_auctions: dict[str, tuple[float, int, float]] = {}
        # Patch 2: Task Metadata (position)
        self.task_metadata: dict[str, np.ndarray] = {}

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

    def evict_stale_neighbors(self, current_time: float, max_age: float) -> int:
        """
        Drop beliefs whose last refresh is older than *max_age*.

        Returns the number of beliefs evicted. Without eviction a belief
        persisted forever once formed -- including beliefs about dead agents,
        which froze at their final position and kept feeding every consumer:
        neighbour density counted phantoms, staleness grew without bound
        (inflating tau_max and collapsing the epsilon bound), and the coverage
        centroid steered agents relative to corpses (audit F-12).
        """
        stale = [
            aid for aid, nb in self._beliefs.items()
            if current_time - nb.timestamp > max_age
        ]
        for aid in stale:
            del self._beliefs[aid]
        return len(stale)

    @property
    def neighbor_count(self) -> int:
        """Number of known neighbors."""
        return len(self._beliefs)

    def clear(self) -> None:
        """Wipe all beliefs."""
        self._beliefs.clear()

    # Phase 2C
    def update_auction(
        self, task_id: str, bid_value: float, bidder_id: int, timestamp: float
    ) -> bool:
        """
        Updates the local best-known bid for a task. Bids are costs: the
        MINIMUM bid wins, tie-broken on lowest agent_id. The stored
        last-seen time is always refreshed, so an auction stays alive in
        this map only while information about it keeps arriving.

        Returns True if the winning (bid, bidder) belief changed.
        """
        if task_id not in self.active_auctions:
            self.active_auctions[task_id] = (bid_value, bidder_id, timestamp)
            return True

        current_best_bid, current_best_id, seen = self.active_auctions[task_id]

        # Min bid (cost) wins
        if bid_value < current_best_bid:
            self.active_auctions[task_id] = (bid_value, bidder_id, timestamp)
            return True
        # Tie-break on lowest agent_id
        if bid_value == current_best_bid and bidder_id < current_best_id:
            self.active_auctions[task_id] = (bid_value, bidder_id, timestamp)
            return True

        # Belief unchanged, but refresh the last-seen time.
        self.active_auctions[task_id] = (
            current_best_bid, current_best_id, max(timestamp, seen),
        )
        return False

    def expire_auctions(self, current_time: float, max_age: float) -> int:
        """
        Drop auction entries not refreshed within *max_age*.

        Returns the number of entries removed. Without expiry the auction
        buffer grew monotonically (122+ dead entries by end of run) and
        random gossip selection was diluted to a ~4% chance of carrying a
        bid that still mattered (audit F-13).
        """
        stale = [
            tid for tid, (_b, _w, ts) in self.active_auctions.items()
            if current_time - ts > max_age
        ]
        for tid in stale:
            del self.active_auctions[tid]
        return len(stale)
