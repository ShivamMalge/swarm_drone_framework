"""
Random Geometric Graph (RGG) builder.

Computes neighbor lists from agent positions using KD-Tree spatial
indexing with radius R.

CRITICAL CONSTRAINTS:
  - Does NOT persist any adjacency matrix.
  - Does NOT expose global graph structure.
  - Returns neighbor lists scoped to a single communication round.
  - Only the Communication Engine calls this; agents never do.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


class RGGBuilder:
    """
    Builds a Random Geometric Graph (ephemeral, per-call).

    Parameters
    ----------
    comm_radius : float
        Maximum communication radius R.
    """

    def __init__(self, comm_radius: float) -> None:
        if comm_radius <= 0:
            raise ValueError("Communication radius must be positive.")
        self._radius = comm_radius

    # ── Per-sender spatial queries ──────────────────────────────────────
    #
    # `build_neighbor_lists` below assumes a single fixed radius for every
    # agent. That is not how this simulator transmits: each sender uses
    # ``comm_radius * tx_power_scale`` with ``tx_power_scale`` in [1.0, 2.0],
    # so a shared-radius query would silently discard dynamic transmission
    # power. These two methods provide the per-sender form instead.

    @staticmethod
    def build_tree(positions: np.ndarray) -> KDTree:
        """
        Build an ephemeral K-D Tree over agent positions.

        The caller owns the tree's lifetime and is responsible for rebuilding
        it whenever positions change. Nothing is cached here.
        """
        return KDTree(positions)

    @staticmethod
    def query_radius(tree: KDTree, point: np.ndarray, radius: float) -> list[int]:
        """
        Return indices of all points within *radius* of *point*.

        Results are returned in ascending index order. This matters: callers
        consume per-neighbour randomness (packet drop, latency), so a
        nondeterministic iteration order would change the RNG stream and break
        reproducible replay.

        The query radius is widened by one part in 1e12 so that points lying
        exactly on the boundary are always returned as candidates. Callers are
        expected to apply their own exact distance test; this method is a
        candidate filter, not the final predicate. Without the widening, a
        last-bit disagreement between the tree's internal metric and the
        caller's ``np.linalg.norm`` could silently drop a boundary edge.
        """
        idx = tree.query_ball_point(point, radius * (1.0 + 1e-12))
        idx.sort()
        return idx

    def build_neighbor_lists(
        self, positions: np.ndarray
    ) -> dict[int, list[int]]:
        """
        Compute neighbor lists from current positions.

        The result is ephemeral — it is NOT stored internally.
        There is no persistent adjacency matrix.

        Parameters
        ----------
        positions : np.ndarray
            Shape (N, 2) array of agent positions.

        Returns
        -------
        dict[int, list[int]]
            Mapping from agent index to list of neighbor indices
            within communication radius.
        """
        tree = KDTree(positions)
        pairs = tree.query_pairs(r=self._radius)

        n = positions.shape[0]
        neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
        for i, j in pairs:
            neighbors[i].append(j)
            neighbors[j].append(i)

        return neighbors
