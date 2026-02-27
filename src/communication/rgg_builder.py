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
