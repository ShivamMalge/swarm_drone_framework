"""
Decentralized Voronoi Coverage Control.

This module implements a localized, numerical approximation of the Voronoi
centroid. Agents use ONLY their own position and the stale beliefs of their
neighbors' positions obtained from their LocalMap.

It strictly adheres to Phase 2 decentralized Fog-of-War rules:
- No global positions accessed
- No global Voronoi diagram computation (e.g., scipy.spatial.Voronoi on entire swarm)
- Computations are bounded strictly to the communication radius.
"""

import numpy as np

def compute_local_centroid(
    agent_pos: np.ndarray,
    neighbor_positions: list[np.ndarray],
    comm_radius: float,
    grid_res: int = 15
) -> np.ndarray:
    """
    Approximate the local Voronoi centroid using numerical spatial sampling.

    Generates a local grid around the agent bounded by comm_radius.
    Points closer to the agent than to any known neighbor belong to the local cell.
    The centroid is the geometric mean of these points.

    Parameters
    ----------
    agent_pos : np.ndarray
        The 2D position of the agent [x, y].
    neighbor_positions : list[np.ndarray]
        List of 2D positions of known neighbors.
    comm_radius : float
        The maximum sensing/communication boundary (limits the cell).
    grid_res : int
        Resolution of the local sampling grid along one axis (N x N points).

    Returns
    -------
    np.ndarray
        The estimated 2D centroid of the local Voronoi cell.
    """
    if not neighbor_positions:
        # Isolated agent's Voronoi cell is infinite, but practically bounded
        # by the communication radius. A symmetric circle means the centroid
        # remains exactly at the agent's current position.
        return agent_pos.copy()

    # Create a local grid around the agent
    xs = np.linspace(agent_pos[0] - comm_radius, agent_pos[0] + comm_radius, grid_res)
    ys = np.linspace(agent_pos[1] - comm_radius, agent_pos[1] + comm_radius, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    
    # Flatten grid points
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Filter 1: points must be within comm_radius limiting circle
    dist_to_agent = np.linalg.norm(points - agent_pos, axis=1)
    mask_circle = dist_to_agent <= comm_radius
    points = points[mask_circle]
    dist_to_agent = dist_to_agent[mask_circle]

    if len(points) == 0:
        return agent_pos.copy()

    # Filter 2: points must be closer to agent than ANY neighbor
    valid_mask = np.ones(len(points), dtype=bool)
    
    for n_pos in neighbor_positions:
        dist_to_neighbor = np.linalg.norm(points - n_pos, axis=1)
        # Point is dominated by neighbor if neighbor is strictly closer
        valid_mask &= (dist_to_agent <= dist_to_neighbor)

    # The local Voronoi cell points
    cell_points = points[valid_mask]

    if len(cell_points) == 0:
        return agent_pos.copy()

    # The centroid is the mean of the cell points
    centroid = np.mean(cell_points, axis=0)
    return centroid
