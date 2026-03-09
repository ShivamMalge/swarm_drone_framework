"""
Unit tests for RGGBuilder — connectivity within radius, no edges beyond.
"""

import numpy as np
from src.communication.rgg_builder import RGGBuilder


class TestRGGBuilder:

    def test_agents_within_radius_connected(self):
        builder = RGGBuilder(comm_radius=10.0)
        positions = np.array([
            [0.0, 0.0],
            [5.0, 0.0],   # within radius of 0
            [100.0, 100.0],  # far away
        ])
        neighbors = builder.build_neighbor_lists(positions)
        assert 1 in neighbors[0]
        assert 0 in neighbors[1]

    def test_agents_beyond_radius_not_connected(self):
        builder = RGGBuilder(comm_radius=10.0)
        positions = np.array([
            [0.0, 0.0],
            [100.0, 100.0],
        ])
        neighbors = builder.build_neighbor_lists(positions)
        assert 1 not in neighbors[0]
        assert 0 not in neighbors[1]

    def test_no_self_loops(self):
        builder = RGGBuilder(comm_radius=50.0)
        positions = np.array([
            [10.0, 10.0],
            [20.0, 20.0],
        ])
        neighbors = builder.build_neighbor_lists(positions)
        for i, nbrs in neighbors.items():
            assert i not in nbrs

    def test_symmetric_edges(self):
        builder = RGGBuilder(comm_radius=15.0)
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 50, size=(20, 2))
        neighbors = builder.build_neighbor_lists(positions)
        for i, nbrs in neighbors.items():
            for j in nbrs:
                assert i in neighbors[j], f"Edge ({i},{j}) is not symmetric"

    def test_no_persisted_state(self):
        """RGGBuilder must not persist adjacency between calls."""
        builder = RGGBuilder(comm_radius=10.0)
        pos1 = np.array([[0.0, 0.0], [5.0, 0.0]])
        pos2 = np.array([[0.0, 0.0], [100.0, 100.0]])

        n1 = builder.build_neighbor_lists(pos1)
        assert 1 in n1[0]

        n2 = builder.build_neighbor_lists(pos2)
        assert 1 not in n2[0]

        # Ensure the builder has no internal adjacency state
        assert not hasattr(builder, '_adjacency')
        assert not hasattr(builder, '_neighbors')
        assert not hasattr(builder, '_graph')
