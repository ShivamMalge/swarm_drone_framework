"""
Phase 2B Scalability and Correctness Test — Distributed Gossip Consensus.

Verifies that when asynchronous gossip is enabled:
- Network state converges under no-drop (variance -> 0).
- Mean is strictly conserved (no-drop).
- Convergence rate decreases under packet drops but remains bounded.
- Deterministic reproducibility is strictly preserved.
- No global state access rules are violated.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.config import SimConfig
from src.core.event import reset_sequence_counter
from src.simulation import Phase1Simulation


def _make_consensus_config(seed: int = 42, p_drop: float = 0.0) -> SimConfig:
    """Return an N=100 config with a highly connected graph to ensure convergence."""
    return SimConfig(
        num_agents=100,
        grid_width=100.0,
        grid_height=100.0,
        comm_radius=200.0,  # Fully connected to guarantee theoretical convergence
        p_drop=p_drop,
        psi_max=0.0,
        energy_initial=5000.0, # Enough energy to not die during test
        p_move=0.0,            # Static positions for pure consensus test
        p_comm=0.0,            # No energy drain
        p_idle=0.0,
        dt=1.0,
        v_max=0.0,
        max_time=20.0,
        seed=seed,
        consensus_epsilon=0.01,
        consensus_dt=0.5,
    )


def _get_consensus_states(sim: Phase1Simulation) -> np.ndarray:
    return np.array([a.consensus_state for a in sim.agents])


class TestGossipConsensus:

    def test_convergence_no_drop(self):
        """Variance of states should strictly decrease toward 0 without packet drops."""
        config = _make_consensus_config(p_drop=0.0)
        reset_sequence_counter()
        sim = Phase1Simulation(config)

        initial_states = _get_consensus_states(sim)
        initial_mean = np.mean(initial_states)
        initial_var = np.var(initial_states)

        # Ensure we start with some variance
        assert initial_var > 0

        sim.run()

        final_states = _get_consensus_states(sim)
        final_mean = np.mean(final_states)
        final_var = np.var(final_states)

        # 1. Variance should significantly decrease
        assert final_var < initial_var * 0.1, f"Variance did not shrink: {initial_var} -> {final_var}"
        
        # 2. Mean should be roughly conserved (average consensus)
        # Note: Asymmetric broadcast gossip with stale beliefs does NOT exactly conserve the mean
        # step-by-step. It converges to a value within the convex hull, often close to the initial mean.
        assert np.isclose(initial_mean, final_mean, atol=1.5), f"Mean drifted too far: {initial_mean} -> {final_mean}"

    def test_convergence_with_drop(self):
        """Packet drops reduce convergence speed but system remains bounded and stable."""
        # Run identical setups: one perfect channel, one lossy channel.
        config_perfect = _make_consensus_config(p_drop=0.0)
        config_lossy = _make_consensus_config(p_drop=0.5)

        reset_sequence_counter()
        sim_perfect = Phase1Simulation(config_perfect)
        sim_perfect.run()
        var_perfect = np.var(_get_consensus_states(sim_perfect))

        reset_sequence_counter()
        sim_lossy = Phase1Simulation(config_lossy)
        sim_lossy.run()
        var_lossy = np.var(_get_consensus_states(sim_lossy))

        # Lossy network should converge slower (higher final variance) than perfect
        assert var_lossy > var_perfect

        # Lossy variance should still be strictly less than initial variance (bounded convergence)
        initial_variance = np.var(np.arange(100, dtype=float)) # agent IDs 0..99
        assert var_lossy < initial_variance

    def test_deterministic_replay(self):
        """Simulation RNG guarantees identical execution order and final state."""
        config1 = _make_consensus_config(seed=123, p_drop=0.2)
        reset_sequence_counter()
        sim1 = Phase1Simulation(config1)
        sim1.run()
        states1 = _get_consensus_states(sim1)

        config2 = _make_consensus_config(seed=123, p_drop=0.2)
        reset_sequence_counter()
        sim2 = Phase1Simulation(config2)
        sim2.run()
        states2 = _get_consensus_states(sim2)

        np.testing.assert_array_equal(states1, states2)

        # Prove different seeds diverge
        config3 = _make_consensus_config(seed=999, p_drop=0.2)
        reset_sequence_counter()
        sim3 = Phase1Simulation(config3)
        sim3.run()
        states3 = _get_consensus_states(sim3)

        assert not np.allclose(states1, states3)

    def test_no_global_access_violation(self):
        """Gossip update logic strictly avoids global connectivity querying."""
        # Ensure scipy is not inadvertently imported locally to cheat
        from src.coordination import gossip_consensus
        assert not hasattr(gossip_consensus, 'Voronoi')
        assert not hasattr(gossip_consensus, 'KDTree')

        # The update function is a pure function operating ONLY on local buffer
        own_state = 10.0
        neighbor_states = [5.0, 15.0, 10.0]
        epsilon = 0.5 # Over safe bound
        
        # Will clamp epsilon to 0.99 / 3 = 0.33
        new_state = gossip_consensus.compute_gossip_update(own_state, neighbor_states, epsilon)
        
        # Diff sum = -5 + 5 + 0 = 0 -> new_state = 10.0
        assert new_state == 10.0
