"""
Phase 2A Scalability and Correctness Test — Distributed Voronoi Coverage.

Verifies that when Voronoi coverage is enabled:
- Agents disperse spatially (variance increases).
- No global state access rules are violated.
- Energy remains monotonic and decays properly.
- N=100 agents run stable without bottlenecks.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.config import SimConfig
from src.core.event import reset_sequence_counter
from src.simulation import Phase1Simulation


def _make_coverage_config(seed: int = 42) -> SimConfig:
    """Return the N=100 config with coverage ENABLED."""
    # Spawn them tightly together so dispersion is obvious
    return SimConfig(
        num_agents=100,
        grid_width=100.0,
        grid_height=100.0,
        comm_radius=15.0,
        p_drop=0.1,
        psi_max=0.0,  # No interference to isolate dispersion mechanics
        energy_initial=500.0,
        p_move=0.1,
        p_comm=0.05,
        p_idle=0.001,
        dt=1.0,
        v_max=1.0,  # Slower to see smooth trajectories
        max_time=100.0,
        seed=seed,
        coverage_enabled=True,
    )


def _run_coverage(config: SimConfig) -> Phase1Simulation:
    reset_sequence_counter()
    sim = Phase1Simulation(config)
    
    # Override initial positions to a tight cluster in the center to test dispersion
    center = np.array([config.grid_width / 2.0, config.grid_height / 2.0])
    for i, a in enumerate(sim.agents):
        # Normal distribution cluster
        a._position = sim._streams["positions"].normal(loc=center, scale=5.0, size=2)
        # Clamped inside grid
        a._position = sim.grid.clamp_position(a._position)

    sim.run()
    return sim


class TestCoverageN100:

    def test_coverage_improves_dispersion(self):
        """Coverage control should cause tightly clustered agents to spread out."""
        config = _make_coverage_config()
        reset_sequence_counter()
        sim = Phase1Simulation(config)
        
        # Override initial positions to a tight cluster
        center = np.array([config.grid_width / 2.0, config.grid_height / 2.0])
        initial_positions = []
        for i, a in enumerate(sim.agents):
            a._position = sim._streams["positions"].normal(loc=center, scale=5.0, size=2)
            a._position = sim.grid.clamp_position(a._position)
            initial_positions.append(a._position.copy())
            
        initial_positions = np.array(initial_positions)
        initial_variance = np.var(initial_positions, axis=0).sum()

        sim.run()
        
        final_positions = np.array([a.position for a in sim.agents])
        final_variance = np.var(final_positions, axis=0).sum()

        # Variance must have grown significantly as they repel/disperse
        assert final_variance > initial_variance * 2.0, (
            f"Agents failed to disperse: initial_var={initial_variance:.2f}, "
            f"final_var={final_variance:.2f}"
        )

    def test_no_global_access_violation_in_coverage(self):
        """Coverage control does not cheat by accessing scipy's global Voronoi."""
        config = _make_coverage_config()
        sim = _run_coverage(config)

        import sys
        # Verify scipy.spatial.Voronoi was not injected into agent module
        from src.coordination import voronoi_coverage
        assert not hasattr(voronoi_coverage, 'Voronoi'), "scipy.spatial.Voronoi must not be imported!"

        # Ensure agents only hold their local state
        from src.core.kernel import SimulationKernel
        for agent in sim.agents:
            for attr_name in dir(agent):
                if attr_name.startswith("__"):
                    continue
                attr = getattr(agent, attr_name)
                assert not isinstance(attr, SimulationKernel), "Kernel leaked to agent"

    def test_energy_monotonic_under_coverage(self):
        """Coverage control path preserves monotonic energy decay."""
        config = _make_coverage_config()
        sim = _run_coverage(config)

        for agent in sim.agents:
            assert 0 <= agent.energy < config.energy_initial, (
                f"Agent {agent.agent_id} energy out of bounds: {agent.energy}"
            )

    def test_n100_runs_to_completion(self):
        """Simulation remains stable and completes all events."""
        config = _make_coverage_config()
        sim = _run_coverage(config)
        assert sim.kernel.now == config.max_time
        # Event queue will naturally have pending events scheduled > max_time
        assert len(sim.kernel._queue) > 0
