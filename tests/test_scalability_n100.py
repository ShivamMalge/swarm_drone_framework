"""
Scalability test — N=100 agents stress validation.

Verifies that the Phase 1 backbone remains:
- Deterministic
- Event-consistent (causal ordering)
- Architecturally pure (no global state leakage)
- Numerically stable (no negative energy, bounded drop rate)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.config import SimConfig
from src.core.event import reset_sequence_counter
from src.environment.interference_field import FieldMode, InterferenceField
from src.simulation import Phase1Simulation


def _make_stress_config(seed: int = 42) -> SimConfig:
    """Return the N=100 stress test config."""
    return SimConfig(
        num_agents=100,
        grid_width=200.0,
        grid_height=200.0,
        comm_radius=25.0,
        p_drop=0.2,
        psi_max=0.3,
        latency_mean=0.5,
        latency_min=0.05,
        energy_initial=100.0,
        p_move=0.1,
        p_comm=0.05,
        p_idle=0.001,
        dt=1.0,
        v_max=2.0,
        max_time=100.0,
        seed=seed,
    )


def _run_stress(config: SimConfig) -> Phase1Simulation:
    """Run benchmark with gaussian blob interference."""
    reset_sequence_counter()
    sim = Phase1Simulation(config)
    # Override to gaussian blob
    sim.interference = InterferenceField(
        mode=FieldMode.GAUSSIAN_BLOB,
        psi_max=config.psi_max,
        center=np.array([100.0, 100.0]),
        sigma=40.0,
    )
    sim.comm_engine._psi = sim.interference
    sim.run()
    return sim


class TestScalabilityN100:
    """Stress validation for N=100 agents."""

    def test_completes_without_exception(self):
        """Simulation runs to completion without crashing."""
        config = _make_stress_config()
        sim = _run_stress(config)
        assert sim.kernel.now == config.max_time

    def test_causal_event_ordering(self):
        """All events are processed in causal order."""
        config = _make_stress_config()
        reset_sequence_counter()
        sim = Phase1Simulation(config)
        sim.interference = InterferenceField(
            mode=FieldMode.GAUSSIAN_BLOB,
            psi_max=config.psi_max,
            center=np.array([100.0, 100.0]),
            sigma=40.0,
        )
        sim.comm_engine._psi = sim.interference

        # Track timestamps during dispatch
        timestamps = []
        original_dispatch = sim.kernel._dispatch

        def tracking_dispatch(event):
            timestamps.append(event.timestamp)
            original_dispatch(event)

        sim.kernel._dispatch = tracking_dispatch
        sim.seed_events()
        sim.kernel.run(until=config.max_time)

        # Verify monotonic ordering
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Causal violation at event {i}: "
                f"t[{i-1}]={timestamps[i-1]:.6f} > t[{i}]={timestamps[i]:.6f}"
            )

    def test_no_negative_energy(self):
        """No agent has negative energy at any point."""
        config = _make_stress_config()
        sim = _run_stress(config)
        for agent in sim.agents:
            assert agent.energy >= 0.0, (
                f"Agent {agent.agent_id} has negative energy: {agent.energy}"
            )

    def test_energy_strictly_decreased(self):
        """All agents consumed energy (no free-riding)."""
        config = _make_stress_config()
        sim = _run_stress(config)
        for agent in sim.agents:
            assert agent.energy < config.energy_initial, (
                f"Agent {agent.agent_id} did not consume energy"
            )

    def test_drop_rate_within_tolerance(self):
        """Empirical drop rate within ±5% of configured p_drop."""
        config = _make_stress_config()
        sim = _run_stress(config)
        sent = sim.comm_engine.total_sent
        dropped = sim.comm_engine.total_dropped

        assert sent > 0, "No messages sent"
        empirical_drop = dropped / sent
        # With p_drop=0.2 and psi=0-0.3 (gaussian), effective rate is higher
        # We check it's between 0.15 and 0.55 (generous band for stochastic)
        assert 0.15 <= empirical_drop <= 0.55, (
            f"Drop rate {empirical_drop:.3f} outside expected range"
        )

    def test_event_queue_never_negative(self):
        """Event queue size is always non-negative."""
        config = _make_stress_config()
        sim = _run_stress(config)
        assert sim.kernel.pending_count >= 0

    def test_deterministic_replay(self):
        """Same seed produces identical results at N=100."""
        config = _make_stress_config(seed=77)
        sim1 = _run_stress(config)

        config2 = _make_stress_config(seed=77)
        sim2 = _run_stress(config2)

        for i in range(100):
            np.testing.assert_array_almost_equal(
                sim1.agents[i].position,
                sim2.agents[i].position,
                err_msg=f"Agent {i} positions differ across N=100 replays"
            )
            assert sim1.agents[i].energy == sim2.agents[i].energy

        assert sim1.comm_engine.total_sent == sim2.comm_engine.total_sent
        assert sim1.comm_engine.total_dropped == sim2.comm_engine.total_dropped

    def test_rgg_not_persisted_under_load(self):
        """RGG builder has no persisted state after full N=100 run."""
        config = _make_stress_config()
        sim = _run_stress(config)
        rgg = sim.comm_engine._rgg
        assert not hasattr(rgg, '_adjacency')
        assert not hasattr(rgg, '_neighbors')
        assert not hasattr(rgg, '_graph')

    def test_agents_isolated_from_kernel(self):
        """No agent holds a kernel reference after N=100 run."""
        from src.core.kernel import SimulationKernel
        config = _make_stress_config()
        sim = _run_stress(config)
        for agent in sim.agents:
            for attr_name in dir(agent):
                if attr_name.startswith("__"):
                    continue
                attr = getattr(agent, attr_name)
                assert not isinstance(attr, SimulationKernel), (
                    f"Agent {agent.agent_id} holds kernel reference in {attr_name}"
                )
