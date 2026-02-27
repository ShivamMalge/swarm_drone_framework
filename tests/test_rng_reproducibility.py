"""
RNG reproducibility tests.

Verifies that SeedSequence-derived streams guarantee
deterministic replay across simulation runs.
"""

import numpy as np

from src.core.config import SimConfig
from src.simulation import Phase1Simulation


def _run_sim(seed: int) -> dict:
    """Run a short simulation and return deterministic summary."""
    config = SimConfig(
        num_agents=5,
        grid_width=30.0,
        grid_height=30.0,
        comm_radius=12.0,
        p_drop=0.2,
        psi_max=0.1,
        energy_initial=30.0,
        dt=1.0,
        v_max=1.0,
        max_time=20.0,
        seed=seed,
    )
    sim = Phase1Simulation(config)
    sim.run()
    return {
        "positions": [a.position.tolist() for a in sim.agents],
        "energies": [a.energy for a in sim.agents],
        "alive": [a.is_alive for a in sim.agents],
        "comm_sent": sim.comm_engine.total_sent,
        "comm_dropped": sim.comm_engine.total_dropped,
        "comm_delivered": sim.comm_engine.total_delivered,
    }


class TestRNGReproducibility:

    def test_same_seed_identical_traces(self):
        """Running twice with the same seed produces identical results."""
        r1 = _run_sim(seed=42)
        r2 = _run_sim(seed=42)

        for i in range(5):
            np.testing.assert_array_almost_equal(
                r1["positions"][i], r2["positions"][i],
                err_msg=f"Agent {i} positions differ across runs with same seed"
            )
            assert r1["energies"][i] == r2["energies"][i], (
                f"Agent {i} energy differs"
            )
            assert r1["alive"][i] == r2["alive"][i]

        assert r1["comm_sent"] == r2["comm_sent"]
        assert r1["comm_dropped"] == r2["comm_dropped"]
        assert r1["comm_delivered"] == r2["comm_delivered"]

    def test_different_seed_different_traces(self):
        """Different seeds produce different trajectories."""
        r1 = _run_sim(seed=42)
        r2 = _run_sim(seed=99)

        # At least one agent must have a different final position
        any_diff = False
        for i in range(5):
            if not np.allclose(r1["positions"][i], r2["positions"][i]):
                any_diff = True
                break
        assert any_diff, "Different seeds produced identical traces"

    def test_agent_streams_independent(self):
        """Each agent's RNG stream is independent from communication RNG."""
        config = SimConfig(num_agents=3, seed=123)
        streams = config.spawn_rng_streams()

        # Draw from agent_0 and packet_drop — they must differ
        a0_vals = [streams["agent_0"].random() for _ in range(10)]
        pd_vals = [streams["packet_drop"].random() for _ in range(10)]

        assert a0_vals != pd_vals, "Agent and packet_drop streams are correlated"

    def test_agent_streams_mutually_independent(self):
        """Different agents have independent RNG streams."""
        config = SimConfig(num_agents=3, seed=456)
        streams = config.spawn_rng_streams()

        a0_vals = [streams["agent_0"].random() for _ in range(10)]
        a1_vals = [streams["agent_1"].random() for _ in range(10)]

        assert a0_vals != a1_vals, "Agent 0 and Agent 1 streams are correlated"

    def test_seed_sequence_spawns_correct_count(self):
        """spawn_rng_streams returns 4 subsystem + N agent streams."""
        config = SimConfig(num_agents=10, seed=0)
        streams = config.spawn_rng_streams()

        assert "positions" in streams
        assert "packet_drop" in streams
        assert "latency" in streams
        assert "task_spawner" in streams
        for i in range(10):
            assert f"agent_{i}" in streams
        assert len(streams) == 14  # 4 + 10
