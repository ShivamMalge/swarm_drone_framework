"""
Global simulation configuration.

All physical constants, network parameters, and energy coefficients
are defined here as a single frozen dataclass.

RNG Reproducibility
-------------------
A single ``seed`` spawns a ``SeedSequence``, which in turn spawns
independent child streams for each stochastic subsystem.  This
guarantees full deterministic replay across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimConfig:
    """
    Immutable simulation configuration.

    Parameters
    ----------
    num_agents : int
        Number of agents (N).
    grid_width : float
        Spatial domain width.
    grid_height : float
        Spatial domain height.
    comm_radius : float
        Communication radius R.
    p_drop : float
        Baseline packet drop probability.
    psi_max : float
        Maximum interference field intensity.
    latency_mean : float
        Mean of the exponential latency distribution (1/μ_τ).
    latency_min : float
        Hard minimum latency floor τ_min.
    energy_initial : float
        Initial energy per agent E_i(0).
    p_move : float
        Energy cost per unit distance moved.
    p_comm : float
        Energy cost per message transmitted.
    p_idle : float
        Energy cost per unit idle time.
    dt : float
        Time interval between scheduled kinematic update events.
    v_max : float
        Maximum velocity magnitude.
    r_collision : float
        Minimum allowed inter-agent distance.
    max_time : float
        Simulation horizon T_max.
    seed : int
        Global PRNG seed for reproducibility.
    """

    num_agents: int = 50
    grid_width: float = 100.0
    grid_height: float = 100.0
    comm_radius: float = 20.0
    p_drop: float = 0.1
    psi_max: float = 0.3
    latency_mean: float = 0.5
    latency_min: float = 0.05
    energy_initial: float = 100.0
    p_move: float = 0.1
    p_comm: float = 0.05
    p_idle: float = 0.001
    dt: float = 1.0
    v_max: float = 2.0
    r_collision: float = 0.5
    max_time: float = 200.0
    seed: int = 42

    def spawn_rng_streams(self) -> dict[str, np.random.Generator]:
        """
        Spawn independent RNG streams from a single SeedSequence.

        Returns a dictionary of named generators, each derived from
        the global seed and guaranteed to be statistically independent.

        Returns
        -------
        dict[str, np.random.Generator]
            Keys: 'positions', 'packet_drop', 'latency',
                  'task_spawner', 'agent_{i}' for i in [0, num_agents).
        """
        root = np.random.SeedSequence(self.seed)

        # Spawn 4 subsystem streams + N agent streams
        children = root.spawn(4 + self.num_agents)

        streams: dict[str, np.random.Generator] = {
            "positions": np.random.default_rng(children[0]),
            "packet_drop": np.random.default_rng(children[1]),
            "latency": np.random.default_rng(children[2]),
            "task_spawner": np.random.default_rng(children[3]),
        }

        for i in range(self.num_agents):
            streams[f"agent_{i}"] = np.random.default_rng(children[4 + i])

        return streams
