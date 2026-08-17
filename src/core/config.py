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

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RegimeConfig:
    """
    Tunable thresholds for regime detection (Passive Detection Layer).

    - dwell_time is the POLLING PERIOD of regime re-evaluation (the interval at
      which REGIME_UPDATE events are rescheduled), not a minimum-dwell
      constraint: nothing forbids a regime change on consecutive evaluations.
      A previous version of this docstring claimed Liberzon-style dwell-time
      enforcement with Zeno-prevention guarantees; no such enforcement exists
      anywhere in the implementation (audit MS-21).
    """
    window_size: int = 10
    dwell_time: float = 5.0
    neighbor_low: int = 5
    variance_high: float = 1.5
    energy_slope_critical: float = -0.8
    staleness_high: float = 3.0


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
    
    # Phase 1: Dynamic Transmission Power
    comm_radius_base: float = 20.0
    comm_radius: float = 20.0 # Keeping for backward compatibility temporarily
    
    # Phase 1: Temporal Anchor (1 tick = 0.1 physical seconds, 10Hz control loop)
    dt: float = 1.0
    
    p_drop: float = 0.1
    psi_max: float = 0.3
    latency_mean: float = 0.5
    latency_min: float = 0.05
    energy_initial: float = 100.0
    p_move: float = 0.1
    p_comm: float = 0.05 # Baseline quadratic energy coefficient
    p_idle: float = 0.001
    v_max: float = 2.0
    r_collision: float = 0.5
    max_time: float = 200.0
    seed: int = 42
    
    # Phase 2: Coverage Control
    coverage_enabled: bool = False

    # Phase 2B: Consensus
    consensus_epsilon: float = 0.02
    consensus_dt: float = 1.0

    # Phase 2C: Auction
    auction_timeout: float = 5.0
    r_task: float = 2.0  # Physical consumption radius

    # Phase 4B: Control Tuning
    tuning_alpha: float = 0.15

    # Phase 3.1: LocalMap belief eviction (audit F-12).
    # Beliefs unrefreshed for longer than this are dropped. Chosen from the
    # measured inter-refresh distribution of beliefs about LIVING neighbours
    # (seed 1000, N=100, 2000 ticks, n=28,620 refreshes): p50=2, p90=7,
    # p99=31, p99.9=76 ticks -- versus DEAD-sender belief ages of median 263,
    # p90=1513. 30.0 sits at the p99 of live-link cadence (~1% spurious
    # eviction risk) while removing phantoms two orders of magnitude sooner
    # than they previously persisted; it also exceeds the FRAGMENTED staleness
    # trigger (3 * staleness_high = 9.0), so staleness-based regime detection
    # still sees ages 9-30 before eviction removes the evidence.
    belief_max_age: float = 30.0

    # Phase 2D: Regime Detection
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    # Experiment & Logging
    test_mode: str | None = None
    log_dir: str = "logs"
    
    # Experiment 1: Percolation
    interference_growth_rate: float = 0.005 # Rate at which psi_max increases
    
    # Experiment 3: Stability
    theta_safe_enabled: bool = True
    stability_delay_max: float = 5.0 # Extra stochastic delay for stability test

    # True Oracle
    global_info_enabled: bool = False

    # How the oracle is billed for communication. The oracle receives global
    # state every tick; this selects what that awareness is charged for.
    #   "all_to_all"    -- billed for every other living agent (N-1 recipients).
    #                      Global awareness implies global bandwidth. Primary model.
    #   "per_neighbour" -- billed only for living agents inside comm_radius,
    #                      identical accounting to every decentralized arm.
    #                      Sensitivity variant: isolates coordination quality
    #                      from communication cost.
    # No effect unless global_info_enabled is True.
    oracle_comm_billing: str = "all_to_all"

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
