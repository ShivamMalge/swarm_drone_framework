"""
Phase 1 Demo — Deterministic Core Backbone.

Initialises N=10 agents, runs event-driven simulation for T=50,
and prints energy traces, position summaries, and communication statistics.

RNG REPRODUCIBILITY: uses SeedSequence-derived streams.
Running this script twice with the same seed guarantees identical output.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import SimConfig
from src.simulation import Phase1Simulation


def main() -> None:
    config = SimConfig(
        num_agents=10,
        grid_width=50.0,
        grid_height=50.0,
        comm_radius=15.0,
        p_drop=0.15,
        psi_max=0.1,
        energy_initial=50.0,
        dt=1.0,
        v_max=1.5,
        max_time=50.0,
        seed=42,
    )

    print("=" * 60)
    print("  Phase 1: Deterministic Core Backbone Demo")
    print("=" * 60)
    print(f"  Agents: {config.num_agents}")
    print(f"  Grid:   {config.grid_width} x {config.grid_height}")
    print(f"  Comm R: {config.comm_radius}, p_drop: {config.p_drop}")
    print(f"  Energy: {config.energy_initial}, p_move: {config.p_move}")
    print(f"  T_max:  {config.max_time}")
    print(f"  Seed:   {config.seed}")
    print(f"  RNG:    SeedSequence-derived independent streams")
    print("=" * 60)

    sim = Phase1Simulation(config)
    events_dispatched = sim.run()

    print(f"\n  Events dispatched: {events_dispatched}")
    print(f"  Final simulation time: {sim.kernel.now:.2f}")

    # Print summary
    summary = sim.summary()
    print(f"\n  Alive agents: {summary['alive_agents']}/{summary['total_agents']}")
    print(f"  Dead agents:  {summary['dead_agents']}")
    print(f"  Total energy remaining: {summary['total_energy_remaining']:.2f}")
    print(f"\n  Comm sent:      {summary['comm_sent']}")
    print(f"  Comm dropped:   {summary['comm_dropped']}")
    print(f"  Comm delivered: {summary['comm_delivered']}")
    print(f"  Drop rate:      {summary['drop_rate']:.3f}")

    # Print per-agent energy
    print(f"\n  {'Agent':>6} {'Energy':>10} {'Alive':>6}")
    print(f"  {'-'*6} {'-'*10} {'-'*6}")
    for agent in sim.agents:
        print(f"  {agent.agent_id:>6} {agent.energy:>10.2f} {'YES' if agent.is_alive else 'DEAD':>6}")

    print("\n" + "=" * 60)
    print(f"  Re-run with seed={config.seed} to verify identical output.")
    print("=" * 60)


if __name__ == "__main__":
    main()
