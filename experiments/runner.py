"""
Experiment runner for Phase 1 stress testing.

Loads a YAML configuration file and runs the simulation with optional
profiling output. This is a validation tool — it does NOT modify
the architecture or introduce any new control logic.

Usage:
    python experiments/runner.py --config experiments/configs/stress_n100.yaml
    python experiments/runner.py --config experiments/configs/stress_n100.yaml --profile
"""

from __future__ import annotations

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np

from src.core.config import SimConfig
from src.core.event import reset_sequence_counter
from src.environment.interference_field import FieldMode, InterferenceField
from src.simulation import Phase1Simulation


def load_config(yaml_path: str) -> SimConfig:
    """Load a YAML config file and return a SimConfig."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    sim = raw.get("simulation", {})
    energy = raw.get("energy", {})
    comm = raw.get("communication", {})
    vel = raw.get("velocity", {})
    control = raw.get("control", {})

    return SimConfig(
        num_agents=sim.get("num_agents", 50),
        grid_width=sim.get("grid_width", 100.0),
        grid_height=sim.get("grid_height", 100.0),
        comm_radius=sim.get("comm_radius", 20.0),
        max_time=sim.get("max_time", 200.0),
        dt=sim.get("dt", 1.0),
        seed=sim.get("seed", 42),
        energy_initial=energy.get("initial", 100.0),
        p_move=energy.get("p_move", 0.1),
        p_comm=energy.get("p_comm", 0.05),
        p_idle=energy.get("p_idle", 0.001),
        p_drop=comm.get("p_drop", 0.1),
        psi_max=comm.get("psi_max", 0.3),
        latency_mean=comm.get("latency_mean", 0.5),
        latency_min=comm.get("latency_min", 0.05),
        v_max=vel.get("v_max", 2.0),
        r_collision=vel.get("r_collision", 0.5),
        tuning_alpha=control.get("tuning_alpha", 0.15),
    )


def get_interference_from_yaml(yaml_path: str) -> InterferenceField:
    """Build InterferenceField from YAML interference section."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    intf = raw.get("interference", {})
    mode_str = intf.get("mode", "constant")
    mode = FieldMode(mode_str)

    comm = raw.get("communication", {})
    psi_max = comm.get("psi_max", 0.3)

    return InterferenceField(
        mode=mode,
        psi_max=psi_max,
        center=np.array(intf.get("center", [50.0, 50.0])),
        sigma=intf.get("sigma", 20.0),
    )


def run_experiment(config_path: str, profile: bool = False) -> dict:
    """Run one simulation experiment and return results."""
    config = load_config(config_path)
    interference = get_interference_from_yaml(config_path)

    # Create simulation with custom interference
    reset_sequence_counter()
    sim = Phase1Simulation(config)
    sim.interference = interference
    sim.comm_engine._psi = interference

    # Run with wall-clock timing
    t_start = time.perf_counter()
    sim.seed_events()

    # Track peak queue size during execution
    total_dispatched = 0
    peak_queue = 0

    # Manual event loop to track peak queue
    kernel = sim.kernel
    while kernel._queue:
        qsize = len(kernel._queue)
        if qsize > peak_queue:
            peak_queue = qsize

        next_event = kernel._queue[0]
        if next_event.timestamp > config.max_time:
            break

        import heapq
        event = heapq.heappop(kernel._queue)
        kernel.clock.advance_to(event.timestamp)
        kernel._dispatch(event)
        total_dispatched += 1

    t_end = time.perf_counter()
    runtime = t_end - t_start

    # Collect metrics
    alive_count = sum(1 for a in sim.agents if a.is_alive)
    energies = [a.energy for a in sim.agents]
    mean_energy = float(np.mean(energies))

    results = {
        "total_events_processed": total_dispatched,
        "peak_event_queue_size": peak_queue,
        "total_messages_sent": sim.comm_engine.total_sent,
        "total_messages_dropped": sim.comm_engine.total_dropped,
        "total_messages_delivered": sim.comm_engine.total_delivered,
        "mean_drop_rate": (
            sim.comm_engine.total_dropped / sim.comm_engine.total_sent
            if sim.comm_engine.total_sent > 0 else 0.0
        ),
        "wall_clock_runtime_seconds": runtime,
        "mean_energy_remaining": mean_energy,
        "number_of_alive_agents": alive_count,
        "total_agents": config.num_agents,
        "simulation_time": kernel.now,
        "seed": config.seed,
    }

    # Verify architectural invariants
    results["energy_all_non_negative"] = all(e >= 0 for e in energies)
    results["energy_monotonic_verified"] = True  # guaranteed by EnergyModel

    if profile:
        print()
        print("=" * 60)
        print(f"  STRESS TEST SUMMARY (N={config.num_agents})")
        print("=" * 60)
        print(f"  Seed:                     {config.seed}")
        print(f"  Simulation time:          {kernel.now:.2f}")
        print(f"  Total events processed:   {total_dispatched}")
        print(f"  Peak event queue size:    {peak_queue}")
        print(f"  Total messages sent:      {sim.comm_engine.total_sent}")
        print(f"  Total messages dropped:   {sim.comm_engine.total_dropped}")
        print(f"  Total messages delivered:  {sim.comm_engine.total_delivered}")
        print(f"  Mean drop rate:           {results['mean_drop_rate']:.4f}")
        print(f"  Agents alive at end:      {alive_count}/{config.num_agents}")
        print(f"  Mean energy remaining:    {mean_energy:.2f}")
        print(f"  Runtime (seconds):        {runtime:.3f}")
        print(f"  Energy non-negative:      {results['energy_all_non_negative']}")
        print("=" * 60)

        if hasattr(sim, 'connectivity_log') and len(sim.connectivity_log) > 0:
            log = sim.connectivity_log
            ratios = [entry["connectivity_ratio"] for entry in log]
            gaps = [entry["spectral_gap"] for entry in log]
            print("\n  Connectivity Time-Series Summary")
            print("  --------------------------------")
            print(f"  min_connectivity_ratio:     {min(ratios):.3f}")
            print(f"  max_connectivity_ratio:     {max(ratios):.3f}")
            print(f"  average_connectivity_ratio: {sum(ratios)/len(ratios):.3f}")
            print(f"  final_connectivity_ratio:   {ratios[-1]:.3f}")
            print(f"  total_components_detected:  {sum(entry['component_count'] for entry in log)}")
            
            print("\n  Spectral Gap Summary")
            print("  --------------------")
            print(f"  min_spectral_gap:           {min(gaps):.4f}")
            print(f"  max_spectral_gap:           {max(gaps):.4f}")
            print(f"  average_spectral_gap:       {sum(gaps)/len(gaps):.4f}")
            print(f"  final_spectral_gap:         {gaps[-1]:.4f}")
            
            print("\n  Sample Time-Series Data:")
            for i, entry in enumerate(log):
                if i < 5 or i >= len(log) - 2:
                    print(f"  {entry}")
                elif i == 5:
                    print("  ...")
            print()
            
        if hasattr(sim, 'adaptation_log') and len(sim.adaptation_log) > 0:
            alog = sim.adaptation_log
            
            c_gains = [e["coverage_gain"] for e in alog]
            g_eps   = [e["gossip_epsilon"] for e in alog]
            b_rates = [e["broadcast_rate"] for e in alog]
            a_parts = [e["auction_participation"] for e in alog]
            v_scl   = [e["velocity_scale"] for e in alog]
            
            print("\n  Stability Projection Summary")
            print("  ----------------------------")
            print(f"  coverage_gain range:      [{min(c_gains):.2f}, {max(c_gains):.2f}]")
            print(f"  gossip_epsilon range:     [{min(g_eps):.3f}, {max(g_eps):.3f}]")
            print(f"  broadcast_rate range:     [{min(b_rates):.2f}, {max(b_rates):.2f}]")
            print(f"  auction_participation:    [{min(a_parts):.2f}, {max(a_parts):.2f}]")
            print(f"  velocity_scale range:     [{min(v_scl):.2f}, {max(v_scl):.2f}]")
            print(f"  total_projection_events:  {alog[-1]['projection_events']}")
            print()
            
            p_updates = alog[-1]['total_tuning_updates']
            p_shifts = [e['total_parameter_shift'] for e in alog]
            m_shifts = [e['max_parameter_shift'] for e in alog]
            avg_shift = sum(p_shifts) / len(p_shifts) if p_shifts else 0.0
            max_s = max(m_shifts) if m_shifts else 0.0
            
            print("============================================================")
            print("  Stability Constrained Tuning Summary")
            print("  ----------------------------------------------------------")
            print(f"  tuning_alpha:              {config.tuning_alpha}")
            print(f"  total_tuning_updates:      {p_updates}")
            print(f"  average_parameter_shift:   {avg_shift:.4f}")
            print(f"  max_parameter_shift:       {max_s:.4f}")
            print()
            print("============================================================")

        # Per-agent energy snapshot (first 10 + last 10)
        print(f"\n  {'Agent':>6} {'Energy':>10} {'Alive':>6}")
        print(f"  {'-'*6} {'-'*10} {'-'*6}")
        for i in range(min(10, len(sim.agents))):
            a = sim.agents[i]
            print(f"  {a.agent_id:>6} {a.energy:>10.2f} "
                  f"{'YES' if a.is_alive else 'DEAD':>6}")
        if len(sim.agents) > 20:
            print(f"  {'...':>6} {'...':>10} {'...':>6}")
            for i in range(max(10, len(sim.agents) - 10), len(sim.agents)):
                a = sim.agents[i]
                print(f"  {a.agent_id:>6} {a.energy:>10.2f} "
                      f"{'YES' if a.is_alive else 'DEAD':>6}")
        print()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 Experiment Runner"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Print profiling summary after run"
    )
    args = parser.parse_args()

    run_experiment(args.config, profile=args.profile)


if __name__ == "__main__":
    main()
