"""
Architectural diagnostics audit.

Every check in this script must be CAPABLE OF FAILING. A previous revision
logged hardcoded True for three of ten checks ("Task Resolution", "Stability
Constraints", "Communication Overhead") and probed a fourth ("Fog-of-War")
for attributes no version of AgentCore ever had -- so the report printed
SUCCESS regardless of what the code did (audit F-16). Each check below
asserts on measured quantities, and the file documents, per check, what
would make it fail. Run with --self-test to deliberately break one invariant
and confirm the corresponding check reports FAILURE.
"""

import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np

sys.path.append(".")

from src.core.config import SimConfig
from src.simulation import Phase1Simulation
from src.regime.classifier import Regime
from src.adaptation.safety_projector import THETA_SAFE_BOUNDS


@dataclass
class AuditResult:
    check: str
    result: str
    notes: str


class ArchitecturalAudit:
    def __init__(self):
        self.results: List[AuditResult] = []

    def log_result(self, check: str, success: bool, notes: str):
        result_str = "SUCCESS" if success else "FAILURE"
        self.results.append(AuditResult(check, result_str, notes))
        print(f"[{result_str}] {check}: {notes}")

    # ── 1. Determinism ──────────────────────────────────────────────────
    def run_deterministic_replay_test(self):
        """Fails if: any agent position or energy differs between identical seeds."""
        print("\n--- 1. Deterministic Replay Test ---")
        config = SimConfig(num_agents=20, max_time=30.0, seed=42)

        sim1 = Phase1Simulation(config)
        sim1.run()
        pos1 = np.array([a.position for a in sim1.agents])
        e1 = np.array([a.energy for a in sim1.agents])

        sim2 = Phase1Simulation(config)
        sim2.run()
        pos2 = np.array([a.position for a in sim2.agents])
        e2 = np.array([a.energy for a in sim2.agents])

        ok = np.array_equal(pos1, pos2) and np.array_equal(e1, e2)
        self.log_result(
            "Deterministic Replay", ok,
            "Positions and energies bit-identical across identical seeds."
            if ok else
            f"DIVERGED: max|dpos|={np.abs(pos1 - pos2).max():.3e}, "
            f"max|dE|={np.abs(e1 - e2).max():.3e}",
        )

    # ── 2. Kernel integrity ─────────────────────────────────────────────
    def run_kernel_integrity_check(self):
        """Fails if: time does not advance, or the kernel is left running."""
        print("\n--- 2. Event Kernel Integrity ---")
        config = SimConfig(num_agents=10, max_time=10.0, seed=42)
        sim = Phase1Simulation(config)
        initial_time = sim.kernel.now
        dispatched = sim.run()
        ok = (sim.kernel.now > initial_time
              and sim.kernel._running is False
              and dispatched > 0)
        self.log_result(
            "Kernel Integrity", ok,
            f"{dispatched} events dispatched; clock advanced "
            f"{initial_time} -> {sim.kernel.now}; kernel quiescent.",
        )

    # ── 3. Fog of war ───────────────────────────────────────────────────
    def run_fog_of_war_test(self):
        """
        Fails if: the oracle channel (oracle_centroid, the one real global-
        information input to AgentCore) carries data while global_info_enabled
        is False, or the flag itself leaks on. A previous version probed for
        attributes ('kernel', 'all_agents', ...) that never existed on any
        AgentCore, and so could not fail.
        """
        print("\n--- 3. Fog-of-War Isolation ---")
        config = SimConfig(num_agents=10, max_time=20.0, seed=42,
                           coverage_enabled=True, global_info_enabled=False)
        sim = Phase1Simulation(config)
        sim.run()

        violations = []
        for agent in sim.agents:
            if agent.global_info_enabled:
                violations.append(f"agent {agent.agent_id}: flag leaked on")
            if agent.oracle_centroid is not None:
                violations.append(f"agent {agent.agent_id}: oracle_centroid set")
        self.log_result(
            "Fog-of-War", not violations,
            "Oracle channel closed on every agent with global_info_enabled=False."
            if not violations else "; ".join(violations),
        )

    # ── 4. Communication locality ───────────────────────────────────────
    def run_communication_locality_check(self):
        """
        Fails if: any packet is delivered between agents farther apart than
        the sender's transmission radius at send time. Instruments the drop
        sampler: every non-dropped packet's (distance, tx_radius) is recorded.
        A previous version measured nothing and logged True.
        """
        print("\n--- 4. Communication Locality ---")
        from src.communication.packet_drop import PacketDropSampler

        delivered_pairs: list[tuple[float, float]] = []
        orig = PacketDropSampler.should_drop

        def spy(self, psi_value, distance, tx_radius):
            dropped = orig(self, psi_value, distance, tx_radius)
            if not dropped:
                delivered_pairs.append((distance, tx_radius))
            return dropped

        PacketDropSampler.should_drop = spy
        try:
            sim = Phase1Simulation(SimConfig(
                num_agents=30, max_time=40.0, seed=42, coverage_enabled=True))
            sim.run()
        finally:
            PacketDropSampler.should_drop = orig

        beyond = [(d, r) for d, r in delivered_pairs if d > r]
        ok = len(delivered_pairs) > 0 and not beyond
        self.log_result(
            "Communication Locality", ok,
            f"{len(delivered_pairs)} deliveries, all within the sender's "
            f"tx_radius." if ok else
            (f"{len(beyond)} deliveries BEYOND tx_radius (worst: {max(beyond)})"
             if beyond else "no deliveries occurred -- cannot certify locality"),
        )

    # ── 5. Task resolution ──────────────────────────────────────────────
    def run_task_allocation_integrity(self):
        """
        Fails if: no auction is ever won, or any task is consumed at a
        distance beyond r_task. Consumption distances are measured by
        instrumenting the kinematic handler's deletion of active_tasks.
        A previous version ran the sim, discarded the result, and logged True.
        """
        print("\n--- 5. Task Allocation Integrity ---")
        config = SimConfig(num_agents=30, max_time=80.0, seed=42, r_task=2.0,
                           grid_width=40.0, grid_height=40.0, comm_radius=25.0,
                           energy_initial=500.0, coverage_enabled=True)
        sim = Phase1Simulation(config)

        consumption_distances: list[float] = []
        orig = Phase1Simulation._handle_kinematic_update

        def spy(self, event):
            agent = self.agents[event.agent_id]
            tid = agent.active_task_id
            pos_before = (self.active_tasks.get(tid).copy()
                          if tid in self.active_tasks else None)
            orig(self, event)
            if tid is not None and pos_before is not None and tid not in self.active_tasks:
                consumption_distances.append(
                    float(np.linalg.norm(agent.position - pos_before)))

        sim._handle_kinematic_update = spy.__get__(sim)
        from src.core.event import EventType
        sim.kernel.register_handler(
            EventType.KINEMATIC_UPDATE, sim._handle_kinematic_update)
        sim.run()

        wins = len(sim.auction_results)
        too_far = [d for d in consumption_distances if d > config.r_task]
        ok = wins > 0 and len(consumption_distances) > 0 and not too_far
        self.log_result(
            "Task Resolution", ok,
            f"{wins} auction wins; {len(consumption_distances)} tasks consumed, "
            f"all within r_task={config.r_task}." if ok else
            f"wins={wins}, consumed={len(consumption_distances)}, "
            f"beyond r_task={too_far}",
        )

    # ── 6. Regime threshold jitter ──────────────────────────────────────
    def run_regime_asynchrony_check(self):
        """Fails if: per-agent regime thresholds are not actually jittered."""
        print("\n--- 6. Regime Detection Asynchrony ---")
        sim = Phase1Simulation(SimConfig(num_agents=50, max_time=10.0, seed=42))
        unique_neighbors = set(a.regime_classifier.config.neighbor_low for a in sim.agents)
        unique_variances = set(round(a.regime_classifier.config.variance_high, 4) for a in sim.agents)
        ok = len(unique_neighbors) > 1 and len(unique_variances) > 1
        self.log_result(
            "Regime Detection", ok,
            f"Thresholds jittered: {len(unique_neighbors)} neighbor levels, "
            f"{len(unique_variances)} variance levels.",
        )

    # ── 7. Safety projection ────────────────────────────────────────────
    def run_stability_projection_check(self):
        """
        Fails if: no projection ever fires, or any agent's final parameters
        sit outside THETA_SAFE_BOUNDS. A previous version reported the count
        and logged True unconditionally -- it passed identically at zero.
        """
        print("\n--- 7. Stability Projection Verification ---")
        config = SimConfig(num_agents=10, max_time=100.0, seed=42,
                           latency_mean=5.0, theta_safe_enabled=True,
                           coverage_enabled=True)
        sim = Phase1Simulation(config)
        sim.run()

        total_projections = sum(a.projection_events for a in sim.agents)
        out_of_bounds = []
        for a in sim.agents:
            for key, (lo, hi) in THETA_SAFE_BOUNDS.items():
                val = getattr(a, key)
                if not (lo - 1e-9 <= val <= hi + 1e-9):
                    out_of_bounds.append((a.agent_id, key, val))
        ok = total_projections > 0 and not out_of_bounds
        self.log_result(
            "Stability Constraints", ok,
            f"{total_projections} projection events; every final parameter "
            f"inside THETA_SAFE_BOUNDS." if ok else
            f"projections={total_projections}, out_of_bounds={out_of_bounds[:3]}",
        )

    # ── 8. Adaptive tuning smoothness ───────────────────────────────────
    def run_adaptive_tuning_check(self):
        """Fails if: any single EMA update jumps a parameter by >= 0.5."""
        print("\n--- 8. Adaptive Tuning Safety ---")
        sim = Phase1Simulation(SimConfig(num_agents=20, max_time=50.0, seed=42,
                                         tuning_alpha=0.15))
        sim.run()
        max_shift = max(a.max_parameter_shift for a in sim.agents)
        self.log_result("Adaptive Tuning", max_shift < 0.5,
                        f"Max single-update parameter shift: {max_shift:.4f}")

    # ── 9. Performance benchmark (NOT a stability check) ────────────────
    def run_performance_benchmark(self):
        """
        Wall-clock benchmark. A previous version labelled this "Stable under
        load", but a slow machine is not an unstable simulation; it is
        reported as a benchmark and only fails on gross regression (>120s
        for a workload that takes ~15s here).
        """
        print("\n--- 9. Performance Benchmark ---")
        config = SimConfig(num_agents=100, max_time=100.0, seed=42, log_dir="logs")
        sim = Phase1Simulation(config)
        start_time = time.time()
        dispatched = sim.run()
        duration = time.time() - start_time
        self.log_result(
            "Performance Benchmark", duration < 120.0,
            f"{dispatched} events in {duration:.2f}s "
            f"({1e6 * duration / max(dispatched, 1):.1f} us/event).",
        )

    def print_report(self):
        print("\n" + "=" * 80)
        print("ARCHITECTURAL DIAGNOSTICS AUDIT REPORT")
        print("=" * 80)
        print(f"{'Check':<30} | {'Result':<10} | {'Notes'}")
        print("-" * 80)
        for r in self.results:
            print(f"{r.check:<30} | {r.result:<10} | {r.notes}")
        print("=" * 80)
        failures = sum(1 for r in self.results if r.result == "FAILURE")
        print(f"{len(self.results) - failures}/{len(self.results)} checks passed.")
        return failures


def run_all(audit: "ArchitecturalAudit") -> int:
    audit.run_deterministic_replay_test()
    audit.run_kernel_integrity_check()
    audit.run_fog_of_war_test()
    audit.run_communication_locality_check()
    audit.run_task_allocation_integrity()
    audit.run_regime_asynchrony_check()
    audit.run_stability_projection_check()
    audit.run_adaptive_tuning_check()
    audit.run_performance_benchmark()
    return audit.print_report()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        # Deliberately break an invariant and confirm the audit CATCHES it.
        # A check that cannot be made to fail is not a check.
        #
        # NOTE the sabotage must be EXTERNAL to the check's own reference:
        # a first version widened THETA_SAFE_BOUNDS instead, and the audit
        # still passed 9/9 -- the "parameters inside bounds" clause compares
        # against the same sabotaged dict (self-referential), and the
        # bisection stage kept firing projection events on its own. Disabling
        # the projector function itself is the honest breakage: zero
        # projection events can then ever be recorded.
        print(">>> SELF-TEST: replacing project_to_theta_safe with a "
              "passthrough; expect 'Stability Constraints' to report "
              "FAILURE. <<<")
        import src.adaptation.safety_projector as sp
        import src.agent.agent_core as ac

        def passthrough(theta_proposed, theta_nominal, dynamic_bounds=None):
            return dict(theta_proposed), 0

        sp.project_to_theta_safe = passthrough
        ac.project_to_theta_safe = passthrough
        audit = ArchitecturalAudit()
        run_all(audit)
        stab = next(r for r in audit.results if r.check == "Stability Constraints")
        ok = stab.result == "FAILURE"
        print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}: sabotaged projector "
              f"was {'caught' if ok else 'NOT caught'} by the audit.")
        sys.exit(0 if ok else 1)
    else:
        failures = run_all(ArchitecturalAudit())
        sys.exit(1 if failures else 0)
