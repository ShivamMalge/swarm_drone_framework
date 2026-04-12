"""
Phase 2AA verification: Event Fusion Engine tests
"""

import time
import numpy as np

from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.percolation_analyzer import PercolationMetrics
from src.analytics.spectral_analyzer import SpectralMetrics
from src.analytics.energy_cascade_analyzer import EnergyMetrics
from src.analytics.consensus_analyzer import ConsensusMetrics
from src.analytics.anomaly_detector import AnomalyMetrics
from src.analytics.swarm_health import HealthMetrics
from src.analytics.system_event import SystemEvent
from src.analytics.event_fusion_engine import EventFusionEngine


def _frame(t: float = 0.0) -> TelemetryFrame:
    return TelemetryFrame(
        time=t, positions=np.zeros((1, 2)),
        energies=np.ones(1) * 100.0,
        adjacency=np.zeros((1, 1), dtype=np.uint8),
        connected_components=[], spectral_gap=0.0,
        consensus_variance=0.0, packet_drop_rate=0.0,
        latency=0.0, regime_state={}, adaptive_parameters={},
        drone_failure_flags=np.zeros(1, dtype=bool),
        agent_states=np.zeros(1),
    )


def _healthy():
    perc = PercolationMetrics(10, 10, 1, 1.0, 1.0, 0.5, 0.0, "STABLE")
    spec = SpectralMetrics(1.0, 1.0, 0.5, 0.0, "STABLE")
    eng = EnergyMetrics(
        mean_energy=100.0, smoothed_mean_energy=100.0, min_energy=100.0,
        energy_variance=0.0, alive_count=10, failure_count=0, new_failures=0,
        cascade_intensity=0.0, smoothed_intensity=0.0, normalized_energy=1.0,
        cascade_margin=0.5, d_mean_energy_dt=0.0,
        energy_gradients=np.zeros(1), dead_agents=np.zeros(0),
        hotspot_indices=[], state="STABLE",
    )
    cns = ConsensusMetrics(
        consensus_variance=0.0, consensus_error=0.0, normalized_variance=0.0,
        smoothed_variance=0.0, smoothed_rate=0.0, consensus_margin=0.5,
        global_convergence_rate=0.0, hotspot_indices=[], state="CONVERGED",
    )
    anom = AnomalyMetrics(np.zeros(1), np.zeros(1), np.zeros(1, dtype=np.uint8), 0, 0, 0)
    health = HealthMetrics(1.0, 1.0, "HEALTHY")
    return perc, spec, eng, cns, anom, health


def _collapsed():
    perc = PercolationMetrics(2, 10, 5, 0.2, 0.2, -0.3, -0.1, "COLLAPSE")
    spec = SpectralMetrics(0.01, 0.01, -0.09, -0.05, "UNSTABLE")
    eng = EnergyMetrics(
        mean_energy=10.0, smoothed_mean_energy=10.0, min_energy=0.0,
        energy_variance=5.0, alive_count=5, failure_count=5, new_failures=2,
        cascade_intensity=0.8, smoothed_intensity=0.7, normalized_energy=0.1,
        cascade_margin=-0.4, d_mean_energy_dt=-2.0,
        energy_gradients=np.zeros(1), dead_agents=np.array([1, 3, 5, 7, 9]),
        hotspot_indices=[0, 2], state="CASCADE",
    )
    cns = ConsensusMetrics(
        consensus_variance=5.0, consensus_error=3.0, normalized_variance=0.8,
        smoothed_variance=0.7, smoothed_rate=0.5, consensus_margin=-0.3,
        global_convergence_rate=0.5, hotspot_indices=[0], state="DIVERGING",
    )
    anom = AnomalyMetrics(np.ones(10) * 0.8, np.ones(10) * 0.9,
                          np.full(10, 2, dtype=np.uint8), 2, 8, 8)
    health = HealthMetrics(0.15, 0.15, "COLLAPSE")
    return perc, spec, eng, cns, anom, health


# ── Tests ───────────────────────────────────────────────


def test_event_trigger_collapse():
    engine = EventFusionEngine()
    f = _frame(1.0)
    perc, spec, eng, cns, anom, health = _collapsed()

    events = engine.analyze_frame(f, perc, spec, eng, cns, anom, health, 1)

    types = {e.type for e in events}
    assert "collapse" in types, f"Expected collapse, got {types}"
    assert "cascade" in types, f"Expected cascade, got {types}"
    assert "health_critical" in types, f"Expected health_critical, got {types}"

    print("[PASS] Collapse scenario triggers correct event types")


def test_event_trigger_healthy():
    engine = EventFusionEngine()
    f = _frame(0.0)
    perc, spec, eng, cns, anom, health = _healthy()

    events = engine.analyze_frame(f, perc, spec, eng, cns, anom, health, 0)

    types = {e.type for e in events}
    assert "recovery" in types, f"Expected recovery, got {types}"
    assert "collapse" not in types
    assert "cascade" not in types

    print("[PASS] Healthy scenario triggers recovery only")


def test_deduplication():
    engine = EventFusionEngine()
    f = _frame(1.0)
    perc, spec, eng, cns, anom, health = _collapsed()

    e1 = engine.analyze_frame(f, perc, spec, eng, cns, anom, health, 1)

    # Same inputs but different time (hash changes because metrics are same
    # but we bump frame_index via different args) — however internally the
    # hash is built from metric values only, so identical metrics → skip.
    f2 = _frame(2.0)
    e2 = engine.analyze_frame(f2, perc, spec, eng, cns, anom, health, 2)

    # Hash skip returns cached list — same object
    assert e1 is e2, "Hash-skip should return cached events"
    print("[PASS] Hash-skip prevents recomputation on identical metrics")


def test_dedup_window():
    engine = EventFusionEngine()

    # First call: triggers events
    f = _frame(1.0)
    perc, spec, eng, cns, anom, health = _collapsed()
    e1 = engine.analyze_frame(f, perc, spec, eng, cns, anom, health, 1)
    assert len(e1) > 0

    # Change metrics slightly to defeat hash-skip but trigger same types
    perc2 = PercolationMetrics(2, 10, 5, 0.19, 0.19, -0.31, -0.11, "COLLAPSE")
    f2 = _frame(2.0)
    e2 = engine.analyze_frame(f2, perc2, spec, eng, cns, anom, health, 2)

    # The dedup window should suppress repeated event types
    # e2 may be empty or a subset since collapse/cascade/health_critical are already in window
    overlap = {e.type for e in e2} & {e.type for e in e1}
    assert len(overlap) == 0, f"Dedup should suppress repeated types, got overlap {overlap}"
    print("[PASS] Deduplication window suppresses repeated event types")


def test_severity_bounds():
    e = SystemEvent("test", 5.0, -1.0, 0.0, 0, "test")
    assert e.severity == 1.0, f"severity not clamped: {e.severity}"
    assert e.confidence == 0.0, f"confidence not clamped: {e.confidence}"

    e2 = SystemEvent("test", 0.5, 0.5, 0.0, 0, "test")
    assert e2.severity == 0.5
    assert e2.confidence == 0.5
    print("[PASS] SystemEvent clamps severity and confidence correctly")


def test_frozen_immutability():
    e = SystemEvent("test", 0.5, 0.5, 0.0, 0, "test")
    try:
        e.severity = 0.9  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
    print("[PASS] SystemEvent is frozen / immutable")


def test_determinism():
    """Two engines with identical input sequences produce identical outputs."""
    e1 = EventFusionEngine()
    e2 = EventFusionEngine()

    f = _frame(1.0)
    perc, spec, eng, cns, anom, health = _collapsed()

    r1 = e1.analyze_frame(f, perc, spec, eng, cns, anom, health, 1)
    r2 = e2.analyze_frame(f, perc, spec, eng, cns, anom, health, 1)

    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.type == b.type
        assert a.severity == b.severity
        assert a.confidence == b.confidence
    print("[PASS] Deterministic: two engines produce identical results")


def test_performance():
    engine = EventFusionEngine()
    f = _frame(0.0)
    perc, spec, eng, cns, anom, health = _collapsed()

    # Warm up
    engine.analyze_frame(f, perc, spec, eng, cns, anom, health, 0)

    t0 = time.perf_counter()
    for i in range(100):
        # Vary connectivity ratio to defeat hash-skip
        perc_var = PercolationMetrics(
            2, 10, 5, 0.2 + i * 0.001, 0.2 + i * 0.001,
            -0.3, -0.1, "COLLAPSE")
        engine.analyze_frame(f, perc_var, spec, eng, cns, anom, health, i)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) * 1000) / 100
    assert avg_ms < 1.0, f"Too slow: {avg_ms:.3f}ms"
    print(f"[PASS] Performance: {avg_ms:.4f}ms per frame (target < 1ms)")


def test_buffer_bound():
    engine = EventFusionEngine()
    f = _frame(0.0)
    perc, spec, eng, cns, anom, health = _collapsed()

    # Force many distinct events by clearing the dedup window each time
    for i in range(300):
        engine._recent_types.clear()  # bypass dedup for stress test
        perc_var = PercolationMetrics(
            2, 10, 5, 0.2 + i * 0.0001, 0.2 + i * 0.0001,
            -0.3, -0.1, "COLLAPSE")
        engine.analyze_frame(f, perc_var, spec, eng, cns, anom, health, i)

    assert len(engine.events) <= 200, f"Buffer exceeded: {len(engine.events)}"
    print(f"[PASS] Event buffer bounded at {len(engine.events)} (max 200)")


if __name__ == "__main__":
    test_event_trigger_collapse()
    test_event_trigger_healthy()
    test_deduplication()
    test_dedup_window()
    test_severity_bounds()
    test_frozen_immutability()
    test_determinism()
    test_performance()
    test_buffer_bound()
    print("\nAll Phase 2AA tests passed.")
