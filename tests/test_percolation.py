"""
Phase 2Q verification: Percolation Analyzer tests.
"""

import os
import sys
import time

import numpy as np
from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.percolation_analyzer import PercolationAnalyzer


def _make_frame_with_graph(n: int, connections: list[tuple[int, int]], failures: list[int] = None) -> TelemetryFrame:
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in connections:
        adj[u, v] = 1
        adj[v, u] = 1

    fail_arr = np.zeros(n, dtype=bool)
    if failures:
        for f in failures:
            fail_arr[f] = True

    return TelemetryFrame(
        time=time.time(),
        positions=np.zeros((n, 2)),
        energies=np.ones(n) * 100,
        adjacency=adj,
        connected_components=[],
        spectral_gap=0.0,
        consensus_variance=0.0,
        packet_drop_rate=0.0,
        latency=0.0,
        regime_state={},
        adaptive_parameters={},
        drone_failure_flags=fail_arr,
    )


def test_lcc_detection():
    # 5 agents:
    # 0-1-2 (LCC = 3)
    # 3-4   (Comp = 2)
    connections = [(0, 1), (1, 2), (3, 4)]
    frame = _make_frame_with_graph(5, connections)
    
    analyzer = PercolationAnalyzer()
    metrics = analyzer.analyze_frame(frame)
    
    assert metrics.lcc_size == 3
    assert metrics.total_agents == 5
    assert metrics.num_components == 2
    assert metrics.connectivity_ratio == 3.0 / 5.0
    print("[PASS] LCC and connectivity ratio correct")


def test_hysteresis_and_collapse_signals():
    # 4 agents
    analyzer = PercolationAnalyzer(collapse_threshold=0.5, recovery_threshold=0.7)
    
    events_collapse = []
    events_recover = []
    
    analyzer.percolation_collapse_detected.connect(lambda t: events_collapse.append(t))
    analyzer.percolation_recovered.connect(lambda t: events_recover.append(t))
    
    # Init: 0-1-2-3 (ratio 4/4 = 1.0)
    frame = _make_frame_with_graph(4, [(0, 1), (1, 2), (2, 3)])
    analyzer.analyze_frame(frame)
    assert len(events_collapse) == 0
    
    # Degrade to ratio 0.5 (2/4). Should trigger collapse (ratio < threshold NOT <=, wait. Our logic: ratio < threshold)
    # threshold = 0.5, let's make it 1/4 = 0.25 to guarantee trigger.
    # Disconnect all: 0, 1, 2, 3
    frame = _make_frame_with_graph(4, [])
    analyzer.analyze_frame(frame)
    assert len(events_collapse) == 1
    assert len(events_recover) == 0
    
    # Recover slightly to 0.5 (2/4 = 0.5). Still in collapse
    frame = _make_frame_with_graph(4, [(0, 1)])
    analyzer.analyze_frame(frame)
    assert len(events_collapse) == 1
    assert len(events_recover) == 0
    
    # Fully recover to 1.0 (4/4 = 1.0)
    frame = _make_frame_with_graph(4, [(0, 1), (1, 2), (2, 3)])
    analyzer.analyze_frame(frame)
    assert len(events_collapse) == 1
    assert len(events_recover) == 1
    
    print("[PASS] Hysteresis and event triggers correct")


def test_failure_handling():
    # 5 agents: 0-1-2 are connected, 2 is DEAD.
    # Alive agents: 0, 1, 3, 4 (Total 4 alive)
    # Connection 0-1 exists.
    # LCC size should be 2. Ratio = 2/4 = 0.5
    connections = [(0, 1), (1, 2)]
    frame = _make_frame_with_graph(5, connections, failures=[2])
    
    analyzer = PercolationAnalyzer()
    metrics = analyzer.analyze_frame(frame)
    
    assert metrics.lcc_size == 2
    assert metrics.total_agents == 4
    assert metrics.connectivity_ratio == 0.5
    print("[PASS] Failure handling correct (dead agents ignored)")


def test_performance_skip():
    frame = _make_frame_with_graph(50, [(i, i+1) for i in range(49)])
    
    analyzer = PercolationAnalyzer()
    
    t0 = time.perf_counter()
    analyzer.analyze_frame(frame)
    t1 = time.perf_counter()
    
    # Process identical frame
    analyzer.analyze_frame(frame)
    t2 = time.perf_counter()
    
    first_pass_ms = (t1 - t0) * 1000
    skip_pass_ms = (t2 - t1) * 1000
    
    assert skip_pass_ms < first_pass_ms, "Cached skip should be faster than BFS"
    assert skip_pass_ms < 0.2, "Cached skip should be extremely fast (<0.2ms)"
    
    print(f"[PASS] Performance test: first_pass={first_pass_ms:.3f}ms, skip_pass={skip_pass_ms:.3f}ms")


if __name__ == "__main__":
    test_lcc_detection()
    test_hysteresis_and_collapse_signals()
    test_failure_handling()
    test_performance_skip()
    print("\nAll Phase 2Q tests passed.")
