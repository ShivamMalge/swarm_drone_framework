"""
Phase 2R verification: Spectral Analyzer tests.
"""

import time
import numpy as np
from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.spectral_analyzer import SpectralAnalyzer


def _make_frame_with_graph(n: int, connections: list[tuple[int, int]], t: float = 0.0) -> TelemetryFrame:
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in connections:
        adj[u, v] = 1
        adj[v, u] = 1

    return TelemetryFrame(
        time=float(t),
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
        drone_failure_flags=np.zeros(n, dtype=bool),
    )


def test_lambda2_fully_connected():
    # Fully connected K_N graph has lambda2 = N
    n = 5
    conns = [(i, j) for i in range(n) for j in range(i + 1, n)]
    frame = _make_frame_with_graph(n, conns)
    
    analyzer = SpectralAnalyzer()
    metrics = analyzer.analyze_frame(frame)
    
    np.testing.assert_allclose(metrics.lambda2, n, atol=1e-5)
    print(f"[PASS] Fully connected K_{n} lambda2 == {n}")


def test_lambda2_disconnected():
    # Disconnected graph has lambda2 = 0
    conns = [(0, 1), (2, 3)]  # 4 nodes, 2 components
    frame = _make_frame_with_graph(4, conns)
    
    analyzer = SpectralAnalyzer()
    metrics = analyzer.analyze_frame(frame)
    
    assert metrics.lambda2 == 0.0
    print("[PASS] Disconnected graph lambda2 == 0.0")


def test_hysteresis_and_collapse_signals():
    analyzer = SpectralAnalyzer(collapse_threshold=0.1, recovery_threshold=0.15)
    
    events_collapse = []
    events_recover = []
    
    analyzer.spectral_instability_detected.connect(lambda t: events_collapse.append(t))
    analyzer.spectral_recovered.connect(lambda t: events_recover.append(t))
    
    # K_4 has lambda2 = 4 (state = STRONG)
    conns = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    frame = _make_frame_with_graph(4, conns, t=1.0)
    analyzer.analyze_frame(frame)
    assert len(events_collapse) == 0
    
    # Path graph P_5 has lambda2 = ~0.38 (we need to go below 0.1)
    # Let's just use disconnected graph to drop instantly
    frame2 = _make_frame_with_graph(4, [], t=2.0)
    analyzer.analyze_frame(frame2)
    assert len(events_collapse) == 1
    assert len(events_recover) == 0
    
    # Small connection (P_4 has lambda2 ~0.58 so we use something weaker or just P_4)
    # Actually P_4 lambda2 is ~0.58. 
    # Let's test the threshold properly. K_4 -> Disconnect -> Complete K_4 again
    frame3 = _make_frame_with_graph(4, conns, t=3.0)
    analyzer.analyze_frame(frame3)
    assert len(events_collapse) == 1
    assert len(events_recover) == 1
    print("[PASS] Hysteresis and instability events correctly triggered")


def test_caching_and_performance():
    # Large graph N=100
    n = 100
    # ring graph
    conns = [(i, (i + 1) % n) for i in range(n)]
    frame = _make_frame_with_graph(n, conns, t=0.0)
    
    analyzer = SpectralAnalyzer()
    
    t0 = time.perf_counter()
    metrics1 = analyzer.analyze_frame(frame)
    t1 = time.perf_counter()
    
    frame2 = _make_frame_with_graph(n, conns, t=1.0)
    t2 = time.perf_counter()
    metrics2 = analyzer.analyze_frame(frame2)
    t3 = time.perf_counter()
    
    calc_time_ms = (t1 - t0) * 1000
    cache_time_ms = (t3 - t2) * 1000
    
    assert calc_time_ms < 20.0, f"Calculation too slow: {calc_time_ms:.2f}ms"
    assert cache_time_ms < 0.5, f"Cache skip too slow: {cache_time_ms:.2f}ms"
    assert metrics1.lambda2 == metrics2.lambda2
    
    print(f"[PASS] Performance tracking: compute={calc_time_ms:.3f}ms  cache={cache_time_ms:.3f}ms")


if __name__ == "__main__":
    test_lambda2_fully_connected()
    test_lambda2_disconnected()
    test_hysteresis_and_collapse_signals()
    test_caching_and_performance()
    print("\nAll Phase 2R tests passed.")
