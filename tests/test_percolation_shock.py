"""
Phase 2AA.1 verification: Percolation Shock Analyzer tests
"""

import time
import numpy as np

from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.percolation_analyzer import PercolationMetrics
from src.analytics.percolation_shock_analyzer import PercolationShockAnalyzer


def _frame(n: int, adj: np.ndarray | None = None,
           fail: list[int] | None = None) -> TelemetryFrame:
    if adj is None:
        adj = np.ones((n, n), dtype=np.uint8) - np.eye(n, dtype=np.uint8)
    fail_arr = np.zeros(n, dtype=bool)
    if fail:
        for i in fail:
            fail_arr[i] = True
    return TelemetryFrame(
        time=0.0, positions=np.random.rand(n, 2) * 100,
        energies=np.ones(n) * 100.0, adjacency=adj,
        connected_components=[], spectral_gap=0.0,
        consensus_variance=0.0, packet_drop_rate=0.0,
        latency=0.0, regime_state={}, adaptive_parameters={},
        drone_failure_flags=fail_arr,
        agent_states=np.ones(n) * 50.0,
    )


def _perc(ratio: float, lcc_nodes: np.ndarray | None = None) -> PercolationMetrics:
    return PercolationMetrics(
        lcc_size=int(ratio * 10), total_agents=10,
        num_components=max(1, int((1 - ratio) * 5)),
        connectivity_ratio=ratio, normalized_ratio=ratio,
        connectivity_margin=ratio - 0.5, d_ratio_dt=0.0,
        state="STABLE" if ratio > 0.6 else "COLLAPSE",
        lcc_nodes=lcc_nodes,
    )


def test_no_shock_when_healthy():
    analyzer = PercolationShockAnalyzer()
    f = _frame(10)
    perc = _perc(1.0, np.arange(10))

    m = analyzer.analyze(f, perc)
    assert not m.shock_active
    assert m.max_distance == 0.0
    print("[PASS] No shock when connectivity is healthy")


def test_shock_triggers_below_threshold():
    analyzer = PercolationShockAnalyzer()
    f = _frame(10)
    lcc = np.array([0, 1, 2, 3, 4])
    perc = _perc(0.5, lcc)

    m = analyzer.analyze(f, perc)
    assert m.shock_active
    # LCC nodes should have distance 0
    for node in lcc:
        assert m.shock_distance[node] == 0.0
    print("[PASS] Shock triggers when LCC ratio < 0.6")


def test_shock_propagation():
    """Build a chain graph: 0-1-2-3-4.  LCC = {0,1}.  Shock should expand."""
    n = 5
    adj = np.zeros((n, n), dtype=np.uint8)
    adj[0, 1] = adj[1, 0] = 1
    adj[1, 2] = adj[2, 1] = 1
    adj[2, 3] = adj[3, 2] = 1
    adj[3, 4] = adj[4, 3] = 1

    analyzer = PercolationShockAnalyzer()
    f = _frame(n, adj)
    lcc = np.array([0, 1])
    perc = _perc(0.4, lcc)

    # Frame 1: seed
    m1 = analyzer.analyze(f, perc)
    assert m1.shock_active
    assert m1.shock_distance[0] == 0.0
    assert m1.shock_distance[1] == 0.0

    # Frame 2: propagate (change adjacency slightly to defeat hash)
    f.positions += 0.001
    m2 = analyzer.analyze(f, perc)
    # Node 2 should now have distance 1 (one hop from node 1)
    assert m2.shock_distance[2] == 1.0, f"Expected 1.0, got {m2.shock_distance[2]}"

    # Frame 3
    f.positions += 0.001
    m3 = analyzer.analyze(f, perc)
    assert m3.shock_distance[3] == 2.0, f"Expected 2.0, got {m3.shock_distance[3]}"

    # Frame 4
    f.positions += 0.001
    m4 = analyzer.analyze(f, perc)
    assert m4.shock_distance[4] == 3.0, f"Expected 3.0, got {m4.shock_distance[4]}"

    print("[PASS] Shock propagation expands correctly along chain")


def test_dead_agents_excluded():
    n = 5
    adj = np.ones((n, n), dtype=np.uint8) - np.eye(n, dtype=np.uint8)
    analyzer = PercolationShockAnalyzer()

    f = _frame(n, adj, fail=[3, 4])
    lcc = np.array([0, 1])
    perc = _perc(0.4, lcc)

    m = analyzer.analyze(f, perc)
    assert m.shock_active
    assert np.isinf(m.shock_distance[3])
    assert np.isinf(m.shock_distance[4])
    assert m.shock_normalized[3] == 0.0
    assert m.shock_normalized[4] == 0.0
    print("[PASS] Dead agents excluded from shock propagation")


def test_normalised_bounds():
    n = 6
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1

    analyzer = PercolationShockAnalyzer()
    f = _frame(n, adj)
    lcc = np.array([0])
    perc = _perc(0.16, lcc)

    # Run several frames to propagate
    for _ in range(10):
        f.positions += 0.001
        m = analyzer.analyze(f, perc)

    assert np.all(m.shock_normalized >= 0.0)
    assert np.all(m.shock_normalized <= 1.0)
    print("[PASS] Normalised shock distances within [0, 1]")


def test_determinism():
    n = 8
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1

    def run():
        a = PercolationShockAnalyzer()
        f = _frame(n, adj.copy())
        lcc = np.array([0, 1])
        perc = _perc(0.25, lcc)
        results = []
        for _ in range(5):
            f.positions += 0.001
            m = a.analyze(f, perc)
            results.append(m.shock_distance.copy())
        return results

    r1 = run()
    r2 = run()
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)
    print("[PASS] Deterministic replay produces identical shock evolution")


def test_hash_skip():
    analyzer = PercolationShockAnalyzer()
    f = _frame(5)
    lcc = np.array([0, 1])
    perc = _perc(0.4, lcc)

    m1 = analyzer.analyze(f, perc)
    h1 = analyzer._last_hash

    m2 = analyzer.analyze(f, perc)
    assert m1 is m2
    assert analyzer._last_hash == h1
    print("[PASS] Hash-skip returns cached metrics on identical input")


def test_performance():
    n = 200
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
    # Add some random connections
    rng = np.random.default_rng(42)
    for _ in range(300):
        a, b = rng.integers(0, n, 2)
        adj[a, b] = adj[b, a] = 1

    analyzer = PercolationShockAnalyzer()
    f = _frame(n, adj)
    lcc = np.arange(50)
    perc = _perc(0.25, lcc)

    # Warm up
    analyzer.analyze(f, perc)

    t0 = time.perf_counter()
    for _ in range(50):
        f.positions += 0.001
        analyzer.analyze(f, perc)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) * 1000) / 50
    assert avg_ms < 5.0, f"Too slow: {avg_ms:.2f}ms"
    print(f"[PASS] Performance: {avg_ms:.3f}ms per frame (N=200)")


if __name__ == "__main__":
    test_no_shock_when_healthy()
    test_shock_triggers_below_threshold()
    test_shock_propagation()
    test_dead_agents_excluded()
    test_normalised_bounds()
    test_determinism()
    test_hash_skip()
    test_performance()
    print("\nAll Phase 2AA.1 tests passed.")
