"""
Phase 2U verification: Network Topology Viewer tests.
"""

import time
import numpy as np

from PySide6.QtWidgets import QApplication

from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.percolation_analyzer import PercolationAnalyzer
from src.gui.network_viewer import NetworkViewer


def _make_frame(n: int, fails: list[int] = None, adj: np.ndarray = None) -> TelemetryFrame:
    if adj is None:
        adj = np.zeros((n, n), dtype=np.uint8)
        
    fail_arr = np.zeros(n, dtype=bool)
    if fails:
        for f in fails:
            fail_arr[f] = True

    return TelemetryFrame(
        time=0.0,
        positions=np.random.rand(n, 2) * 100,
        energies=np.ones(n),
        adjacency=adj,
        connected_components=[],
        spectral_gap=0.0,
        consensus_variance=0.0,
        packet_drop_rate=0.0,
        latency=0.0,
        regime_state={},
        adaptive_parameters={},
        drone_failure_flags=fail_arr,
        agent_states=np.zeros(n)
    )


def test_dead_agent_exclusion_and_degree():
    app = QApplication.instance()
    if not app:
        app = QApplication([])
        
    viewer = NetworkViewer()
    
    # K_3 graph: 0-1-2 triangle
    adj = np.zeros((3, 3), dtype=np.uint8)
    adj[0,1] = adj[1,0] = 1
    adj[1,2] = adj[2,1] = 1
    adj[0,2] = adj[2,0] = 1
    
    # 2 is dead
    frame = _make_frame(3, fails=[2], adj=adj)
    
    # Percolation computes frame.connected_components
    perc = PercolationAnalyzer()
    m = perc.analyze_frame(frame)
    # LCC should be size 2. Components should reflect alive ones.
    # Note: test uses perc to populate things theoretically, but perc doesn't modify frame in 2Q.
    # NetworkViewer uses frame.connected_components, let's inject it:
    frame.connected_components = [[0, 1], [2]]
    
    viewer.update_frame(frame, m)
    
    # Verify cached stuff
    sizes = viewer._cached_sizes
    brushes = viewer._cached_brushes
    
    # Degrees array calculated internally inside viewer. Degrees before masking was 2, 2, 2.
    # Sizes logic: size = 6.0 + 1.5 * degree
    assert sizes[0] == 6.0 + 1.5 * 2
    assert sizes[1] == 6.0 + 1.5 * 2
    
    # Brush for dead agent 2 should be the dead palette
    assert brushes[2] == viewer._color_dead
    assert brushes[0] == viewer._comp_colors[0] # LCC
    
    print("[PASS] Correct component coloring, degree, and dead-agent exclusion")


def test_hash_skip_correctness():
    app = QApplication.instance()
    if not app:
        app = QApplication([])
        
    viewer = NetworkViewer()
    frame = _make_frame(4)
    viewer.update_frame(frame)
    
    hash1 = viewer._last_adj_hash
    assert hash1 != -1
    
    # update positions only
    frame.positions += 1.0
    viewer.update_frame(frame)
    hash2 = viewer._last_adj_hash
    
    # should be cached
    assert hash1 == hash2
    
    # Modify adj -> should uncache
    frame.adjacency[1,2] = 1
    frame.adjacency[2,1] = 1
    viewer.update_frame(frame)
    hash3 = viewer._last_adj_hash
    
    assert hash3 != hash2
    
    print("[PASS] Hash-skip topological caching correct")


def test_performance():
    app = QApplication.instance()
    if not app:
        app = QApplication([])
        
    n = 200
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n-1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
        
    frame = _make_frame(n, adj=adj)
    frame.connected_components = [list(range(n))]
    
    perc = PercolationAnalyzer()
    m = perc.analyze_frame(frame)
    
    viewer = NetworkViewer()
    
    # Warmup
    viewer.update_frame(frame, m)
    
    # Rapid position updates (testing the cached line segments draw)
    t0 = time.perf_counter()
    for _ in range(50):
        frame.positions += 0.5
        viewer.update_frame(frame, m)
    t1 = time.perf_counter()
    
    avg_ms = ((t1 - t0) * 1000) / 50
    assert avg_ms < 5.0, f"Frame render too slow: {avg_ms:.2f}ms (target < 5ms)"
    
    # Rapid topology updates (testing un-cached full rebuilds)
    t2 = time.perf_counter()
    for _ in range(10):
        # randomly flip an edge
        i, j = np.random.randint(0, n, 2)
        frame.adjacency[i, j] ^= 1
        frame.adjacency[j, i] = frame.adjacency[i, j]
        viewer.update_frame(frame, m)
    t3 = time.perf_counter()
    
    avg_ms_rebuild = ((t3 - t2) * 1000) / 10
    
    print(f"[PASS] Performance tracking: draw={avg_ms:.3f}ms | rebuild={avg_ms_rebuild:.3f}ms")


if __name__ == "__main__":
    test_dead_agent_exclusion_and_degree()
    test_hash_skip_correctness()
    test_performance()
    print("\nAll Phase 2U tests passed.")
