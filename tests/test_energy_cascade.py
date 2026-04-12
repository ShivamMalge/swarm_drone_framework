"""
Phase 2S verification: Energy Cascade Analyzer tests.
"""

import time
import numpy as np
from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.energy_cascade_analyzer import EnergyCascadeAnalyzer


def _make_frame(n: int, t: float, energies: list[float], fails: list[int] = None, adj: np.ndarray = None) -> TelemetryFrame:
    if adj is None:
        adj = np.zeros((n, n), dtype=np.uint8)
        
    fail_arr = np.zeros(n, dtype=bool)
    if fails:
        for f in fails:
            fail_arr[f] = True

    return TelemetryFrame(
        time=float(t),
        positions=np.zeros((n, 2)),
        energies=np.array(energies, dtype=np.float64),
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


def test_energy_metrics_and_dead_agent_exclusion():
    # 3 agents. #2 is dead.
    energies = [100.0, 50.0, 0.0]
    frame = _make_frame(3, 0.0, energies, fails=[2])
    
    analyzer = EnergyCascadeAnalyzer()
    metrics = analyzer.analyze_frame(frame)
    
    assert metrics.alive_count == 2
    assert metrics.failure_count == 1
    # mean of alive (100, 50) is 75
    assert metrics.mean_energy == 75.0
    assert metrics.min_energy == 50.0
    
    print("[PASS] Energy metrics & dead agent exclusion correct")


def test_dEdt_and_early_warning():
    analyzer = EnergyCascadeAnalyzer(stress_drop_rate=2.0)
    
    warnings = []
    analyzer.energy_stress_warning.connect(lambda t: warnings.append(t))
    
    # Frame 0: e=100
    f1 = _make_frame(2, 0.0, [100.0, 100.0])
    analyzer.analyze_frame(f1)
    
    # Frame 1: 1 second later, e=95 (drop 5 J/s, > threshold 2.0)
    f2 = _make_frame(2, 1.0, [95.0, 95.0])
    analyzer.analyze_frame(f2)
    
    assert len(warnings) == 1, "Early warning should trigger on rapid drop"
    
    # Frame 2: e=94 (drop 1 J/s, < threshold 2.0)
    f3 = _make_frame(2, 2.0, [94.0, 94.0])
    analyzer.analyze_frame(f3)
    
    # Should not trigger again immediately (has hysteresis/state check)
    assert len(warnings) == 1
    
    print("[PASS] dE/dt calculation & Early Warning trigger correct")


def test_cascade_detection_and_hysteresis():
    analyzer = EnergyCascadeAnalyzer(cascade_threshold=0.3, recovery_threshold=0.1, window_frames=2)
    
    cascades = []
    recovers = []
    analyzer.energy_cascade_detected.connect(lambda t: cascades.append(t))
    analyzer.energy_cascade_recovered.connect(lambda t: recovers.append(t))
    
    # t=0: 4 agents
    f0 = _make_frame(4, 0.0, [10.0, 10.0, 10.0, 10.0])
    analyzer.analyze_frame(f0)
    
    # t=1: 1 failure -> fails=1, alive=3. intensity in window=1 / 3 = 0.33. 
    # Also need mean drop for cascade check! 
    # Last mean was 10. New mean is 8. Drop rate = 2 > 0.
    f1 = _make_frame(4, 1.0, [8.0, 8.0, 8.0, 0.0], fails=[3])
    m1 = analyzer.analyze_frame(f1)
    
    assert len(cascades) == 1
    
    # t=2: No new failures, window length 2 so window drops old failure if we do one more step 
    # Actually wait, ring buffer size=2.
    # At t=0 append(0). [0, 0] sum=0
    # At t=1 append(1). [0, 1] sum=1 -> intensity 1/3 = 0.33
    # At t=2 append(0). [0, 1] sum=1 -> intensity 1/3 = 0.33
    f2 = _make_frame(4, 2.0, [8.0, 8.0, 8.0, 0.0], fails=[3])
    m2 = analyzer.analyze_frame(f2)
    
    assert len(cascades) == 1
    assert len(recovers) == 0
    
    # t=3 append(0). [0, 0] sum=0 -> intensity 0/3 = 0.0
    f3 = _make_frame(4, 3.0, [8.0, 8.0, 8.0, 0.0], fails=[3])
    m3 = analyzer.analyze_frame(f3)
    
    assert len(recovers) == 1
    
    print("[PASS] Cascade trigger & hysteresis correct")


def test_hotspot_detection():
    # 4 agents in line: 0-1-2-3
    adj = np.zeros((4,4), dtype=np.uint8)
    adj[0,1] = adj[1,0] = 1
    adj[1,2] = adj[2,1] = 1
    adj[2,3] = adj[3,2] = 1
    
    # Energies:
    # 0 = 100
    # 1 = 10  (LOW)
    # 2 = 100
    # 3 = 10  (LOW but not between two highs, only 1 neighbor=100)
    
    # For agent 1: E=10. Neighbors (0, 2) mean E=(100+100)/2 = 100.
    # Gradient = 10 - 100 = -90. Threshold is < -2.0 & < 0.2*max
    
    f = _make_frame(4, 0.0, [100.0, 10.0, 100.0, 100.0], adj=adj)
    analyzer = EnergyCascadeAnalyzer()
    m = analyzer.analyze_frame(f)
    assert 1 in m.hotspot_indices
    # Agent 0: has neighbor 1 (E=10). Gradient = 100 - 10 = +90. (Safe)
    assert 0 not in m.hotspot_indices
    
    print("[PASS] Spatial hotspot gradients correct")


def test_performance():
    n = 200
    energies = list(np.random.uniform(50, 100, n))
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n-1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
        
    frame = _make_frame(n, 0.0, energies, adj=adj)
    
    analyzer = EnergyCascadeAnalyzer()
    
    # Warmup
    analyzer.analyze_frame(frame)
    
    t0 = time.perf_counter()
    for i in range(10):
        # vary energies slightly
        frame.energies -= 0.1
        frame.time += 0.1
        analyzer.analyze_frame(frame)
    t1 = time.perf_counter()
    
    avg_ms = ((t1 - t0) * 1000) / 10
    
    assert avg_ms < 1.0, f"Performance too slow: {avg_ms:.2f}ms per frame (must be < 1ms)"
    
    print(f"[PASS] Performance tracking: {avg_ms:.3f}ms per frame (Target < 1ms)")


if __name__ == "__main__":
    test_energy_metrics_and_dead_agent_exclusion()
    test_dEdt_and_early_warning()
    test_cascade_detection_and_hysteresis()
    test_hotspot_detection()
    test_performance()
    print("\nAll Phase 2S tests passed.")
