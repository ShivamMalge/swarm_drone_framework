"""
Phase 2T verification: Consensus Analyzer tests.
"""

import time
import numpy as np
from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.consensus_analyzer import ConsensusAnalyzer


def _make_frame(n: int, t: float, states: list[float], fails: list[int] = None, adj: np.ndarray = None) -> TelemetryFrame:
    if adj is None:
        adj = np.zeros((n, n), dtype=np.uint8)
        
    fail_arr = np.zeros(n, dtype=bool)
    if fails:
        for f in fails:
            fail_arr[f] = True

    return TelemetryFrame(
        time=float(t),
        positions=np.zeros((n, 2)),
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
        agent_states=np.array(states, dtype=np.float64)
    )


def test_variance_and_dead_agent_exclusion():
    # 3 agents. #2 is dead.
    states = [10.0, 20.0, 100.0]  # If 100 was included variance would be huge
    frame = _make_frame(3, 0.0, states, fails=[2])
    
    analyzer = ConsensusAnalyzer()
    metrics = analyzer.analyze_frame(frame)
    
    # Alive states: 10, 20. Mean=15. Var=25
    assert metrics.consensus_variance == 25.0
    # Error: mean(|x_i - mean(x)|) = mean(|10-15|, |20-15|) = 5
    assert metrics.consensus_error == 5.0
    
    print("[PASS] Variance & Error metric correctness with dead agent exclusion")


def test_convergence_and_divergence_rates():
    analyzer = ConsensusAnalyzer(convergence_threshold=0.1)
    
    converged = []
    diverged = []
    analyzer.consensus_converged.connect(lambda t: converged.append(t))
    analyzer.consensus_diverging.connect(lambda t: diverged.append(t))
    
    # Frame 0: High variance
    f0 = _make_frame(2, 0.0, [0.0, 10.0]) # Mean=5, Var=25
    analyzer.analyze_frame(f0)
    
    # Frame 1: Decreased variance
    f1 = _make_frame(2, 1.0, [2.0, 8.0]) # Mean=5, Var=9
    m1 = analyzer.analyze_frame(f1)
    
    assert m1.global_convergence_rate == -16.0
    assert m1.state == "SLOW_CONVERGENCE"
    
    # Frame 2: Increased variance
    f2 = _make_frame(2, 2.0, [0.0, 10.0]) # Mean=5, Var=25
    m2 = analyzer.analyze_frame(f2)
    
    assert m2.global_convergence_rate == 16.0
    assert m2.state == "DIVERGING"
    assert len(diverged) == 1
    
    # Frame 3: Converged
    f3 = _make_frame(2, 3.0, [5.0, 5.0]) # Var = 0 < 0.1
    m3 = analyzer.analyze_frame(f3)
    
    assert m3.state == "CONVERGED"
    assert len(converged) == 1
    
    print("[PASS] Convergence rates, diverging, and converging events")


def test_oscillation_detection():
    analyzer = ConsensusAnalyzer()
    events = []
    analyzer.consensus_oscillating.connect(lambda t: events.append(t))
    
    var_seq = [10.0, 11.0, 9.0, 12.0, 8.0, 13.0]
    
    for i, v in enumerate(var_seq):
        f = _make_frame(1, float(i), [v]) # Var of 1 agent is 0. Wait, need 2 agents to control var easily:
        # Actually doing [0, v] has var = v^2 / 4
        # Just use [10-d, 10+d] so Var = d^2
        d = np.sqrt(v)
        f = _make_frame(2, float(i), [10.0 - d, 10.0 + d])
        metrics = analyzer.analyze_frame(f)
        
    assert len(events) >= 1
    assert metrics.state == "OSCILLATING"
    
    print("[PASS] Oscillation sign-flip detection correct")


def test_hotspot_detection():
    adj = np.zeros((4,4), dtype=np.uint8)
    adj[0,1] = adj[1,0] = 1
    adj[1,2] = adj[2,1] = 1
    adj[2,3] = adj[3,2] = 1
    
    f = _make_frame(4, 0.0, [100.0, 0.0, 100.0, 100.0], adj=adj)
    analyzer = ConsensusAnalyzer()
    m = analyzer.analyze_frame(f)
    print(m.hotspot_indices)
    assert 1 in m.hotspot_indices
    assert 0 in m.hotspot_indices
    
    print("[PASS] Spatial hotspot gradients correct")


def test_performance():
    n = 200
    states = list(np.random.uniform(0, 100, n))
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n-1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
        
    frame = _make_frame(n, 0.0, states, adj=adj)
    
    analyzer = ConsensusAnalyzer()
    
    # Warmup
    analyzer.analyze_frame(frame)
    
    t0 = time.perf_counter()
    for i in range(10):
        frame.agent_states += 0.1
        frame.time += 0.1
        analyzer.analyze_frame(frame)
    t1 = time.perf_counter()
    
    avg_ms = ((t1 - t0) * 1000) / 10
    
    assert avg_ms < 1.0, f"Performance too slow: {avg_ms:.2f}ms per frame (must be < 1ms)"
    
    print(f"[PASS] Performance tracking: {avg_ms:.3f}ms per frame (Target < 1ms)")


if __name__ == "__main__":
    test_variance_and_dead_agent_exclusion()
    test_convergence_and_divergence_rates()
    test_oscillation_detection()
    test_hotspot_detection()
    test_performance()
    print("\nAll Phase 2T tests passed.")
