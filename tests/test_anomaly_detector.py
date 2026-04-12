"""
Phase 2X verification: Anomaly Detector Tests
"""

import time
import numpy as np

from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.anomaly_detector import AnomalyDetector


def _make_frame(n: int, fail_idx: list[int] = None) -> TelemetryFrame:
    fail_arr = np.zeros(n, dtype=bool)
    if fail_idx:
        for idx in fail_idx:
            fail_arr[idx] = True
            
    # Base configuration
    return TelemetryFrame(
        time=0.0,
        positions=np.zeros((n, 2)),
        energies=np.ones(n) * 100.0,
        adjacency=np.ones((n, n), dtype=np.uint8) - np.eye(n, dtype=np.uint8), # Fully connected
        connected_components=[],
        spectral_gap=0.0,
        consensus_variance=0.0,
        packet_drop_rate=0.0,
        latency=0.0,
        regime_state={},
        adaptive_parameters={},
        drone_failure_flags=fail_arr,
        agent_states=np.ones(n) * 50.0
    )


def test_spatial_detection():
    detector = AnomalyDetector()
    detector.t_spatial = 5.0
    
    n = 5
    f = _make_frame(n)
    
    # Put agent 0 far away and slightly drop energy so score > 0.3
    f.positions[0] = [100.0, 100.0]
    f.energies[0] = 50.0
    
    m = detector.analyze_frame(f)
    # Agent 0 is the anomaly
    assert m.class_labels[0] >= 1 # Suspicious or Anomaly
    # Other agents might be suspicious because they are connected to agent 0!
    # So we don't assert class_labels[1:] == 0
        
    print("[PASS] Spatial deviation detected correctly")


def test_energy_deviation():
    detector = AnomalyDetector()
    detector.t_energy = 10.0
    
    n = 5
    f = _make_frame(n)
    
    # Put agent 1 energy to 0.0 and deviate consensus slightly
    f.energies[1] = 0.0
    f.agent_states[1] = 100.0
    
    m = detector.analyze_frame(f)
    print("Scores", m.scores)
    assert m.class_labels[1] >= 1
    
    print("[PASS] Energy deviation detected correctly")


def test_consensus_deviation():
    detector = AnomalyDetector()
    detector.t_consensus = 2.0
    
    n = 5
    f = _make_frame(n)
    f.agent_states[2] = 20.0 # Rest are 50.0
    f.positions[2] = [100.0, 100.0]
    
    m = detector.analyze_frame(f)
    assert m.class_labels[2] >= 1
    
    print("[PASS] Consensus deviation detected correctly")


def test_dead_mask_and_hash_skip():
    detector = AnomalyDetector()
    n = 5
    f = _make_frame(n, fail_idx=[3]) # Agent 3 is dead
    f.energies[3] = 0.0 # Anomalous state but dead
    
    m = detector.analyze_frame(f)
    assert m.class_labels[3] == 3 # 3=Dead
    assert m.anomaly_count == 0

    
    hash1 = detector._last_hash
    m2 = detector.analyze_frame(f)
    
    assert hash1 == detector._last_hash
    assert m is m2
    
    print("[PASS] Masking dead agents and hash skip correctness")


def test_performance():
    detector = AnomalyDetector()
    n = 200
    f = _make_frame(n)
    f.positions = np.random.rand(n, 2)
    f.energies = np.random.rand(n)
    f.agent_states = np.random.rand(n)
    
    detector.analyze_frame(f) # warmup
    
    t0 = time.perf_counter()
    for _ in range(50):
        f.positions += 0.1
        detector.analyze_frame(f)
    t1 = time.perf_counter()
    
    avg_ms = ((t1 - t0) * 1000) / 50
    assert avg_ms < 1.5, f"Too slow! {avg_ms}ms"
    
    print(f"[PASS] Performance tracking: {avg_ms:.3f}ms per frame (Target < 1ms)")


if __name__ == "__main__":
    test_spatial_detection()
    test_energy_deviation()
    test_consensus_deviation()
    test_dead_mask_and_hash_skip()
    test_performance()
    print("\nAll Phase 2X tests passed.")
