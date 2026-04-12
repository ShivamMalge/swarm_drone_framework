"""
Phase 2Y verification: Research Metrics Engine tests
"""

import time
import numpy as np
import os

from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.percolation_analyzer import PercolationMetrics
from src.analytics.spectral_analyzer import SpectralMetrics
from src.analytics.energy_cascade_analyzer import EnergyMetrics
from src.analytics.consensus_analyzer import ConsensusMetrics
from src.analytics.anomaly_detector import AnomalyMetrics
from src.analytics.swarm_health import HealthMetrics
from src.analytics.research_metrics import ResearchMetricsEngine

def _make_mock_metrics():
    frame = TelemetryFrame(time=0.0, positions=np.zeros(1), energies=np.zeros(1),
                           adjacency=np.zeros((1, 1), dtype=np.uint8), connected_components=[],
                           spectral_gap=0.0, consensus_variance=0.0, packet_drop_rate=0.0,
                           latency=0.0, regime_state={}, adaptive_parameters={},
                           drone_failure_flags=np.zeros(1, dtype=bool), agent_states=np.zeros(1))
    
    perc = PercolationMetrics(1, 1, 1, 1.0, 1.0, 0.5, 0.0, "STABLE")
    spec = SpectralMetrics(1.0, 1.0, 0.5, 0.0, "STABLE")
    eng = EnergyMetrics(100.0, 100.0, 100.0, 0.0, 1, 0, 0, 0.0, 0.0, 1.0, 0.5, 0.0, np.zeros(1), np.array([]), [], "STABLE")
    cns = ConsensusMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, [], "STABLE")
    anom = AnomalyMetrics(np.zeros(1), np.zeros(1), np.zeros(1, dtype=np.uint8), 0, 0, 0)
    health = HealthMetrics(1.0, 1.0, "HEALTHY")
    return frame, perc, spec, eng, cns, anom, health

def test_aggregation():
    engine = ResearchMetricsEngine("test_scen", 42)
    
    # Send normal frame
    frame, perc, spec, eng, cns, anom, health = _make_mock_metrics()
    frame.time = 0.0
    m = engine.add_frame_metrics(frame, perc, spec, eng, cns, anom, health)
    
    # 0 time passed
    assert m.total_time == 0.0
    assert m.stability.time_in_stable == 0.0
    assert m.total_frames == 1
    
    # Send another frame with dt = 1.0
    frame.time = 1.0
    eng.state = "CASCADE" # Introduce cascade correctly identifying it
    anom.anomaly_count = 5 
    
    m2 = engine.add_frame_metrics(frame, perc, spec, eng, cns, anom, health)
    
    assert m2.total_time == 1.0
    assert m2.stability.time_in_stable == 1.0 # 1 sec in stable
    assert m2.energy.cascade_frequency == 1 # edge triggered
    assert m2.anomaly.total_anomalies == 5
    assert m2.anomaly.peak_anomalies == 5
    
    print("[PASS] Aggregation functions correctly evaluating bounded temporal metrics")

def test_json_export():
    engine = ResearchMetricsEngine()
    frame, perc, spec, eng, cns, anom, health = _make_mock_metrics()
    engine.add_frame_metrics(frame, perc, spec, eng, cns, anom, health)
    
    engine.export_json("test_out.json")
    assert os.path.exists("test_out.json")
    os.remove("test_out.json")
    
    print("[PASS] JSON export successfully writes flat dictionaries mappings")

def test_performance():
    engine = ResearchMetricsEngine()
    frame, perc, spec, eng, cns, anom, health = _make_mock_metrics()
    
    engine.add_frame_metrics(frame, perc, spec, eng, cns, anom, health) # warm up
    
    t0 = time.perf_counter()
    for _ in range(50):
        frame.time += 0.5
        engine.add_frame_metrics(frame, perc, spec, eng, cns, anom, health)
    t1 = time.perf_counter()
    
    avg_ms = ((t1 - t0) * 1000) / 50
    assert avg_ms < 1.0, f"Too slow! {avg_ms:.3f}ms"
    print(f"[PASS] Performance tracking: {avg_ms:.4f}ms per call (Target < 1ms)")


if __name__ == "__main__":
    test_aggregation()
    test_json_export()
    test_performance()
    print("\nAll Phase 2Y tests passed.")
