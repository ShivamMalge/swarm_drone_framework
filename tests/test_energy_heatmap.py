"""
Phase 2AC verification: Energy Heatmap Mapper tests.
"""

import time
import numpy as np

from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.energy_heatmap_mapper import EnergyHeatmapMapper


# ── Helpers ─────────────────────────────────────────────

def _frame(n: int, energies: np.ndarray,
           adj: np.ndarray | None = None,
           fail: list[int] | None = None) -> TelemetryFrame:
    if adj is None:
        adj = np.ones((n, n), dtype=np.uint8) - np.eye(n, dtype=np.uint8)
        
    fail_arr = np.zeros(n, dtype=bool)
    if fail:
        for i in fail:
            fail_arr[i] = True
            
    return TelemetryFrame(
        time=0.0,
        positions=np.random.rand(n, 2) * 100,
        energies=energies,
        adjacency=adj,
        connected_components=[],
        spectral_gap=0.0,
        consensus_variance=0.0,
        packet_drop_rate=0.0,
        latency=0.0,
        regime_state={},
        adaptive_parameters={},
        drone_failure_flags=fail_arr,
        agent_states=np.ones(n) * 50.0,
    )


# ── Tests ───────────────────────────────────────────────

def test_normalization_correctness():
    mapper = EnergyHeatmapMapper()
    energies = np.array([10.0, 50.0, 100.0, 50.0, 10.0])
    # K5 graph
    f = _frame(5, energies)
    
    r = mapper.map_energies(f)
    # min=10, max=100 -> range=90
    # 10 -> 0.0, 50 -> 40/90 = 0.444, 100 -> 1.0
    assert r.energy_norm[0] == 0.0
    assert abs(r.energy_norm[1] - 0.4444) < 1e-3
    assert r.energy_norm[2] == 1.0
    
    print(f"[PASS] Normalization correct: {r.energy_norm}")


def test_gradient_correctness():
    """Verify local topological gradients."""
    mapper = EnergyHeatmapMapper()
    # Star graph: node 0 connected to 1,2,3
    adj = np.zeros((4, 4), dtype=np.uint8)
    adj[0, 1:] = 1
    adj[1:, 0] = 1
    
    # 0 = 90 (center), others = 10, 10, 10
    energies = np.array([90.0, 10.0, 10.0, 10.0])
    f = _frame(4, energies, adj)
    
    r = mapper.map_energies(f)
    
    # Node 0 neighbor mean: (10+10+10)/3 = 10 -> E - mean = 80
    # Node 1 neighbor mean: 90/1 = 90 -> E - mean = 80
    # So max gradient is 80. Normalized gradient should be 1.0 for all!
    assert abs(r.gradient_norm[0] - 1.0) < 1e-6
    assert abs(r.gradient_norm[1] - 1.0) < 1e-6
    print(f"[PASS] Gradient correctness: {r.gradient_norm}")


def test_hotspot_detection():
    """Hotspot triggers if G_norm > 0.6 AND E_norm > 0.5."""
    mapper = EnergyHeatmapMapper()
    # Two nodes connected
    adj = np.zeros((2, 2), dtype=np.uint8)
    adj[0, 1] = adj[1, 0] = 1
    
    # Node 0 has high energy, Node 1 has very low
    energies = np.array([100.0, 0.0])
    f = _frame(2, energies, adj)
    
    r = mapper.map_energies(f)
    
    # Node 0: E_norm = 1.0. neighbor mean = 0.0 -> G = 100 -> G_norm = 1.0
    # Node 1: E_norm = 0.0. neighbor mean = 100.0 -> G = 100 -> G_norm = 1.0
    
    # Hotspot mask requires BOTH: G_norm > 0.6 AND E_norm > 0.5
    # Node 0: G_norm=1.0 > 0.6 (True) AND E_norm=1.0 > 0.5 (True) -> True
    # Node 1: G_norm=1.0 > 0.6 (True) AND E_norm=0.0 > 0.5 (False) -> False
    assert r.hotspots[0] == True
    assert r.hotspots[1] == False
    
    print("[PASS] Hotspot detection (G_norm > 0.6 AND E_norm > 0.5)")


def test_dead_agent_masking():
    mapper = EnergyHeatmapMapper()
    adj = np.ones((4, 4), dtype=np.uint8) - np.eye(4, dtype=np.uint8)
    energies = np.array([50.0, 50.0, 1000.0, 50.0])
    
    # Node 2 is dead, despite having huge energy
    f = _frame(4, energies, adj, fail=[2])
    
    r = mapper.map_energies(f)
    
    # Node 2 should be completely zeroed out in outputs
    assert r.energy_norm[2] == 0.0
    assert r.gradient_norm[2] == 0.0
    assert r.cascade_intensity[2] == 0.0
    assert not r.hotspots[2]
    
    # The normalization should ignore 1000.0, treating max as 50.0
    # Resulting in remaining nodes having uniform 0.0 normalization (max=min=50)
    assert r.energy_norm[0] == 0.0
    
    print("[PASS] Dead agent exclusion correctly drops outliers")


def test_no_nans():
    mapper = EnergyHeatmapMapper()
    # All energies identical (causes division by 0 if not handled)
    f_same = _frame(5, np.ones(5) * 50.0)
    r_same = mapper.map_energies(f_same)
    
    assert not np.any(np.isnan(r_same.energy_norm))
    assert not np.any(np.isnan(r_same.gradient_norm))
    assert not np.any(np.isnan(r_same.cascade_intensity))
    
    # Completely disconnected graph (degree 0 -> div by 0 for neighbor mean)
    adj_zero = np.zeros((3, 3), dtype=np.uint8)
    f_zero = _frame(3, np.array([10.0, 20.0, 30.0]), adj_zero)
    r_zero = mapper.map_energies(f_zero)
    
    assert not np.any(np.isnan(r_zero.energy_norm))
    assert not np.any(np.isnan(r_zero.gradient_norm))
    
    print("[PASS] No NaNs in edge cases (constant energy, disconnected)")


def test_stability_over_time():
    """Hash-skip maintains exact objects if nothing changes."""
    mapper = EnergyHeatmapMapper()
    f = _frame(10, np.random.rand(10) * 100)
    
    r1 = mapper.map_energies(f)
    # Different frame but identical telemetry arrays -> hash hits
    f2 = _frame(10, f.energies, f.adjacency)
    
    r2 = mapper.map_energies(f2)
    assert r1 is r2
    
    print("[PASS] Stable caching via hash-skip")


def test_determinism():
    f = _frame(20, np.random.rand(20) * 100)
    
    m1 = EnergyHeatmapMapper()
    m2 = EnergyHeatmapMapper()
    
    r1 = m1.map_energies(f)
    r2 = m2.map_energies(f)
    
    np.testing.assert_array_equal(r1.cascade_intensity, r2.cascade_intensity)
    np.testing.assert_array_equal(r1.hotspots, r2.hotspots)
    
    print("[PASS] Deterministic multi-instance mappings")


def test_performance():
    mapper = EnergyHeatmapMapper()
    n = 200
    adj = np.ones((n, n), dtype=np.uint8) - np.eye(n, dtype=np.uint8)
    # Add random drops
    adj[np.random.rand(n, n) > 0.5] = 0
    adj = np.maximum(adj, adj.T)
    f = _frame(n, np.random.rand(n) * 100, adj)
    
    # Warm up
    mapper.map_energies(f)
    
    t0 = time.perf_counter()
    for _ in range(100):
        # Bust cache by changing energies
        f.energies = np.random.rand(n) * 100
        mapper.map_energies(f)
    t1 = time.perf_counter()
    
    avg_ms = ((t1 - t0) * 1000) / 100
    assert avg_ms < 1.0, f"Too slow: {avg_ms:.2f}ms"
    print(f"[PASS] Performance: {avg_ms:.3f}ms per frame (N={n}, target < 1ms)")


if __name__ == "__main__":
    test_normalization_correctness()
    test_gradient_correctness()
    test_hotspot_detection()
    test_dead_agent_masking()
    test_no_nans()
    test_stability_over_time()
    test_determinism()
    test_performance()
    print("\nAll Phase 2AC tests passed.")
