"""
Phase 2W verification: Swarm Health Index fusion logic tests.
"""

import time
import numpy as np

from src.analytics.percolation_analyzer import PercolationMetrics
from src.analytics.spectral_analyzer import SpectralMetrics
from src.analytics.energy_cascade_analyzer import EnergyMetrics
from src.analytics.consensus_analyzer import ConsensusMetrics
from src.analytics.swarm_health import SwarmHealthAnalyzer


def _mock_metrics(n_perc=1.0, n_spec=1.0, n_eng=1.0, n_var=0.0,
                  dp=0.0, ds=0.0, de=0.0, dc=0.0,
                  sp="STABLE", se="STABLE", sc="CONVERGED"):
    return (
        PercolationMetrics(lcc_size=10, total_agents=10, num_components=1, connectivity_ratio=1.0, 
                           normalized_ratio=n_perc, connectivity_margin=n_perc - 0.5, d_ratio_dt=dp, state=sp),
        SpectralMetrics(lambda2=2.0, lambda2_normalized=n_spec, spectral_margin=n_spec - 0.1, dl2_dt=ds, state=sp),
        EnergyMetrics(mean_energy=100.0, smoothed_mean_energy=100.0, min_energy=0.0, energy_variance=0.0,
                      alive_count=10, failure_count=0, new_failures=0, cascade_intensity=0.0, smoothed_intensity=0.0,
                      normalized_energy=n_eng, cascade_margin=0.0 - 0.1, d_mean_energy_dt=de,
                      energy_gradients=np.zeros(1), dead_agents=np.zeros(0), hotspot_indices=[], state=se),
        ConsensusMetrics(consensus_variance=0.0, consensus_error=0.0, normalized_variance=n_var, 
                         smoothed_variance=n_var, smoothed_rate=dc, consensus_margin=n_var - 0.05, 
                         global_convergence_rate=dc, hotspot_indices=[], state=sc)
    )


def test_base_health_weights():
    analyzer = SwarmHealthAnalyzer()
    analyzer.w_perc = 0.3
    analyzer.w_spec = 0.2
    analyzer.w_eng = 0.3
    analyzer.w_cns = 0.2
    
    # All perfect
    perc, spec, eng, cns = _mock_metrics(1.0, 1.0, 1.0, 0.0)
    m = analyzer.analyze(perc, spec, eng, cns, 1.0)
    
    assert m.raw_score == 1.0
    assert m.state == "HEALTHY"
    
    # Terrible state
    perc, spec, eng, cns = _mock_metrics(0.0, 0.0, 0.0, 1.0)
    m2 = analyzer.analyze(perc, spec, eng, cns, 2.0)
    
    # 0.3*0 + 0.2*0 + 0.3*0 + 0.2*(1-1)
    assert m2.raw_score == 0.0
    assert m2.state == "COLLAPSE"
    
    print("[PASS] Weighted base calculation correct")


def test_rate_penalties():
    analyzer = SwarmHealthAnalyzer()
    
    # Dropping logic:
    # p_perc = min(0, dp) * -0.5
    # p_spec = min(0, ds) * -0.5
    # p_eng = min(0, de) * -0.1
    # p_cns = max(0, dc) * 0.1
    
    # Provide perfect inputs but terrible velocities
    perc, spec, eng, cns = _mock_metrics(1.0, 1.0, 1.0, 0.0,
                                         dp=-0.4, ds=-0.4, de=-1.0, dc=2.0)
    
    # Penalty limits clamp at 0.5 maximum.
    # dp=-0.4 => 0.2
    # ds=-0.4 => 0.2    (0.4)
    # de=-1.0 => 0.1    (0.5)
    # dc=2.0 => 0.2     (0.7 -> clamped to 0.5)
    
    m = analyzer.analyze(perc, spec, eng, cns, 1.0)
    # Expected: 1.0 - 0.5 = 0.5
    assert m.raw_score == 0.5
    
    print("[PASS] Rate penalties correctly clamped and scaled")


def test_event_overrides():
    analyzer = SwarmHealthAnalyzer()
    
    # Test PERCOLATION COLLAPSE 
    # Force state string.
    perc, spec, eng, cns = _mock_metrics(1.0, 1.0, 1.0, 0.0, sp="COLLAPSE")
    m = analyzer.analyze(perc, spec, eng, cns, 1.0)
    assert m.raw_score <= 0.15
    assert m.state == "COLLAPSE"
    
    # Test ENERGY CASCADE
    analyzer = SwarmHealthAnalyzer() # reset EMA
    perc, spec, eng, cns = _mock_metrics(1.0, 1.0, 1.0, 0.0, se="CASCADE")
    m2 = analyzer.analyze(perc, spec, eng, cns, 2.0)
    assert m2.raw_score <= 0.5
    assert m2.state == "CRITICAL" or m2.state == "COLLAPSE" or m2.state == "DEGRADING"
    
    print("[PASS] Hard event logic overrides bounds correctly")


def test_hash_skip_correctness():
    analyzer = SwarmHealthAnalyzer()
    
    perc, spec, eng, cns = _mock_metrics(1.0, 1.0, 1.0, 0.0)
    m1 = analyzer.analyze(perc, spec, eng, cns, 1.0)
    
    hash1 = analyzer._last_hash
    
    m2 = analyzer.analyze(perc, spec, eng, cns, 2.0) # Same inputs, diff time
    hash2 = analyzer._last_hash
    
    assert hash1 == hash2
    assert m1 is m2
    
    perc2, spec2, eng2, cns2 = _mock_metrics(0.9, 1.0, 1.0, 0.0)
    m3 = analyzer.analyze(perc2, spec2, eng2, cns2, 3.0)
    assert analyzer._last_hash != hash2
    
    print("[PASS] Hash-skip identical recompute logic correct")


def test_performance():
    analyzer = SwarmHealthAnalyzer()
    perc, spec, eng, cns = _mock_metrics(1.0, 1.0, 1.0, 0.0)
    
    # Warmup
    analyzer.analyze(perc, spec, eng, cns, 0.0)
    
    t0 = time.perf_counter()
    for i in range(100):
        # vary slightly to avoid hash skip entirely
        perc.normalized_ratio -= 0.001
        analyzer.analyze(perc, spec, eng, cns, float(i))
        
    t1 = time.perf_counter()
    avg_ms = ((t1 - t0) * 1000) / 100
    
    assert avg_ms < 1.0, f"Execution time too high: {avg_ms:.2f}ms"
    
    print(f"[PASS] Performance tracking: {avg_ms:.4f}ms per fusion (Target < 1ms)")


if __name__ == "__main__":
    test_base_health_weights()
    test_rate_penalties()
    test_event_overrides()
    test_hash_skip_correctness()
    test_performance()
    print("\nAll Phase 2W tests passed.")
