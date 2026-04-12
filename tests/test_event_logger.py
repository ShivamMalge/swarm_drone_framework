"""
Phase 2V verification: Event Logger tests.
"""

import time
from src.analytics.event_logger import EventLogger


def test_severity_mapping():
    logger = EventLogger()
    events = []
    logger.event_logged.connect(lambda e: events.append(e))
    
    logger.log_event("percolation_collapse_detected", "perc", 1.0)
    logger.log_event("spectral_instability_detected", "spec", 2.0)
    logger.log_event("consensus_converged", "cons", 3.0)
    logger.log_event("random_info_event", "mod", 4.0)
    
    assert events[0].severity == "CRITICAL"
    assert events[1].severity == "WARNING"
    assert events[2].severity == "RECOVERY"
    assert events[3].severity == "INFO"
    
    print("[PASS] Severity mapping correct")


def test_deduplication():
    logger = EventLogger(dedup_window=2.0)
    events = []
    logger.event_logged.connect(lambda e: events.append(e))
    
    # First triggers
    logger.log_event("energy_cascade_detected", "energy", 1.0)
    assert len(events) == 1
    
    # Duplicate within 2s window
    logger.log_event("energy_cascade_detected", "energy", 1.5)
    logger.log_event("energy_cascade_detected", "energy", 2.0)
    assert len(events) == 1  # Suppressed!
    
    # Wait until outside window
    logger.log_event("energy_cascade_detected", "energy", 3.5) # 3.5 - 1.0 = 2.5 > 2.0
    assert len(events) == 2
    
    # Different event entirely doesn't get suppressed
    logger.log_event("spectral_instability_detected", "spec", 3.6)
    assert len(events) == 3
    
    print("[PASS] Deduplication logic correct (temporal suppression)")


def test_buffer_overwrite():
    logger = EventLogger(max_size=3, dedup_window=0.0) # no dedup
    for i in range(5):
        logger.log_event(f"event_{i}", "mod", float(i))
        
    assert len(logger.events) == 3
    assert logger.events[0].event_type == "event_2"
    assert logger.events[2].event_type == "event_4"
    
    print("[PASS] Fixed-size buffer overwrite bounds correct")


def test_performance():
    logger = EventLogger()
    
    t0 = time.perf_counter()
    for i in range(100):
        # vary type slightly to prevent dedup
        logger.log_event(f"event_{i}", "mod", float(i))
    t1 = time.perf_counter()
    
    avg_ms = ((t1 - t0) * 1000) / 100
    assert avg_ms < 0.2, f"Too slow: {avg_ms} ms per event"
    
    print(f"[PASS] Performance tracking: {avg_ms:.4f}ms per event (Target < 0.2ms)")


if __name__ == "__main__":
    test_severity_mapping()
    test_deduplication()
    test_buffer_overwrite()
    test_performance()
    print("\nAll Phase 2V tests passed.")
