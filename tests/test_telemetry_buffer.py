"""Phase 2B verification: TelemetryBuffer thread-safety and latest-frame semantics."""

import threading
import time

from src.core.config import SimConfig
from src.simulation import Phase1Simulation
from src.telemetry.telemetry_emitter import TelemetryEmitter
from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.telemetry_frame import TelemetryFrame


def test_single_slot_overwrite():
    """Push multiple frames; get_latest always returns the last one."""
    buf = TelemetryBuffer()

    assert buf.get_latest() is None, "Empty buffer must return None"
    assert not buf.has_new_data, "Empty buffer has no new data"

    f1 = TelemetryFrame.empty(5)
    f2 = TelemetryFrame.empty(5)
    # Differentiate by time
    object.__setattr__(f1, "time", 1.0)
    object.__setattr__(f2, "time", 2.0)

    buf.push(f1)
    assert buf.has_new_data
    buf.push(f2)  # overwrites f1

    latest = buf.get_latest()
    assert latest is not None
    assert latest.time == 2.0, f"Expected 2.0, got {latest.time}"
    assert not buf.has_new_data, "Flag should reset after get_latest()"

    print("[PASS] single_slot_overwrite")


def test_clear():
    buf = TelemetryBuffer()
    buf.push(TelemetryFrame.empty(3))
    buf.clear()
    assert buf.get_latest() is None
    assert not buf.has_new_data
    print("[PASS] clear")


def test_thread_safety():
    """Concurrent push/get across threads — no crashes, latest frame wins."""
    cfg = SimConfig(num_agents=10, max_time=5.0, seed=99)
    sim = Phase1Simulation(cfg)
    sim.seed_events()
    emitter = TelemetryEmitter(sim, cfg)
    buf = TelemetryBuffer()

    push_count = 0
    read_count = 0
    errors = []

    def producer():
        nonlocal push_count
        for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
            sim.kernel.run(until=t)
            frame = emitter.extract_frame()
            buf.push(frame)
            push_count += 1
            time.sleep(0.001)

    def consumer():
        nonlocal read_count
        for _ in range(20):
            f = buf.get_latest()
            if f is not None:
                # Validate frame integrity
                if f.positions.shape != (10, 2):
                    errors.append(f"Bad shape: {f.positions.shape}")
                read_count += 1
            time.sleep(0.002)

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Thread errors: {errors}"
    assert push_count == 5
    print(f"[PASS] thread_safety (pushed={push_count}, read={read_count})")

    # Final frame must be the last pushed
    final = buf.get_latest()
    assert final is not None
    assert final.time == 5.0, f"Expected t=5.0, got {final.time}"
    print("[PASS] latest_frame_priority")


if __name__ == "__main__":
    test_single_slot_overwrite()
    test_clear()
    test_thread_safety()
    print("\nAll Phase 2B tests passed.")
