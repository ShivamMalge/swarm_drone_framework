"""
Phase 2C verification: SimulationWorker determinism, thread safety,
time progression, and telemetry streaming.
"""

import sys
import time

from PySide6.QtCore import QCoreApplication, QTimer

from src.core.config import SimConfig
from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.worker import SimulationWorker


# ── Test 1: Determinism (same seed → identical output) ───────

def test_determinism():
    """Two runs with the same seed must produce identical telemetry."""
    cfg = SimConfig(num_agents=10, max_time=1.0, seed=42)

    results = []
    for run_idx in range(2):
        buf = TelemetryBuffer()
        worker = SimulationWorker(
            simulation_config=cfg,
            telemetry_buffer=buf,
            frame_dt=0.05,
        )
        worker.start()
        worker.wait()

        frame = buf.get_latest()
        assert frame is not None, f"Run {run_idx}: no frame produced"
        results.append(frame)

    f0, f1 = results
    assert f0.time == f1.time, f"Time mismatch: {f0.time} vs {f1.time}"
    assert (f0.positions == f1.positions).all(), "Position mismatch"
    assert (f0.energies == f1.energies).all(), "Energy mismatch"
    assert f0.spectral_gap == f1.spectral_gap, "Spectral gap mismatch"
    assert f0.consensus_variance == f1.consensus_variance, "Consensus var mismatch"
    assert f0.packet_drop_rate == f1.packet_drop_rate, "Drop rate mismatch"
    assert f0.regime_state == f1.regime_state, "Regime mismatch"
    print("[PASS] determinism")


# ── Test 2: Time progression ────────────────────────────────

def test_time_progression():
    """Time must advance in fixed frame_dt steps."""
    cfg = SimConfig(num_agents=5, max_time=0.5, seed=7)
    buf = TelemetryBuffer()
    worker = SimulationWorker(
        simulation_config=cfg,
        telemetry_buffer=buf,
        frame_dt=0.05,
    )
    worker.start()
    worker.wait()

    frame = buf.get_latest()
    assert frame is not None
    # max_time=0.5, frame_dt=0.05 → final time should be 0.50
    assert abs(frame.time - 0.50) < 1e-9, f"Expected t=0.50, got {frame.time}"
    print(f"[PASS] time_progression (final_t={frame.time})")


# ── Test 3: Pause / Resume ──────────────────────────────────

def test_pause_resume():
    """Worker should pause and resume without corruption."""
    cfg = SimConfig(num_agents=5, max_time=2.0, seed=99)
    buf = TelemetryBuffer()
    worker = SimulationWorker(
        simulation_config=cfg,
        telemetry_buffer=buf,
        frame_dt=0.05,
    )
    worker.start_simulation()
    time.sleep(0.1)

    worker.pause_simulation()
    assert worker.is_paused
    t_paused = worker.current_time
    time.sleep(0.1)
    assert worker.current_time == t_paused, "Time must not advance while paused"

    worker.resume_simulation()
    assert not worker.is_paused
    time.sleep(0.1)
    assert worker.current_time > t_paused, "Time must advance after resume"

    worker.stop_simulation()
    frame = buf.get_latest()
    assert frame is not None
    print(f"[PASS] pause_resume (paused_at={t_paused:.2f}, final={frame.time:.2f})")


# ── Test 4: Reset ────────────────────────────────────────────

def test_reset():
    """Reset must clear state and allow a fresh run."""
    cfg = SimConfig(num_agents=5, max_time=0.5, seed=1)
    buf = TelemetryBuffer()
    worker = SimulationWorker(
        simulation_config=cfg,
        telemetry_buffer=buf,
        frame_dt=0.05,
    )
    worker.start_simulation()
    worker.wait()

    worker.reset_simulation()
    assert buf.get_latest() is None, "Buffer must be cleared after reset"
    assert worker.current_time == 0.0, "Time must reset to 0"
    print("[PASS] reset")


# ── Test 5: Telemetry stream integrity ───────────────────────

def test_telemetry_integrity():
    """Every frame pushed must have valid shapes and fields."""
    cfg = SimConfig(num_agents=15, max_time=1.0, seed=55)
    buf = TelemetryBuffer()
    worker = SimulationWorker(
        simulation_config=cfg,
        telemetry_buffer=buf,
        frame_dt=0.05,
    )
    worker.start()
    worker.wait()

    frame = buf.get_latest()
    assert frame is not None
    assert frame.positions.shape == (15, 2)
    assert frame.energies.shape == (15,)
    assert frame.adjacency.shape == (15, 15)
    assert frame.drone_failure_flags.shape == (15,)
    assert len(frame.regime_state) == 15
    assert isinstance(frame.adaptive_parameters, dict)
    assert "coverage_gains" in frame.adaptive_parameters
    print("[PASS] telemetry_integrity")


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv)

    test_determinism()
    test_time_progression()
    test_pause_resume()
    test_reset()
    test_telemetry_integrity()

    print("\nAll Phase 2C tests passed.")
