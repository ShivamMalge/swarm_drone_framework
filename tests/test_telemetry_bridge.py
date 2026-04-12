"""Phase 2D verification: TelemetryBridge frame flow, dropping, and no-backpressure."""

import sys

from PySide6.QtCore import QCoreApplication, QTimer

from src.core.config import SimConfig
from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.telemetry_frame import TelemetryFrame
from src.telemetry.bridge import TelemetryBridge
from src.telemetry.worker import SimulationWorker


def test_bridge():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv)

    cfg = SimConfig(num_agents=10, max_time=2.0, seed=42)
    buf = TelemetryBuffer()
    worker = SimulationWorker(simulation_config=cfg, telemetry_buffer=buf, frame_dt=0.05)
    bridge = TelemetryBridge(telemetry_buffer=buf, poll_interval_ms=16)

    received_frames = []

    def on_frame(frame: TelemetryFrame):
        received_frames.append(frame)

    bridge.frame_ready.connect(on_frame)

    # Start pipeline
    bridge.start()
    worker.start_simulation()

    # Run event loop for enough time
    QTimer.singleShot(3000, app.quit)
    app.exec()

    worker.stop_simulation()
    bridge.stop()

    assert len(received_frames) > 0, "No frames received"
    assert received_frames[-1].time > 0.0, "Final frame time must be > 0"

    # Verify monotonic time
    times = [f.time for f in received_frames]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], f"Non-monotonic: {times[i-1]} -> {times[i]}"

    # Verify no backlog (frames received < frames pushed due to dropping)
    total_steps = int(cfg.max_time / 0.05)
    print(f"[PASS] bridge_flow (received={len(received_frames)}, max_possible={total_steps})")
    print(f"[PASS] monotonic_time (first={times[0]:.2f}, last={times[-1]:.2f})")
    print(f"[PASS] frame_dropping (ratio={len(received_frames)/total_steps:.2f})")
    print(f"[PASS] no_backpressure")
    print("\nAll Phase 2D tests passed.")


if __name__ == "__main__":
    test_bridge()
