"""Phase 2E verification: MainWindow launch, layout, signal connection."""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from src.core.config import SimConfig
from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.worker import SimulationWorker
from src.telemetry.bridge import TelemetryBridge
from src.gui.main_window import MainWindow


def test_main_window():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    cfg = SimConfig(num_agents=10, max_time=1.0, seed=42)
    buf = TelemetryBuffer()
    worker = SimulationWorker(simulation_config=cfg, telemetry_buffer=buf, frame_dt=0.05)
    bridge = TelemetryBridge(telemetry_buffer=buf, poll_interval_ms=16)
    window = MainWindow(telemetry_bridge=bridge)

    # 1. Window launches
    window.show()
    assert window.isVisible(), "Window not visible"
    print("[PASS] window_launch")

    # 2. Layout placeholders exist
    assert window.swarm_map_placeholder is not None
    assert window.regime_placeholder is not None
    assert window.adaptive_placeholder is not None
    assert window.spectral_placeholder is not None
    assert window.energy_placeholder is not None
    assert window.consensus_placeholder is not None
    print("[PASS] layout_structure")

    # 3. Signal triggers update_frame
    bridge.start()
    worker.start_simulation()

    QTimer.singleShot(2000, app.quit)
    app.exec()

    worker.stop_simulation()
    bridge.stop()

    assert window._frame_count > 0, f"No frames received: {window._frame_count}"
    print(f"[PASS] signal_connected (frames={window._frame_count})")

    # 4. Responsive (if we got here, no freeze)
    print("[PASS] no_blocking")

    print("\nAll Phase 2E tests passed.")


if __name__ == "__main__":
    test_main_window()
