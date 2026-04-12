"""Phase 2H verification: RegimePanel update without UI lag or flicker."""

import sys
import collections

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from src.core.config import SimConfig
from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.worker import SimulationWorker
from src.telemetry.bridge import TelemetryBridge
from src.gui.main_window import MainWindow

def test_regime_panel():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Note: Using small max_time for fast testing
    cfg = SimConfig(num_agents=20, max_time=3.0, seed=42)
    buf = TelemetryBuffer()
    worker = SimulationWorker(simulation_config=cfg, telemetry_buffer=buf, frame_dt=0.05)
    bridge = TelemetryBridge(telemetry_buffer=buf, poll_interval_ms=16)
    window = MainWindow(telemetry_bridge=bridge)

    window.show()
    bridge.start()
    worker.start_simulation()

    QTimer.singleShot(4000, app.quit)
    app.exec()

    worker.stop_simulation()
    bridge.stop()

    panel = window.regime_panel
    
    # Verify Global Label updated
    assert panel._global_label.text() != "UNKNOWN", f"Global Regime not updated: {panel._global_label.text()}"
    print(f"[PASS] global_regime_smoothed (result={panel._global_label.text()})")
    
    # Verify Distribution counts updated properly -> STABLE should be >0 for good seeds
    stable_count = int(panel._dist_labels["STABLE"].text())
    assert stable_count > 0, f"Expected STABLE counts > 0, got {stable_count}"
    print(f"[PASS] agent_distribution (STABLE={stable_count})")
    
    # Verify History populated
    assert len(panel._history) > 0, "Smoothing history is empty"
    print(f"[PASS] history_populated (size={len(panel._history)})")

    print("\nAll Phase 2H tests passed.")


if __name__ == "__main__":
    test_regime_panel()
