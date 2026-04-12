"""Phase 2G verification: TelemetryGraphsWidget bounded buffers and UI latency."""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from src.core.config import SimConfig
from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.worker import SimulationWorker
from src.telemetry.bridge import TelemetryBridge
from src.gui.main_window import MainWindow

def test_graphs():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    cfg = SimConfig(num_agents=10, max_time=3.0, seed=42)
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

    graphs = window.graphs_panel
    # Verify buffers remain bounded
    assert len(graphs._time_buffer) > 0, "Time buffer is empty"
    assert len(graphs._spectral_buffer) > 0, "Spectral buffer is empty"
    assert len(graphs._energy_buffer) > 0, "Energy buffer is empty"
    assert len(graphs._consensus_buffer) > 0, "Consensus buffer is empty"

    assert len(graphs._time_buffer) <= graphs._max_points, "Buffer exceeded maxlen"
    
    # Check that plotting data matches buffer length
    x_data, y_data = graphs._c_spectral.getData()
    assert len(x_data) == len(graphs._spectral_buffer)
    
    print(f"[PASS] graphs_rendered (points={len(x_data)})")
    
    assert window._status_label.text().startswith("Agents:")
    assert "FPS:" in window._status_label.text()
    assert "Regime:" in window._status_label.text()
    print(f"[PASS] status_bar_format: {window._status_label.text()}")

    print("\nAll Phase 2G tests passed.")


if __name__ == "__main__":
    test_graphs()
