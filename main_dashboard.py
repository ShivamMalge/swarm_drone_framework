"""
Dashboard entry point.

Wires: SimConfig → Worker → Buffer → Bridge → MainWindow
"""

import sys

from PySide6.QtWidgets import QApplication

from src.core.config import SimConfig
from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.worker import SimulationWorker
from src.telemetry.bridge import TelemetryBridge
from src.telemetry.exporter import TelemetryExporter
from src.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    cfg = SimConfig(num_agents=30, max_time=200.0, seed=42)
    buf = TelemetryBuffer()
    exporter = TelemetryExporter(output_root="outputs")
    worker = SimulationWorker(
        simulation_config=cfg, telemetry_buffer=buf,
        frame_dt=0.05, exporter=exporter,
    )
    bridge = TelemetryBridge(telemetry_buffer=buf, poll_interval_ms=16)

    window = MainWindow(telemetry_bridge=bridge, worker=worker)
    window.show()

    bridge.start()
    worker.start_simulation()

    exit_code = app.exec()

    worker.stop_simulation()
    bridge.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
