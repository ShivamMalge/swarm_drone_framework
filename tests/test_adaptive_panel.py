"""Phase 2I verification: AdaptivePanel scalar extraction and flash timing."""

import sys
import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from src.telemetry.telemetry_frame import TelemetryFrame
from src.gui.adaptive_panel import AdaptivePanel


def test_adaptive_panel():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    panel = AdaptivePanel()
    panel.show()

    # Create dummy frame with array data
    n_agents = 10
    frame1 = TelemetryFrame.empty(n_agents)
    frame1.adaptive_parameters = {
        "coverage_gains": np.full(n_agents, 0.45),
        "gossip_epsilons": np.full(n_agents, 0.12),
        "velocity_scales": np.full(n_agents, 1.8),
        "broadcast_rates": np.full(n_agents, 0.6),
        "auction_participations": np.full(n_agents, 1.0),
        "projection_events_total": 5,
    }

    # Step 1: Initial flush
    panel.update_frame(frame1)
    
    assert panel._labels["coverage_gains"].text() == "0.450", "Incorrect format/value for float mean"
    assert panel._labels["projection_events_total"].text() == "5", "Incorrect format for int total"
    print("[PASS] parameter_extraction (means calculated correctly)")

    # Step 2: Ensure color is default white/gray initially
    assert "color: #c9d1d9;" in panel._labels["coverage_gains"].styleSheet()

    # Create frame 2 with change to trigger flash
    frame2 = TelemetryFrame.empty(n_agents)
    frame2.adaptive_parameters = frame1.adaptive_parameters.copy()
    frame2.adaptive_parameters["coverage_gains"] = np.full(n_agents, 0.99)
    
    panel.update_frame(frame2)
    
    assert panel._labels["coverage_gains"].text() == "0.990"
    assert "color: #d29922;" in panel._labels["coverage_gains"].styleSheet(), "Label did not flash yellow"
    assert panel._reset_timer.isActive(), "Timer did not start on change"
    print("[PASS] visual_flash (color changed, timer started)")

    # Close quickly
    QTimer.singleShot(100, app.quit)
    app.exec()

    print("\nAll Phase 2I tests passed.")


if __name__ == "__main__":
    test_adaptive_panel()
