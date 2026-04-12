"""Phase 2J verification: SimulationControls state machine and worker interaction."""

import sys
from unittest.mock import MagicMock

from PySide6.QtWidgets import QApplication

from src.gui.controls import SimulationControls

def test_controls_state_machine():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    controls = SimulationControls()

    # Initial state
    assert controls.state == "STOPPED"
    assert controls.btn_start.isEnabled()
    assert not controls.btn_pause.isEnabled()
    assert not controls.btn_resume.isEnabled()
    assert not controls.btn_reset.isEnabled()
    assert controls.btn_step.isEnabled()

    # Start
    controls.btn_start.click()
    assert controls.state == "RUNNING"
    assert not controls.btn_start.isEnabled()
    assert controls.btn_pause.isEnabled()
    assert controls.btn_reset.isEnabled()
    assert not controls.btn_step.isEnabled()

    # Pause
    controls.btn_pause.click()
    assert controls.state == "PAUSED"
    assert controls.btn_resume.isEnabled()
    assert controls.btn_step.isEnabled()

    # Step (should remain paused)
    controls.btn_step.click()
    assert controls.state == "PAUSED"

    # Resume
    controls.btn_resume.click()
    assert controls.state == "RUNNING"

    # Reset
    controls.btn_reset.click()
    assert controls.state == "STOPPED"
    assert controls.btn_start.isEnabled()

    print("[PASS] state_machine (correct enable/disable transitions)")

    # Simulate Speed check
    mc = MagicMock()
    controls.speed_changed.connect(mc)
    controls.slider.setValue(150)
    mc.assert_called_with(1.50)
    print(f"[PASS] speed_slider (1.5x -> {controls.lbl_speed_val.text()})")

    print("\nAll Phase 2J tests passed.")

if __name__ == "__main__":
    test_controls_state_machine()
