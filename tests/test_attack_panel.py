"""Phase 2K verification: AttackPanel signals and limits."""

import sys
from unittest.mock import MagicMock

from PySide6.QtWidgets import QApplication
from src.gui.attack_panel import AttackPanel


def test_attack_panel():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    panel = AttackPanel()

    # 1. Mock signals
    jam_mock = MagicMock()
    intf_mock = MagicMock()
    drain_mock = MagicMock()
    
    panel.jamming_toggled.connect(jam_mock)
    panel.interference_changed.connect(intf_mock)
    panel.energy_drain_toggled.connect(drain_mock)

    # 2. Test Jamming Toggle (should debounce properly)
    panel.chk_jamming.setChecked(True)
    assert panel.attack_state["jamming"] is True
    jam_mock.assert_called_once_with(True)
    assert panel.sld_interference.isEnabled() is True
    print("[PASS] jamming_toggle (enabled slider and emitted)")

    # 3. Test Interference Clamp (0-100 mapped to 0.0-1.0)
    panel.sld_interference.setValue(50)
    intf_mock.assert_called_once_with(0.5)
    print("[PASS] interference_slider (emits clamped 0.5)")

    import time
    time.sleep(0.15)

    # 4. Energy drain
    panel.chk_energy.setChecked(True)
    drain_mock.assert_called_once_with(True)
    assert panel.sld_drain.isEnabled() is True
    print("[PASS] energy_drain_toggle (enabled slider and emitted)")
    
    # 5. Status text output
    s_text = panel.lbl_status.text()
    assert "Jamming: ON" in s_text
    assert "E-Drain: ON" in s_text
    print(f"[PASS] status_text ({s_text.splitlines()[0]})")

    print("\nAll Phase 2K tests passed.")

if __name__ == "__main__":
    test_attack_panel()
