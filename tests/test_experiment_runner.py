"""Phase 2L verification: ExperimentRunner state guards and signal emission."""

import sys
import time
from unittest.mock import MagicMock

from PySide6.QtWidgets import QApplication

from src.gui.experiment_runner import ExperimentRunner, SCENARIOS, DURATION_MAP


def test_experiment_runner():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    runner = ExperimentRunner()

    # 1. Verify scenarios loaded
    items = [runner.combo.itemText(i) for i in range(runner.combo.count())]
    assert items == SCENARIOS
    print(f"[PASS] scenarios_loaded ({len(items)} scenarios)")

    # 2. Run signal emitted with correct name
    run_mock = MagicMock()
    stop_mock = MagicMock()
    runner.run_experiment.connect(run_mock)
    runner.stop_experiment.connect(stop_mock)

    runner.combo.setCurrentText("jamming_attack")
    runner.btn_run.click()
    run_mock.assert_called_once_with("jamming_attack")
    assert runner.is_running is True
    assert runner.btn_run.isEnabled() is False
    assert runner.btn_stop.isEnabled() is True
    assert runner.combo.isEnabled() is False
    print("[PASS] run_signal (jamming_attack emitted)")

    # 3. Duplicate run blocked
    run_mock.reset_mock()
    runner.btn_run.click()  # btn disabled, but test direct guard
    run_mock.assert_not_called()
    print("[PASS] duplicate_guard (second run blocked)")

    # 4. Stop
    runner.btn_stop.click()
    stop_mock.assert_called_once()
    assert runner.is_running is False
    assert runner.btn_run.isEnabled() is True
    assert runner.btn_stop.isEnabled() is False
    print("[PASS] stop_signal (emitted, state reset)")

    # 5. Debounce
    time.sleep(0.35)
    runner.btn_run.click()
    assert runner.is_running is True
    print("[PASS] debounce (run after cooldown)")

    # 6. mark_finished
    runner.mark_finished()
    assert runner.is_running is False
    assert "FINISHED" in runner.lbl_status.text()
    print("[PASS] mark_finished (status updated)")

    # 7. Duration display
    runner.combo.setCurrentText("energy_cascade")
    assert "15" in runner.lbl_duration.text()
    print(f"[PASS] duration_display ({runner.lbl_duration.text()})")

    print("\nAll Phase 2L tests passed.")


if __name__ == "__main__":
    test_experiment_runner()
