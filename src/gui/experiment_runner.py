"""
ExperimentRunner — Automated scenario execution interface (Phase 2L).

Observer layer. Emits signals to trigger predefined experiment configurations
on the SimulationWorker. No direct kernel or simulation access.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


SCENARIOS = [
    "baseline",
    "percolation_collapse",
    "energy_cascade",
    "jamming_attack",
]

DURATION_MAP = {
    "baseline": 10.0,
    "percolation_collapse": 15.0,
    "energy_cascade": 15.0,
    "jamming_attack": 12.0,
}


class ExperimentRunner(QWidget):
    """
    Scenario selector and run/stop controls.
    Maintains local is_running guard to prevent overlapping experiments.
    """

    run_experiment = Signal(str)
    stop_experiment = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.is_running = False
        self._last_run_time = 0.0
        self._debounce_ms = 300

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header = QLabel("Experiment Runner")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #58a6ff;")
        layout.addWidget(header)

        # Scenario selector
        row1 = QHBoxLayout()
        lbl_scenario = QLabel("Scenario:")
        lbl_scenario.setStyleSheet("color: #8b949e;")
        row1.addWidget(lbl_scenario)

        self.combo = QComboBox()
        self.combo.addItems(SCENARIOS)
        self.combo.setStyleSheet(
            """
            QComboBox {
                background-color: #21262d; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 4px;
                padding: 4px 8px; min-width: 160px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #161b22; color: #c9d1d9;
                selection-background-color: #30363d;
            }
            """
        )
        row1.addWidget(self.combo)
        row1.addStretch()
        layout.addLayout(row1)

        # Buttons
        row2 = QHBoxLayout()

        btn_style = """
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d; border-radius: 4px;
                color: #c9d1d9; min-height: 28px; min-width: 90px;
            }
            QPushButton:hover { background-color: #30363d; }
            QPushButton:disabled { color: #484f58; background-color: #0d1117; border-color: #21262d; }
        """

        self.btn_run = QPushButton("Run")
        self.btn_run.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.btn_run.setStyleSheet(btn_style)
        self.btn_run.clicked.connect(self._on_run)
        row2.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.btn_stop.setStyleSheet(btn_style)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        row2.addWidget(self.btn_stop)

        row2.addStretch()
        layout.addLayout(row2)

        # Status
        self.lbl_status = QLabel("Scenario: — | Status: IDLE")
        self.lbl_status.setFont(QFont("Consolas", 9))
        self.lbl_status.setStyleSheet(
            "color: #8b949e; background-color: #161b22; padding: 6px;"
            "border-radius: 4px; border: 1px solid #30363d;"
        )
        layout.addWidget(self.lbl_status)

        # Duration info
        self.lbl_duration = QLabel("")
        self.lbl_duration.setFont(QFont("Consolas", 8))
        self.lbl_duration.setStyleSheet("color: #484f58;")
        layout.addWidget(self.lbl_duration)

        layout.addStretch()

        # Update duration on selection change
        self.combo.currentTextChanged.connect(self._on_scenario_changed)
        self._on_scenario_changed(self.combo.currentText())

    # ── Handlers ─────────────────────────────────────────────

    def _on_run(self) -> None:
        now = time.perf_counter() * 1000
        if now - self._last_run_time < self._debounce_ms:
            return
        if self.is_running:
            return
        self._last_run_time = now

        scenario = self.combo.currentText()
        self.is_running = True
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo.setEnabled(False)
        self.lbl_status.setText(f"Scenario: {scenario} | Status: RUNNING")
        self.lbl_status.setStyleSheet(
            "color: #3fb950; background-color: #161b22; padding: 6px;"
            "border-radius: 4px; border: 1px solid #30363d;"
        )
        self.run_experiment.emit(scenario)

    def _on_stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo.setEnabled(True)
        self.lbl_status.setText(f"Scenario: {self.combo.currentText()} | Status: STOPPED")
        self.lbl_status.setStyleSheet(
            "color: #8b949e; background-color: #161b22; padding: 6px;"
            "border-radius: 4px; border: 1px solid #30363d;"
        )
        self.stop_experiment.emit()

    def _on_scenario_changed(self, name: str) -> None:
        dur = DURATION_MAP.get(name, 0)
        self.lbl_duration.setText(f"Duration: {dur:.0f}s" if dur else "")

    def mark_finished(self) -> None:
        """Called externally when the worker reports experiment completion."""
        self.is_running = False
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo.setEnabled(True)
        self.lbl_status.setText(f"Scenario: {self.combo.currentText()} | Status: FINISHED")
        self.lbl_status.setStyleSheet(
            "color: #d2a8ff; background-color: #161b22; padding: 6px;"
            "border-radius: 4px; border: 1px solid #30363d;"
        )
