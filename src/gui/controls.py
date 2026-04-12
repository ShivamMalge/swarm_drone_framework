"""
SimulationControls — Observer-layer command interface (Phase 2J).

Emits pure Qt signals for worker control. Enforces local state-machine
logic to safely enable/disable allowed button actions (preventing illegal
transitions like pause while stopped).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QWidget,
)


class SimulationControls(QWidget):
    """
    Simulation control buttons and speed slider. State machine:
    STOPPED, RUNNING, PAUSED.
    """

    start_clicked = Signal()
    pause_clicked = Signal()
    resume_clicked = Signal()
    reset_clicked = Signal()
    step_clicked = Signal()
    speed_changed = Signal(float)  # multiplier (0.25 -> 2.0)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = "STOPPED"
        self._init_ui()
        self._apply_state_rules()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ── Buttons ──────────────────────────────────────────
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_resume = QPushButton("Resume")
        self.btn_step = QPushButton("Step")
        self.btn_reset = QPushButton("Reset")

        for btn in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_step, self.btn_reset):
            btn.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
            btn.setMinimumHeight(30)
            btn.setMinimumWidth(80)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #21262d; 
                    border: 1px solid #30363d; 
                    border-radius: 4px;
                    color: #c9d1d9;
                }
                QPushButton:hover { background-color: #30363d; }
                QPushButton:disabled { color: #484f58; background-color: #0d1117; border-color: #21262d; }
                """
            )
            layout.addWidget(btn)

        # ── Speed Slider ──────────────────────────────────────
        layout.addSpacing(20)
        
        lbl_speed = QLabel("Speed:")
        lbl_speed.setStyleSheet("color: #8b949e;")
        layout.addWidget(lbl_speed)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(25)   # 0.25x
        self.slider.setMaximum(200)  # 2.0x
        self.slider.setValue(100)    # 1.0x
        self.slider.setFixedWidth(150)
        layout.addWidget(self.slider)

        self.lbl_speed_val = QLabel("1.00x")
        self.lbl_speed_val.setStyleSheet("color: #c9d1d9; min-width: 40px;")
        layout.addWidget(self.lbl_speed_val)

        layout.addStretch()

        # ── Connections ──────────────────────────────────────
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_resume.clicked.connect(self._on_resume)
        self.btn_step.clicked.connect(self._on_step)
        self.btn_reset.clicked.connect(self._on_reset)
        self.slider.valueChanged.connect(self._on_slider)

    # ── Handlers ──────────────────────────────────────────────

    def _on_start(self) -> None:
        self._set_state("RUNNING")
        self.start_clicked.emit()

    def _on_pause(self) -> None:
        self._set_state("PAUSED")
        self.pause_clicked.emit()

    def _on_resume(self) -> None:
        self._set_state("RUNNING")
        self.resume_clicked.emit()

    def _on_step(self) -> None:
        self._set_state("PAUSED")  # Step leaves it paused
        self.step_clicked.emit()

    def _on_reset(self) -> None:
        self._set_state("STOPPED")
        self.reset_clicked.emit()

    def _on_slider(self, val: int) -> None:
        speed = val / 100.0
        self.lbl_speed_val.setText(f"{speed:.2f}x")
        self.speed_changed.emit(speed)

    # ── Rule Enforcement ──────────────────────────────────────

    def _set_state(self, state: str) -> None:
        self.state = state
        self._apply_state_rules()

    def _apply_state_rules(self) -> None:
        s = self.state
        self.btn_start.setEnabled(s == "STOPPED")
        self.btn_pause.setEnabled(s == "RUNNING")
        self.btn_resume.setEnabled(s == "PAUSED")
        self.btn_reset.setEnabled(s in ("RUNNING", "PAUSED"))
        self.btn_step.setEnabled(s in ("STOPPED", "PAUSED"))
