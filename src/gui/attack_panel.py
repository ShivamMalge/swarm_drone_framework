"""
AttackPanel — Observer-layer disturbance injection (Phase 2K).

Emits pure Qt signals for setting attack state. Enforces local state-machine
logic to debounce interactions and safely clamp values before emission.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class AttackPanel(QWidget):
    """
    Simulation disturbance controls.
    State guards prevent redundant emissions and clamp numerical bounds.
    """

    jamming_toggled = Signal(bool)
    interference_changed = Signal(float)
    energy_drain_toggled = Signal(bool)
    energy_drain_rate_changed = Signal(float)
    connectivity_drop_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.attack_state = {
            "jamming": False,
            "interference": 0.0,
            "energy_drain": False,
            "drain_rate": 0.0,
            "connectivity_drop": False,
        }
        self._last_toggle_time = 0.0
        self._debounce_ms = 100  # ms limit for rapid toggles

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header = QLabel("Attack Simulation")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #f85149;")  # Core Red
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(12)

        # ── Toggle Styles ────────────────────────────────────
        chk_style = """
            QCheckBox { color: #8b949e; spacing: 8px; font-weight: bold; }
            QCheckBox::indicator { width: 16px; height: 16px; background-color: #21262d; border-radius: 3px; border: 1px solid #30363d; }
            QCheckBox::indicator:checked { background-color: #f85149; border: 1px solid #da3633; }
        """

        # 1. Jamming
        self.chk_jamming = QCheckBox("Comm Jamming")
        self.chk_jamming.setStyleSheet(chk_style)
        self.chk_jamming.toggled.connect(self._on_jamming_toggled)

        self.lbl_interference = QLabel("Interference:")
        self.lbl_interference.setStyleSheet("color: #8b949e;")
        self.sld_interference = QSlider(Qt.Orientation.Horizontal)
        self.sld_interference.setRange(0, 100)
        self.sld_interference.setEnabled(False)
        self.sld_interference.valueChanged.connect(self._on_interference_changed)

        self.lbl_interference_val = QLabel("0.0%")
        self.lbl_interference_val.setStyleSheet("color: #c9d1d9; font-family: Consolas;")

        grid.addWidget(self.chk_jamming, 0, 0, 1, 3)
        grid.addWidget(self.lbl_interference, 1, 0)
        grid.addWidget(self.sld_interference, 1, 1)
        grid.addWidget(self.lbl_interference_val, 1, 2)

        # 2. Energy Drain
        self.chk_energy = QCheckBox("Energy Drain")
        self.chk_energy.setStyleSheet(chk_style)
        self.chk_energy.toggled.connect(self._on_energy_toggled)

        self.lbl_drain = QLabel("Drain Rate:")
        self.lbl_drain.setStyleSheet("color: #8b949e;")
        self.sld_drain = QSlider(Qt.Orientation.Horizontal)
        self.sld_drain.setRange(0, 100)
        self.sld_drain.setEnabled(False)
        self.sld_drain.valueChanged.connect(self._on_drain_changed)

        self.lbl_drain_val = QLabel("0.0/s")
        self.lbl_drain_val.setStyleSheet("color: #c9d1d9; font-family: Consolas;")

        grid.addWidget(self.chk_energy, 2, 0, 1, 3)
        grid.addWidget(self.lbl_drain, 3, 0)
        grid.addWidget(self.sld_drain, 3, 1)
        grid.addWidget(self.lbl_drain_val, 3, 2)

        # 3. Connectivity Drop
        self.chk_conn = QCheckBox("Connectivity Drop")
        self.chk_conn.setStyleSheet(chk_style)
        self.chk_conn.toggled.connect(self._on_conn_toggled)
        
        grid.addWidget(self.chk_conn, 4, 0, 1, 3)

        layout.addLayout(grid)

        # Status Summary Box
        self.lbl_status = QLabel(self._get_status_text())
        self.lbl_status.setFont(QFont("Consolas", 8))
        self.lbl_status.setStyleSheet("color: #8b949e; background-color: #161b22; padding: 6px; border-radius: 4px; border: 1px solid #30363d;")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

    # ── Handlers ─────────────────────────────────────────────

    def _debounce_check(self) -> bool:
        now = time.perf_counter() * 1000
        if now - self._last_toggle_time < self._debounce_ms:
            return False
        self._last_toggle_time = now
        return True

    def _update_status(self) -> None:
        self.lbl_status.setText(self._get_status_text())

    def _get_status_text(self) -> str:
        s = self.attack_state
        return (
            f"Jamming: {'ON' if s['jamming'] else 'OFF'} | "
            f"Intf: {s['interference']:.2f}\n"
            f"E-Drain: {'ON' if s['energy_drain'] else 'OFF'} | "
            f"Rate: {s['drain_rate']:.2f}/s\n"
            f"Conn_Drop: {'ON' if s['connectivity_drop'] else 'OFF'}"
        )

    def _on_jamming_toggled(self, checked: bool) -> None:
        if not self._debounce_check() or self.attack_state["jamming"] == checked:
            self.chk_jamming.blockSignals(True)
            self.chk_jamming.setChecked(self.attack_state["jamming"])
            self.chk_jamming.blockSignals(False)
            return

        self.attack_state["jamming"] = checked
        self.sld_interference.setEnabled(checked)
        self.jamming_toggled.emit(checked)
        self._update_status()

    def _on_interference_changed(self, val: int) -> None:
        pct = float(max(0, min(100, val))) / 100.0
        if self.attack_state["interference"] == pct:
            return
            
        self.attack_state["interference"] = pct
        self.lbl_interference_val.setText(f"{pct * 100:.1f}%")
        self.interference_changed.emit(pct)
        self._update_status()

    def _on_energy_toggled(self, checked: bool) -> None:
        if not self._debounce_check() or self.attack_state["energy_drain"] == checked:
            self.chk_energy.blockSignals(True)
            self.chk_energy.setChecked(self.attack_state["energy_drain"])
            self.chk_energy.blockSignals(False)
            return

        self.attack_state["energy_drain"] = checked
        self.sld_drain.setEnabled(checked)
        self.energy_drain_toggled.emit(checked)
        self._update_status()

    def _on_drain_changed(self, val: int) -> None:
        rate = float(max(0, min(100, val))) / 10.0  # Max 10.0/s
        if self.attack_state["drain_rate"] == rate:
            return

        self.attack_state["drain_rate"] = rate
        self.lbl_drain_val.setText(f"{rate:.1f}/s")
        self.energy_drain_rate_changed.emit(rate)
        self._update_status()

    def _on_conn_toggled(self, checked: bool) -> None:
        if not self._debounce_check() or self.attack_state["connectivity_drop"] == checked:
            self.chk_conn.blockSignals(True)
            self.chk_conn.setChecked(self.attack_state["connectivity_drop"])
            self.chk_conn.blockSignals(False)
            return

        self.attack_state["connectivity_drop"] = checked
        self.connectivity_drop_toggled.emit(checked)
        self._update_status()
