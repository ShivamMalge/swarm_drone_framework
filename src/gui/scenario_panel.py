"""
scenario_panel.py — Scenario Configuration Studio (Phase 2P)
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget, QScrollArea, QFormLayout, QLineEdit, QCheckBox,
    QDoubleSpinBox, QSpinBox, QMessageBox, QGroupBox
)

from src.scenario.scenario_model import ScenarioConfig
from src.scenario.scenario_loader import ScenarioLoader
from src.scenario.scenario_validator import ScenarioValidator


class ScenarioPanel(QWidget):
    """
    Visual scenario builder and runner.
    """
    
    # Signal emitted to request starting a custom scenario
    run_custom_scenario = Signal(object)  # emits ScenarioConfig

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.loader = ScenarioLoader()
        self.current_config = ScenarioConfig()
        
        self._init_ui()
        self._refresh_list()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        header = QLabel("Scenario Studio")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #d2a8ff;")
        main_layout.addWidget(header)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        self.combo_scenarios = QComboBox()
        self.combo_scenarios.addItem("-- New Custom --")
        toolbar.addWidget(self.combo_scenarios)
        
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self._on_load)
        toolbar.addWidget(self.btn_load)
        
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._on_save)
        toolbar.addWidget(self.btn_save)
        
        main_layout.addLayout(toolbar)
        
        # Scrollable form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        
        style = "background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d;"
        
        # Core
        self.val_name = QLineEdit(self.current_config.name)
        self.val_name.setStyleSheet(style)
        form_layout.addRow("Name:", self.val_name)
        
        self.val_seed = QSpinBox()
        self.val_seed.setRange(0, 999999)
        self.val_seed.setValue(self.current_config.seed)
        self.val_seed.setStyleSheet(style)
        form_layout.addRow("Seed:", self.val_seed)
        
        self.val_agents = QSpinBox()
        self.val_agents.setRange(1, 500)
        self.val_agents.setValue(self.current_config.num_agents)
        self.val_agents.setStyleSheet(style)
        form_layout.addRow("Num Agents:", self.val_agents)
        
        self.val_radius = QDoubleSpinBox()
        self.val_radius.setRange(1.0, 100.0)
        self.val_radius.setValue(self.current_config.communication_radius)
        self.val_radius.setStyleSheet(style)
        form_layout.addRow("Comm Radius:", self.val_radius)
        
        # Energy
        grp_energy = QGroupBox("Energy")
        lay_energy = QFormLayout(grp_energy)
        self.val_energy_init = QDoubleSpinBox()
        self.val_energy_init.setRange(0.0, 1000.0)
        self.val_energy_init.setValue(self.current_config.energy_params.initial_energy)
        self.val_energy_init.setStyleSheet(style)
        lay_energy.addRow("Initial:", self.val_energy_init)
        
        self.val_energy_drain = QDoubleSpinBox()
        self.val_energy_drain.setDecimals(4)
        self.val_energy_drain.setSingleStep(0.001)
        self.val_energy_drain.setRange(0.0, 1.0)
        self.val_energy_drain.setValue(self.current_config.energy_params.drain_rate)
        self.val_energy_drain.setStyleSheet(style)
        lay_energy.addRow("Drain Rate:", self.val_energy_drain)
        form_layout.addRow(grp_energy)
        
        # Interference
        grp_inf = QGroupBox("Interference")
        lay_inf = QFormLayout(grp_inf)
        self.val_inf_en = QCheckBox()
        self.val_inf_en.setChecked(self.current_config.interference.enabled)
        lay_inf.addRow("Enabled:", self.val_inf_en)
        
        self.val_inf_int = QDoubleSpinBox()
        self.val_inf_int.setRange(0.0, 1.0)
        self.val_inf_int.setValue(self.current_config.interference.intensity)
        self.val_inf_int.setStyleSheet(style)
        lay_inf.addRow("Intensity:", self.val_inf_int)
        form_layout.addRow(grp_inf)
        
        # Sim parameters
        grp_sim = QGroupBox("Simulation")
        lay_sim = QFormLayout(grp_sim)
        self.val_duration = QDoubleSpinBox()
        self.val_duration.setRange(1.0, 10000.0)
        self.val_duration.setValue(self.current_config.simulation.duration)
        self.val_duration.setStyleSheet(style)
        lay_sim.addRow("Duration (s):", self.val_duration)
        form_layout.addRow(grp_sim)
        
        scroll.setWidget(form_widget)
        main_layout.addWidget(scroll)
        
        # Live preview / feedback
        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setStyleSheet("color: #ff7b72;")
        self.lbl_feedback.setWordWrap(True)
        main_layout.addWidget(self.lbl_feedback)
        
        # Run Button
        self.btn_run = QPushButton("Run Custom Scenario")
        self.btn_run.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #238636; color: white; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2ea043; }"
        )
        self.btn_run.clicked.connect(self._on_run)
        main_layout.addWidget(self.btn_run)

    def _refresh_list(self) -> None:
        self.combo_scenarios.clear()
        self.combo_scenarios.addItem("-- New Custom --")
        for f in self.loader.list_scenarios():
            self.combo_scenarios.addItem(f)

    def _sync_to_model(self) -> None:
        self.current_config.name = self.val_name.text()
        self.current_config.seed = self.val_seed.value()
        self.current_config.num_agents = self.val_agents.value()
        self.current_config.communication_radius = self.val_radius.value()
        self.current_config.energy_params.initial_energy = self.val_energy_init.value()
        self.current_config.energy_params.drain_rate = self.val_energy_drain.value()
        self.current_config.interference.enabled = self.val_inf_en.isChecked()
        self.current_config.interference.intensity = self.val_inf_int.value()
        self.current_config.simulation.duration = self.val_duration.value()

    def _sync_from_model(self) -> None:
        self.val_name.setText(self.current_config.name)
        self.val_seed.setValue(self.current_config.seed)
        self.val_agents.setValue(self.current_config.num_agents)
        self.val_radius.setValue(self.current_config.communication_radius)
        self.val_energy_init.setValue(self.current_config.energy_params.initial_energy)
        self.val_energy_drain.setValue(self.current_config.energy_params.drain_rate)
        self.val_inf_en.setChecked(self.current_config.interference.enabled)
        self.val_inf_int.setValue(self.current_config.interference.intensity)
        self.val_duration.setValue(self.current_config.simulation.duration)

    def _on_load(self) -> None:
        idx = self.combo_scenarios.currentIndex()
        if idx <= 0:
            return
        filename = self.combo_scenarios.currentText()
        try:
            self.current_config = self.loader.load_scenario(filename)
            self._sync_from_model()
            self.lbl_feedback.setText("")
        except Exception as e:
            self.lbl_feedback.setText(f"Load failed: {str(e)}")

    def _on_save(self) -> None:
        self._sync_to_model()
        errors = ScenarioValidator.validate(self.current_config)
        if errors:
            self.lbl_feedback.setText("\n".join(errors))
            return
            
        try:
            self.loader.save_scenario(self.current_config)
            self._refresh_list()
            self.lbl_feedback.setText("Saved successfully.", color="#3fb950")
            self.lbl_feedback.setStyleSheet("color: #3fb950;")
        except Exception as e:
            self.lbl_feedback.setStyleSheet("color: #ff7b72;")
            self.lbl_feedback.setText(f"Save failed: {str(e)}")

    def _on_run(self) -> None:
        self._sync_to_model()
        errors = ScenarioValidator.validate(self.current_config)
        if errors:
            self.lbl_feedback.setStyleSheet("color: #ff7b72;")
            self.lbl_feedback.setText("\n".join(errors))
            return
            
        warns = ScenarioValidator.check_performance_safety(self.current_config)
        if warns:
            msg = "\n".join(warns) + "\n\nProceed anyway?"
            reply = QMessageBox.question(self, 'Performance Warning', msg, 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.lbl_feedback.setText("")
        self.run_custom_scenario.emit(self.current_config)
