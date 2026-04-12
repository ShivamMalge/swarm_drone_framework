"""
MainWindow — Root GUI container for the Forensic Telemetry Dashboard.

Observer layer only. Receives data exclusively via TelemetryBridge signals.
No simulation, kernel, or agent access.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QPushButton,
    QGroupBox
)

import collections
import time

from src.telemetry.bridge import TelemetryBridge
from src.telemetry.telemetry_frame import TelemetryFrame
from src.gui.graphs import TelemetryGraphsWidget
from src.gui.regime_panel import RegimePanel
from src.gui.adaptive_panel import AdaptivePanel
from src.gui.network_viewer import NetworkViewer
from src.gui.event_timeline import EventTimeline
from src.gui.health_panel import HealthPanel
from src.gui.anomaly_panel import AnomalyPanel
from src.gui.research_panel import ResearchPanel
from src.gui.event_timeline_panel import EventTimelinePanel
from src.gui.event_log_panel import EventLogPanel
from src.gui.controls import SimulationControls
from src.gui.attack_panel import AttackPanel
from src.gui.experiment_runner import ExperimentRunner
from src.gui.mission_playback import MissionPlaybackPanel
from src.gui.scenario_panel import ScenarioPanel
from src.gui.percolation_panel import PercolationPanel
from src.gui.spectral_panel import SpectralPanel
from src.gui.spectral_alarm_panel import SpectralAlarmPanel
from src.gui.energy_panel import EnergyPanel
from src.gui.consensus_panel import ConsensusPanel
from src.analytics.percolation_analyzer import PercolationAnalyzer
from src.analytics.spectral_analyzer import SpectralAnalyzer
from src.analytics.energy_cascade_analyzer import EnergyCascadeAnalyzer
from src.analytics.consensus_analyzer import ConsensusAnalyzer
from src.analytics.event_logger import EventLogger
from src.analytics.swarm_health import SwarmHealthAnalyzer
from src.analytics.anomaly_detector import AnomalyDetector
from src.analytics.research_metrics import ResearchMetricsEngine
from src.analytics.event_fusion_engine import EventFusionEngine
from src.analytics.percolation_shock_analyzer import PercolationShockAnalyzer
from src.analytics.spectral_alarm import SpectralAlarm
from src.analytics.energy_heatmap_mapper import EnergyHeatmapMapper
from src.telemetry.worker import SimulationWorker


class MainWindow(QMainWindow):
    """
    Root dashboard window. Layout:

    ┌──────────────────┬────────────┐
    │  Swarm Map       │ Side Panel │
    │  (Phase 2F)      │            │
    │                  │  Regime    │
    │                  │  Adaptive  │
    ├──────────────────┴────────────┤
    │  Bottom Graphs                │
    │  Spectral | Energy | Consensus│
    └───────────────────────────────┘
    """

    def __init__(
        self,
        telemetry_bridge: TelemetryBridge,
        worker: SimulationWorker,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._bridge = telemetry_bridge
        self._worker = worker
        self._frame_count = 0
        self._last_frame_time = time.perf_counter()
        self._fps_buffer = collections.deque(maxlen=60)
        self._mode = "LIVE"
        self._render_time_ms = 0.0
        self._edge_downscaled = False

        self.percolation_analyzer = PercolationAnalyzer()
        self.spectral_analyzer = SpectralAnalyzer()
        self.energy_analyzer = EnergyCascadeAnalyzer()
        self.consensus_analyzer = ConsensusAnalyzer()
        self.health_analyzer = SwarmHealthAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.research_engine = ResearchMetricsEngine()
        self.event_fusion = EventFusionEngine()
        self.shock_analyzer = PercolationShockAnalyzer()
        self.spectral_alarm = SpectralAlarm()
        self.energy_heatmap_mapper = EnergyHeatmapMapper()
        self.event_logger = EventLogger()

        self._warmup_count = 0 
        self._warmup_required = 3

        self._init_window()
        self._init_layout()
        self._connect_signals()

    # ── Window setup ─────────────────────────────────────────

    def _init_window(self) -> None:
        self.setWindowTitle("Swarm Autonomy — Forensic Telemetry Dashboard")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 950)

        # Dark palette + Global font density
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0d1117; }
            QWidget     { background-color: #0d1117; color: #c9d1d9; font-size: 11px; }
            QLabel      { color: #c9d1d9; }
            QFrame      { border: 1px solid #21262d; border-radius: 6px; background-color: #161b22; }
            """
        )

    # ── Layout construction ──────────────────────────────────

    # ── Layout construction ──────────────────────────────────

    def _init_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(10)

        # 1. Navbar
        navbar_layout = QHBoxLayout()
        navbar_layout.setContentsMargins(0, 0, 0, 10)
        self.btn_mode_scenario = QPushButton("Scenario")
        self.btn_mode_monitor = QPushButton("System Monitor")
        self.btn_mode_visuals = QPushButton("Visualizations")
        
        for btn in (self.btn_mode_scenario, self.btn_mode_monitor, self.btn_mode_visuals):
            btn.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
            btn.setMinimumHeight(40)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                """
                QPushButton { background-color: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 0 20px;}
                QPushButton:hover { background-color: #30363d; }
                QPushButton:checked { background-color: #1f6feb; color: white; border-color: #388bfd; }
                """
            )
            btn.setCheckable(True)
            navbar_layout.addWidget(btn)
            
        navbar_layout.addStretch()
        root_layout.addLayout(navbar_layout)

        # 2. Main content split
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # Left side: Main Visualization
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.controls = SimulationControls()
        left_layout.addWidget(self.controls)

        self.swarm_map = NetworkViewer()
        self.swarm_map.setMinimumSize(800, 600)
        left_layout.addWidget(self.swarm_map, stretch=1)

        content_layout.addWidget(left_panel, stretch=7)

        # Right side: Dynamic Stacked Panel
        self.right_stack = QStackedWidget()
        self.right_stack.setStyleSheet("QStackedWidget { background-color: #0d1117; border: none; }")
        
        self._build_panels()
        
        self.right_stack.addWidget(self._make_scenario_mode())
        self.right_stack.addWidget(self._make_monitor_mode())
        self.right_stack.addWidget(self._make_visuals_mode())

        content_layout.addWidget(self.right_stack, stretch=3)
        root_layout.addLayout(content_layout)

        # Wiring
        self.btn_mode_scenario.clicked.connect(lambda: self.switch_mode("scenario"))
        self.btn_mode_monitor.clicked.connect(lambda: self.switch_mode("monitor"))
        self.btn_mode_visuals.clicked.connect(lambda: self.switch_mode("visuals"))
        
        self.switch_mode("scenario")

        # Status bar
        self._status_label = QLabel("Initializing simulation...")
        self._status_label.setStyleSheet("color: #8b949e; padding: 2px 6px;")
        self.statusBar().addPermanentWidget(self._status_label)
        self.statusBar().setStyleSheet("background-color: #161b22; border-top: 1px solid rgba(255,255,255,0.08);")

    def switch_mode(self, mode: str) -> None:
        self.btn_mode_scenario.setChecked(mode == "scenario")
        self.btn_mode_monitor.setChecked(mode == "monitor")
        self.btn_mode_visuals.setChecked(mode == "visuals")
        
        if mode == "scenario":
            self.right_stack.setCurrentIndex(0)
        elif mode == "monitor":
            self.right_stack.setCurrentIndex(1)
        elif mode == "visuals":
            self.right_stack.setCurrentIndex(2)

    def _build_panels(self) -> None:
        self.regime_panel = RegimePanel()
        self.adaptive_panel = AdaptivePanel()
        self.attack_panel = AttackPanel()
        self.scenario_panel = ScenarioPanel()
        self.percolation_panel = PercolationPanel()
        self.spectral_panel = SpectralPanel()
        self.spectral_alarm_panel = SpectralAlarmPanel()
        self.energy_panel = EnergyPanel()
        self.consensus_panel = ConsensusPanel()
        self.health_panel = HealthPanel()
        self.anomaly_panel = AnomalyPanel()
        self.research_panel = ResearchPanel()
        self.fused_timeline = EventTimelinePanel()
        self.fused_log = EventLogPanel()
        self.event_timeline = EventTimeline()
        self.experiment_runner = ExperimentRunner()
        self.replay_panel = MissionPlaybackPanel()
        self.graphs_panel = TelemetryGraphsWidget()

        self.event_timeline.setMinimumHeight(120)
        self.event_timeline.setMaximumHeight(180)
        self.fused_timeline.setMinimumHeight(120)
        self.fused_timeline.setMaximumHeight(180)
        self.fused_log.setMinimumHeight(120)
        self.fused_log.setMaximumHeight(200)
        
        self.spectral_panel.setMaximumHeight(120)
        self.energy_panel.setMaximumHeight(120)
        self.spectral_alarm_panel.setMaximumHeight(120)

    def _wrap_in_group(self, title: str, widgets: list[QWidget]) -> QGroupBox:
        box = QGroupBox(title)
        box.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        box.setStyleSheet(
            "QGroupBox { border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; margin-top: 1ex; padding: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; color: #58a6ff; }"
        )
        lay = QVBoxLayout(box)
        lay.setSpacing(8)
        for w in widgets:
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            lay.addWidget(w)
        return box

    def _make_scenario_mode(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #0d1117; }")
        
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        
        lay.addWidget(self._wrap_in_group("Scenario Config", [self.scenario_panel, self.experiment_runner]))
        lay.addWidget(self._wrap_in_group("Execution & Interference", [self.attack_panel]))
        lay.addWidget(self._wrap_in_group("Replay Session", [self.replay_panel]))
        lay.addStretch()
        
        scroll.setWidget(w)
        return scroll

    def _make_monitor_mode(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #0d1117; }")
        
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        
        core_sys = self._wrap_in_group("Core Systems", [
            self.percolation_panel,
            self.spectral_panel,
            self.spectral_alarm_panel,
            self.energy_panel,
            self.consensus_panel
        ])
        intel_layer = self._wrap_in_group("Intelligence Layer", [
            self.anomaly_panel,
            self.health_panel,
            self.research_panel,
            self.regime_panel,
            self.adaptive_panel
        ])
        event_layer = self._wrap_in_group("Event Layer", [
            self.fused_timeline,
            self.fused_log,
            self.event_timeline
        ])
        
        lay.addWidget(core_sys)
        lay.addWidget(intel_layer)
        lay.addWidget(event_layer)
        lay.addStretch()
        
        scroll.setWidget(w)
        return scroll

    def _make_visuals_mode(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.graphs_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.graphs_panel)
        return w
    # ── Signal wiring ────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._bridge.frame_ready.connect(self.update_frame)

        # Worker controls
        self.controls.start_clicked.connect(self._worker.start_simulation)
        self.controls.pause_clicked.connect(self._worker.pause_simulation)
        self.controls.resume_clicked.connect(self._worker.resume_simulation)
        self.controls.reset_clicked.connect(self._worker.reset_simulation)
        self.controls.step_clicked.connect(self._worker.step_simulation)
        self.controls.speed_changed.connect(self._worker.set_speed)

        # Attack Controls
        self.attack_panel.jamming_toggled.connect(self._worker.set_jamming)
        self.attack_panel.interference_changed.connect(self._worker.set_interference)
        self.attack_panel.energy_drain_toggled.connect(self._worker.set_energy_drain)
        self.attack_panel.energy_drain_rate_changed.connect(self._worker.set_drain_rate)
        self.attack_panel.connectivity_drop_toggled.connect(self._worker.set_connectivity_drop)

        # Experiment & Scenario Runner
        self.experiment_runner.run_experiment.connect(self._worker.run_experiment)
        self.experiment_runner.stop_experiment.connect(self._worker.stop_experiment)
        self._worker.experiment_finished.connect(self.experiment_runner.mark_finished)
        self.scenario_panel.run_custom_scenario.connect(self._worker.run_custom_scenario)

        # Replay
        self.replay_panel.replay_frame.connect(self.update_frame)
        self.replay_panel.reset_state.connect(self.reset_analyzers)
        
        self.research_panel.btn_export.clicked.connect(lambda: self.research_engine.export_json("run_metrics_export.json"))
        
        # Connect Analyzers to Event Logger
        self.event_logger.event_logged.connect(self.event_timeline.add_event)
        
        self.percolation_analyzer.percolation_collapse_detected.connect(lambda t: self.event_logger.log_event("percolation_collapse_detected", "percolation", t))
        self.percolation_analyzer.percolation_recovered.connect(lambda t: self.event_logger.log_event("percolation_recovered", "percolation", t))
        
        self.spectral_analyzer.spectral_instability_detected.connect(lambda t: self.event_logger.log_event("spectral_instability_detected", "spectral", t))
        self.spectral_analyzer.spectral_recovered.connect(lambda t: self.event_logger.log_event("spectral_recovered", "spectral", t))
        
        self.energy_analyzer.energy_cascade_detected.connect(lambda t: self.event_logger.log_event("energy_cascade_detected", "energy", t))
        self.energy_analyzer.energy_cascade_recovered.connect(lambda t: self.event_logger.log_event("energy_cascade_recovered", "energy", t))
        self.energy_analyzer.energy_stress_warning.connect(lambda t: self.event_logger.log_event("energy_stress_warning", "energy", t))
        
        self.consensus_analyzer.consensus_diverging.connect(lambda t: self.event_logger.log_event("consensus_diverging", "consensus", t))
        self.consensus_analyzer.consensus_converged.connect(lambda t: self.event_logger.log_event("consensus_converged", "consensus", t))
        self.consensus_analyzer.consensus_oscillating.connect(lambda t: self.event_logger.log_event("consensus_oscillating", "consensus", t))
        self.consensus_analyzer.consensus_stalled.connect(lambda t: self.event_logger.log_event("consensus_stalled", "consensus", t))
        
        # Event bindings for Network Viewer
        self.percolation_analyzer.percolation_collapse_detected.connect(lambda t: None) # It gets data from perc_metrics
        self.spectral_analyzer.spectral_instability_detected.connect(lambda t: self.swarm_map.trigger_spectral_instability())
        self.spectral_analyzer.spectral_recovered.connect(lambda t: self.swarm_map.clear_spectral_instability())
        self.energy_analyzer.energy_cascade_detected.connect(lambda t: self.swarm_map.trigger_energy_cascade())
        self.energy_analyzer.energy_cascade_recovered.connect(lambda t: self.swarm_map.clear_energy_cascade())
        

    # ── Frame handler (stub for Phase 2F+) ───────────────────

    def reset_analyzers(self) -> None:
        self.percolation_analyzer = PercolationAnalyzer()
        self.spectral_analyzer = SpectralAnalyzer()
        self.energy_analyzer = EnergyCascadeAnalyzer()
        self.consensus_analyzer = ConsensusAnalyzer()
        self.health_analyzer = SwarmHealthAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        # We don't reset research_engine fully because it aggregates run metrics,
        # but to trace identically during a seek, we must clear it.
        self.research_engine.reset()
        self.event_fusion.reset()
        self.shock_analyzer.reset()
        self.spectral_alarm.reset()
        self.energy_heatmap_mapper._last_hash = -1
        
        self.event_logger.events.clear()
        self.event_timeline._event_list.clear()
        self.fused_timeline.clear()
        self.fused_log.clear()

        # Re-trigger warmup
        self._warmup_count = 0
        self._status_label.setText("Waiting for telemetry... (Warming up)")
        
    def update_frame(self, frame: TelemetryFrame) -> None:
        """Receive telemetry frame and propagate to child panels."""
        t0 = time.perf_counter()
        self._frame_count += 1
        
        # We must call analyzers first so network viewer can use perc_metrics
        perc_metrics = self.percolation_analyzer.analyze_frame(frame)
        self.percolation_panel.update_metrics(perc_metrics)
        
        spec_metrics = self.spectral_analyzer.analyze_frame(frame)
        self.spectral_panel.update_metrics(spec_metrics)
        
        alarm_result = self.spectral_alarm.analyze(
            frame.adjacency, ~frame.drone_failure_flags,
            timestamp=frame.time, frame_index=self._frame_count,
        )
        self.spectral_alarm_panel.update_alarm(alarm_result)
        if alarm_result.events:
            self.fused_timeline.add_events(alarm_result.events)
            self.fused_log.add_events(alarm_result.events)
        
        eng_metrics = self.energy_analyzer.analyze_frame(frame)
        self.energy_panel.update_metrics(eng_metrics)
        
        cns_metrics = self.consensus_analyzer.analyze_frame(frame)
        self.consensus_panel.update_metrics(cns_metrics)
        
        health_metrics = self.health_analyzer.analyze(perc_metrics, spec_metrics, eng_metrics, cns_metrics, frame.time)
        self.health_panel.update_metrics(health_metrics)
        
        anom_metrics = self.anomaly_detector.analyze_frame(frame)
        self.anomaly_panel.update_metrics(anom_metrics)
        
        run_metrics = self.research_engine.add_frame_metrics(frame, perc_metrics, spec_metrics, eng_metrics, cns_metrics, anom_metrics, health_metrics)
        self.research_panel.update_metrics(run_metrics)

        # Event fusion
        fused = self.event_fusion.analyze_frame(
            frame, perc_metrics, spec_metrics, eng_metrics,
            cns_metrics, anom_metrics, health_metrics, self._frame_count
        )
        if fused:
            self.fused_timeline.add_events(fused)
            self.fused_log.add_events(fused)
            self.swarm_map.set_fused_events(fused)

        # Shock analysis
        shock = self.shock_analyzer.analyze(frame, perc_metrics)
        self.swarm_map.update_shock(shock)

        # Energy Heatmap
        heatmap_res = self.energy_heatmap_mapper.map_energies(frame)
        self.swarm_map.update_energy_heatmap(heatmap_res)

        # -- UI updates below --
        # Check Warmup mode
        if self._warmup_count < self._warmup_required:
            self.swarm_map.update_frame(frame, perc_metrics, anom_metrics)
            self._warmup_count += 1
            if self._warmup_count == self._warmup_required:
                self._status_label.setText("Telemetry Active")
            return
        
        self.swarm_map.update_frame(frame, perc_metrics, anom_metrics)
        self.graphs_panel.update_frame(frame)
        self.regime_panel.update_frame(frame)
        self.adaptive_panel.update_frame(frame)



        self._render_time_ms = (time.perf_counter() - t0) * 1000.0

        now = time.perf_counter()
        dt = max(now - self._last_frame_time, 0.001)
        self._last_frame_time = now
        self._fps_buffer.append(1.0 / dt)
        fps = sum(self._fps_buffer) / len(self._fps_buffer)

        # Adaptive edge downscaling (handled internally or disabled conditionally)
        if fps < 30 and not self._edge_downscaled:
            self._edge_downscaled = True
        elif fps > 50 and self._edge_downscaled:
            self._edge_downscaled = False

        regime = "UNKNOWN"
        if frame.regime_state:
            counter = collections.Counter(frame.regime_state.values())
            counter.pop("DEAD", None)
            if counter:
                regime = counter.most_common(1)[0][0]

        # Detect mode
        mode = self._mode

        # Status mapping for events
        last_event = "None"
        if len(self.event_logger.events) > 0:
            last_event = self.event_logger.events[-1].event_type
            
        self._status_label.setText(
            f"State: {self.controls.state} | "
            f"Agents: {int((~frame.drone_failure_flags).sum())} | "
            f"FPS: {int(fps)} | "
            f"Regime: {regime} | "
            f"Conn: {perc_metrics.connectivity_ratio:.2f} | "
            f"LCC: {perc_metrics.lcc_size}/{perc_metrics.total_agents} | "
            f"λ₂: {spec_metrics.lambda2:.2f} | Alarm: {alarm_result.state} | "
            f"E: {eng_metrics.mean_energy:.1f} | Alive: {eng_metrics.alive_count} | Fail: {eng_metrics.failure_count} | "
            f"CVar: {cns_metrics.consensus_variance:.3f} | Rate: {cns_metrics.global_convergence_rate:+.3f} | "
            f"Components: {perc_metrics.num_components} | LCC: {perc_metrics.connectivity_ratio * 100:.0f}% | "
            f"Events: {len(self.event_logger.events)} | Last: {last_event} | "
            f"Anomalies: {anom_metrics.anomaly_count} | Suspicious: {int((anom_metrics.class_labels == 1).sum())} | "
            f"Health: {health_metrics.health_score:.2f} ({health_metrics.state}) | "
            f"Mode: {mode}"
        )

    def set_mode(self, mode: str) -> None:
        self._mode = mode
