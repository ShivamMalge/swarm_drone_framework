"""
ReplayPanel — UI controls for mission replay (Phase 2N).

Observer layer. Manages run selection, timeline scrubbing, and playback
speed. Completely independent of SimulationWorker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.replay.replay_loader import ReplayLoader
from src.replay.replay_engine import ReplayEngine, PlaybackState
from src.replay.replay_controller import ReplayController


class ReplayPanel(QWidget):
    """
    Run selector + timeline + playback controls.
    Emits frame_ready via its internal ReplayController.
    """

    replay_frame = Signal(object)  # forwarded from controller

    def __init__(
        self,
        output_root: str | Path = "outputs",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._output_root = Path(output_root)
        self._runs: list[dict] = []
        self._controller: ReplayController | None = None
        self._engine: ReplayEngine | None = None
        self._total_frames: int = 0

        self._init_ui()
        self._scan_runs()

        # Seek debounce
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(80)
        self._seek_timer.timeout.connect(self._do_seek)
        self._pending_seek: int = 0
        self._slider_dragging = False

    # ── UI ───────────────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QLabel("Mission Replay")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #d2a8ff;")
        layout.addWidget(header)

        # Run selector
        row_sel = QHBoxLayout()
        row_sel.addWidget(QLabel("Run:"))
        self.combo_runs = QComboBox()
        self.combo_runs.setMinimumWidth(180)
        self.combo_runs.setStyleSheet(
            "QComboBox { background-color: #21262d; color: #c9d1d9; "
            "border: 1px solid #30363d; border-radius: 4px; padding: 3px 6px; }"
        )
        row_sel.addWidget(self.combo_runs)

        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self._on_load)
        row_sel.addWidget(self.btn_load)
        row_sel.addStretch()
        layout.addLayout(row_sel)

        # Buttons
        btn_style = (
            "QPushButton { background-color: #21262d; border: 1px solid #30363d; "
            "border-radius: 4px; color: #c9d1d9; min-height: 26px; min-width: 60px; } "
            "QPushButton:hover { background-color: #30363d; } "
            "QPushButton:disabled { color: #484f58; background-color: #0d1117; }"
        )

        row_ctrl = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        for b in (self.btn_play, self.btn_pause, self.btn_stop):
            b.setStyleSheet(btn_style)
            b.setEnabled(False)
            row_ctrl.addWidget(b)

        # Speed
        row_ctrl.addSpacing(10)
        row_ctrl.addWidget(QLabel("Speed:"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x", "1x", "2x", "4x"])
        self.combo_speed.setCurrentText("1x")
        self.combo_speed.setStyleSheet(
            "QComboBox { background-color: #21262d; color: #c9d1d9; "
            "border: 1px solid #30363d; border-radius: 4px; padding: 2px 4px; }"
        )
        self.combo_speed.setEnabled(False)
        row_ctrl.addWidget(self.combo_speed)
        row_ctrl.addStretch()
        layout.addLayout(row_ctrl)

        # Timeline slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        layout.addWidget(self.slider)

        # Time label
        self.lbl_time = QLabel("0.00 / 0.00 s  |  Frame: 0 / 0")
        self.lbl_time.setFont(QFont("Consolas", 8))
        self.lbl_time.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.lbl_time)

        layout.addStretch()

        # Connections
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)
        self.combo_speed.currentTextChanged.connect(self._on_speed)
        self.slider.valueChanged.connect(self._on_seek)

    # ── Run scanning ─────────────────────────────────────────

    def _scan_runs(self) -> None:
        self.combo_runs.clear()
        self._runs.clear()

        if not self._output_root.exists():
            return

        # Try runs_index.json first
        idx_path = self._output_root / "runs_index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                self._runs = json.load(f)
        else:
            # Fallback: directory scan
            for d in sorted(self._output_root.iterdir()):
                if d.is_dir() and d.name.startswith("run_"):
                    meta_path = d / "metadata.json"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)
                        self._runs.append({
                            "run_id": meta.get("run_id", d.name),
                            "scenario": meta.get("scenario_name", "?"),
                            "path": str(d),
                        })

        for r in self._runs:
            self.combo_runs.addItem(f"{r.get('scenario', '?')} — {r['run_id']}")

    # ── Handlers ─────────────────────────────────────────────

    def _on_load(self) -> None:
        idx = self.combo_runs.currentIndex()
        if idx < 0 or idx >= len(self._runs):
            return

        run_info = self._runs[idx]
        run_dir = run_info.get("path", str(self._output_root / f"run_{run_info['run_id']}"))

        loader = ReplayLoader(run_dir)
        frames = loader.load()

        self._engine = ReplayEngine(frames)
        self._controller = ReplayController(self._engine, parent=self)
        self._controller.frame_ready.connect(self._on_frame)
        self._controller.playback_finished.connect(self._on_finished)

        self._total_frames = len(frames)
        self.slider.setRange(0, max(0, self._total_frames - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(True)

        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.combo_speed.setEnabled(True)

        dur = self._engine.total_duration
        self.lbl_time.setText(f"0.00 / {dur:.2f} s  |  Frame: 0 / {self._total_frames}")

    def _on_play(self) -> None:
        if self._controller:
            self._controller.start()
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)

    def _on_pause(self) -> None:
        if self._controller:
            self._controller.pause()
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)

    def _on_stop(self) -> None:
        if self._controller:
            self._controller.stop()
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.slider.setValue(0)

    def _on_speed(self, text: str) -> None:
        if self._controller:
            val = float(text.replace("x", ""))
            self._controller.set_speed(val)

    def _on_seek(self, val: int) -> None:
        self._pending_seek = val
        self._slider_dragging = True
        self._seek_timer.start()

    def _do_seek(self) -> None:
        if self._controller and self._engine:
            self._controller.seek(self._pending_seek)
        self._slider_dragging = False

    def _on_frame(self, frame) -> None:
        if self._engine:
            idx = self._engine.index
            dur = self._engine.total_duration
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self.lbl_time.setText(
                f"{frame.time:.2f} / {dur:.2f} s  |  Frame: {idx} / {self._total_frames}"
            )
        self.replay_frame.emit(frame)

    def _on_finished(self) -> None:
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def refresh_runs(self) -> None:
        """Rescan output directory."""
        self._scan_runs()
