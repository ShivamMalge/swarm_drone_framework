"""
mission_playback.py — GUI for precise mission review integrating analytics timeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPainter, QColor
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
    QStyleOptionSlider, QStyle
)

from src.replay.replay_loader import ReplayLoader
from src.replay.replay_engine import ReplayEngine, PlaybackState
from src.replay.mission_controller import MissionController


class TimelineSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.events: dict[int, str] = {} # frame_idx -> event_type

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if not self.events:
            return
            
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        
        # Calculate available pixel range for ticks
        rect = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self)
        
        painter = QPainter(self)
        min_v = self.minimum()
        max_v = self.maximum()
        if max_v <= min_v:
            return
            
        range_v = max_v - min_v
        
        for frame, etype in self.events.items():
            if etype == "collapse":
                color = QColor(248, 81, 73, 200) # Red
            elif etype == "cascade":
                color = QColor(210, 153, 34, 200) # Yellow
            else:
                color = QColor(88, 166, 255, 200) # Blue
                
            ratio = (frame - min_v) / range_v
            x = rect.x() + int(ratio * rect.width())
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRect(x - 1, rect.y() + 2, 3, rect.height() - 4)


class MissionPlaybackPanel(QWidget):
    """
    Advanced replay suite linking 2Q-2Y analytics sequences deterministically.
    """
    
    replay_frame = Signal(object)
    reset_state = Signal()

    def __init__(self, output_root: str | Path = "outputs", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_root = Path(output_root)
        self._runs: list[dict] = []
        self._controller: MissionController | None = None
        self._engine: ReplayEngine | None = None

        self._init_ui()
        self._scan_runs()

        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(80)
        self._seek_timer.timeout.connect(self._do_seek)
        self._pending_seek: int = 0
        self._slider_dragging = False

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Top row: Run selector
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Mission:"))
        self.cb_runs = QComboBox()
        self.cb_runs.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cb_runs.currentIndexChanged.connect(self._on_run_selected)
        top_layout.addWidget(self.cb_runs, stretch=1)
        
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self._load_selected)
        top_layout.addWidget(self.btn_load)
        layout.addLayout(top_layout)

        # Timeline Slider (with overrides for event marking)
        self.slider = TimelineSlider()
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        # Controls row
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)

        self.cb_speed = QComboBox()
        self.cb_speed.addItems(["0.5x", "1.0x", "2.0x", "4.0x"])
        self.cb_speed.setCurrentIndex(1)
        self.cb_speed.currentIndexChanged.connect(self._on_speed_changed)

        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.lbl_frame.setFont(QFont("Consolas", 10))

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(QLabel("Speed:"))
        ctrl_layout.addWidget(self.cb_speed)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.lbl_frame)

        layout.addLayout(ctrl_layout)
        self._set_controls_enabled(False)

    def _scan_runs(self) -> None:
        self.cb_runs.blockSignals(True)
        self.cb_runs.clear()
        self._runs.clear()

        if self._output_root.exists():
            for d in sorted(self._output_root.iterdir(), reverse=True):
                if d.is_dir() and (d / "telemetry.parquet").exists():
                    cfg_path = d / "metadata.json"
                    if cfg_path.exists():
                        try:
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                        except Exception:
                            meta = {"name": d.name}
                    else:
                        meta = {"name": d.name}

                    meta["_dir"] = d
                    self._runs.append(meta)
                    display = f"{meta.get('name', d.name)} ({d.name})"
                    self.cb_runs.addItem(display)

        if self.cb_runs.count() > 0:
            self.cb_runs.setCurrentIndex(0)
            
        self.cb_runs.blockSignals(False)

    def _on_run_selected(self, index: int) -> None:
        pass

    def _load_selected(self) -> None:
        idx = self.cb_runs.currentIndex()
        if idx < 0:
            return

        run_info = self._runs[idx]
        run_dir = run_info["_dir"]

        parquet_path = run_dir / "telemetry.parquet"
        if not parquet_path.exists():
            return

        loader = ReplayLoader(str(parquet_path))
        self._engine = ReplayEngine(loader)
        
        # We replace the controller dynamically if it exists but for simplicity we rely on MainWindow binding
        # Wait, if we create a NEW controller, MainWindow won't be connected to it!
        # The best pattern is to expose the engine so MainWindow can bind it.
        # But Phase2N pattern: Panel emits replay_frame. Let's do that!
        if self._controller is not None:
             self._controller.pause()
             # Disconnect old signals
             self._controller.frame_ready.disconnect()
             self._controller.playback_finished.disconnect()

        self._controller = MissionController(self._engine)
        # Bind timeline updates
        self._controller.frame_ready.connect(self._on_frame_ready)
        self._controller.playback_finished.connect(self._on_finished)
        
        # Forward globally
        self._controller.frame_ready.connect(self.replay_frame)
        self._controller.reset_state.connect(self.reset_state)

        self._set_controls_enabled(True)
        self._total_frames = self._engine.total_frames()
        self.slider.setMinimum(0)
        self.slider.setMaximum(self._total_frames - 1)
        self.slider.setValue(0)
        
        # Load Analytics marking from events.json if possible (otherwise ignore)
        events_path = run_dir / "events.json"
        events_dict = {}
        if events_path.exists():
            try:
                with open(events_path, "r") as f:
                    event_data = json.load(f)
                    for e in event_data:
                        # map timestamp roughly to frame if stored, here we estimate
                        frame_idx = int(e.get("timestamp_s", 0) * 60) # assuming 60fps
                        frame_idx = min(frame_idx, self._total_frames - 1)
                        typ = e.get("event_type", "")
                        if "collapse" in typ:
                            events_dict[frame_idx] = "collapse"
                        elif "cascade" in typ:
                            events_dict[frame_idx] = "cascade"
                        elif "anomaly" in typ:
                            events_dict[frame_idx] = "anomaly"
            except Exception:
                pass
                
        self.slider.events = events_dict
        self.slider.update()

        # Emit first frame immediately
        self._controller.seek(0)
        self._update_label(0)

    def _set_controls_enabled(self, e: bool) -> None:
        self.slider.setEnabled(e)
        self.btn_play.setEnabled(e)
        self.btn_pause.setEnabled(e)
        self.btn_stop.setEnabled(e)
        self.cb_speed.setEnabled(e)

    def _on_play(self) -> None:
        if self._controller:
            self._controller.start()

    def _on_pause(self) -> None:
        if self._controller:
            self._controller.pause()

    def _on_stop(self) -> None:
        if self._controller:
            self._controller.stop()
            self.slider.blockSignals(True)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self._update_label(0)

    def _on_speed_changed(self) -> None:
        if not self._controller:
            return
        text = self.cb_speed.currentText()
        mult = float(text.replace("x", ""))
        self._controller.set_speed(mult)

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        if self._controller:
            self._engine.pause()

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        if self._controller and self._engine.state == PlaybackState.PLAYING:
            self._controller.start()

    def _on_slider_changed(self, value: int) -> None:
        self._pending_seek = value
        self._seek_timer.start()

    def _do_seek(self) -> None:
        if self._controller:
            self._controller.seek(self._pending_seek)

    def _on_frame_ready(self, frame) -> None:
        idx = self._engine.index
        if not self._slider_dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
        self._update_label(idx)

    def _update_label(self, idx: int) -> None:
        self.lbl_frame.setText(f"Frame: {idx} / {self._total_frames - 1}")

    def _on_finished(self) -> None:
        self._on_stop()
