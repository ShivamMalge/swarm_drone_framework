"""
TelemetryExporter — Research-grade asynchronous export pipeline (Phase 2M).

Observer layer. Consumes TelemetryFrame objects accumulated during an experiment
and writes them to disk in CSV, JSON, and Parquet formats inside a unique
run directory.

Architecture:
  TelemetryBridge → frame_ready → ExportManager.record_frame()
  experiment_finished → ExportManager.flush()  (background thread)

Threading:
  All disk I/O runs in a background QThread to avoid blocking the GUI
  or the SimulationWorker.

Data integrity:
  - Frames stored in chronological order (append-only deque)
  - No mutation of TelemetryFrame objects
  - No recomputation of metrics
  - Config snapshot frozen at experiment start via dataclasses.asdict()
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import SimConfig
from src.telemetry.telemetry_frame import TelemetryFrame

# Optional parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False


# ── Helpers ──────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _generate_run_id(scenario: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{scenario}_s{seed}"


def _config_to_dict(cfg: SimConfig) -> dict[str, Any]:
    """Produce a JSON-safe snapshot of the frozen SimConfig."""
    d = dataclasses.asdict(cfg)
    # Remove non-serializable fields
    d.pop("regime", None)
    return d


# ── Frame → flat row ────────────────────────────────────────

def _frame_to_row(frame: TelemetryFrame, n_agents: int) -> dict[str, Any]:
    """Convert a TelemetryFrame to a flat dictionary for tabular export."""
    row: dict[str, Any] = {
        "time": frame.time,
        "spectral_gap": frame.spectral_gap,
        "consensus_variance": frame.consensus_variance,
        "packet_drop_rate": frame.packet_drop_rate,
        "latency": frame.latency,
        "n_components": len(frame.connected_components),
        "alive_agents": int((~frame.drone_failure_flags).sum()),
    }

    # Per-agent columns (positions, energies, failure)
    for i in range(n_agents):
        row[f"pos_x_{i}"] = float(frame.positions[i, 0]) if i < len(frame.positions) else 0.0
        row[f"pos_y_{i}"] = float(frame.positions[i, 1]) if i < len(frame.positions) else 0.0
        row[f"energy_{i}"] = float(frame.energies[i]) if i < len(frame.energies) else 0.0
        row[f"failed_{i}"] = bool(frame.drone_failure_flags[i]) if i < len(frame.drone_failure_flags) else True

    # Regime (per-agent string)
    for i in range(n_agents):
        row[f"regime_{i}"] = frame.regime_state.get(i, "UNKNOWN")

    return row


# ── Core Exporter ────────────────────────────────────────────

class TelemetryExporter:
    """
    Accumulates TelemetryFrames and exports them asynchronously.

    Thread-safe: ``record_frame`` may be called from any thread.
    ``flush`` spawns a daemon thread for disk I/O.
    """

    def __init__(
        self,
        output_root: str | Path = "outputs",
        max_buffer: int = 10000,
    ) -> None:
        self._output_root = Path(output_root)
        self._max_buffer = max_buffer

        self._frames: deque[TelemetryFrame] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()

        self._run_id: str = ""
        self._scenario: str = ""
        self._config_snapshot: dict[str, Any] = {}
        self._start_time: float = 0.0

        # Callbacks
        self.on_export_started: list = []
        self.on_export_completed: list = []

    # ── Experiment lifecycle ─────────────────────────────────

    def begin_experiment(
        self, scenario: str, config: SimConfig
    ) -> str:
        """Freeze config, generate run_id, reset buffer."""
        with self._lock:
            self._frames.clear()
        self._scenario = scenario
        self._config_snapshot = _config_to_dict(config)
        self._run_id = _generate_run_id(scenario, config.seed)
        self._start_time = time.time()
        return self._run_id

    def record_frame(self, frame: TelemetryFrame) -> None:
        """Append frame (thread-safe, bounded)."""
        with self._lock:
            self._frames.append(frame)

    def flush(self, formats: tuple[str, ...] = ("csv", "json", "parquet")) -> None:
        """Export accumulated frames to disk in a background thread."""
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()

        if not frames:
            return

        run_id = self._run_id
        for cb in self.on_export_started:
            cb(run_id)

        t = threading.Thread(
            target=self._write_all,
            args=(frames, run_id, formats),
            daemon=True,
        )
        t.start()

    # ── Disk I/O (runs on background thread) ─────────────────

    def _write_all(
        self,
        frames: list[TelemetryFrame],
        run_id: str,
        formats: tuple[str, ...],
    ) -> None:
        run_dir = self._output_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        n_agents = len(frames[0].positions) if frames else 0

        # ── metadata.json ─────────────────────────────────────
        metadata = {
            "run_id": run_id,
            "scenario_name": self._scenario,
            "seed": self._config_snapshot.get("seed", -1),
            "start_time": self._start_time,
            "end_time": time.time(),
            "duration": frames[-1].time - frames[0].time if len(frames) > 1 else 0.0,
            "total_agents": n_agents,
            "total_frames": len(frames),
            "config_snapshot": self._config_snapshot,
        }

        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, cls=_NumpyEncoder)

        # ── Convert frames to rows ────────────────────────────
        rows = [_frame_to_row(fr, n_agents) for fr in frames]

        if "csv" in formats:
            self._write_csv(run_dir / "telemetry.csv", rows)
        if "json" in formats:
            self._write_json(run_dir / "telemetry.json", frames, n_agents)
        if "parquet" in formats and _HAS_PARQUET:
            self._write_parquet(run_dir / "telemetry.parquet", rows)

        path = str(run_dir)

        # Update runs_index.json
        self._update_runs_index(run_id, path)

        for cb in self.on_export_completed:
            cb(run_id, path)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    @staticmethod
    def _write_json(path: Path, frames: list[TelemetryFrame], n_agents: int) -> None:
        data = []
        for fr in frames:
            entry = {
                "time": fr.time,
                "positions": fr.positions.tolist(),
                "energies": fr.energies.tolist(),
                "spectral_gap": fr.spectral_gap,
                "consensus_variance": fr.consensus_variance,
                "packet_drop_rate": fr.packet_drop_rate,
                "latency": fr.latency,
                "connected_components": fr.connected_components,
                "regime_state": {str(k): v for k, v in fr.regime_state.items()},
                "adaptive_parameters": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in fr.adaptive_parameters.items()
                },
                "drone_failure_flags": fr.drone_failure_flags.tolist(),
            }
            data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=1, cls=_NumpyEncoder)

    @staticmethod
    def _write_parquet(path: Path, rows: list[dict]) -> None:
        if not rows or not _HAS_PARQUET:
            return
        table = pa.Table.from_pydict(
            {k: [r[k] for r in rows] for k in rows[0].keys()}
        )
        pq.write_table(table, str(path), compression="snappy")

    def _update_runs_index(self, run_id: str, run_path: str) -> None:
        idx_path = self._output_root / "runs_index.json"
        entries: list[dict] = []
        if idx_path.exists():
            try:
                with open(idx_path) as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, OSError):
                entries = []

        entries.append({
            "run_id": run_id,
            "scenario": self._scenario,
            "timestamp": time.time(),
            "path": run_path,
        })

        with open(idx_path, "w") as f:
            json.dump(entries, f, indent=2)

    # ── Accessors ─────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)
