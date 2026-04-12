"""
ReplayLoader — Deserialises exported run data into TelemetryFrame sequences.

Supports Parquet (primary), JSON, and CSV sources produced by Phase 2M.
Strictly read-only: no metric recomputation, no frame mutation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.telemetry.telemetry_frame import TelemetryFrame

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False


@dataclass
class RunMetadata:
    """Frozen snapshot of experiment metadata loaded from metadata.json."""
    run_id: str = ""
    scenario_name: str = ""
    seed: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    total_agents: int = 0
    total_frames: int = 0
    config_snapshot: dict[str, Any] = field(default_factory=dict)


class ReplayLoader:
    """
    Loads an exported run directory and yields TelemetryFrames in
    strict chronological order.

    Priority: parquet > json > csv
    """

    def __init__(self, run_dir: str | Path) -> None:
        self._dir = Path(run_dir)
        self.metadata = RunMetadata()
        self._frames: list[TelemetryFrame] = []

    # ── Public API ───────────────────────────────────────────

    def load(self) -> list[TelemetryFrame]:
        """Load metadata + frames. Returns ordered frame list."""
        self._load_metadata()
        self._load_frames()
        self._validate()
        return self._frames

    @property
    def frames(self) -> list[TelemetryFrame]:
        return self._frames

    # ── Metadata ─────────────────────────────────────────────

    def _load_metadata(self) -> None:
        meta_path = self._dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self._dir}")
        with open(meta_path) as f:
            raw = json.load(f)
        self.metadata = RunMetadata(
            run_id=raw.get("run_id", ""),
            scenario_name=raw.get("scenario_name", ""),
            seed=raw.get("seed", 0),
            start_time=raw.get("start_time", 0.0),
            end_time=raw.get("end_time", 0.0),
            duration=raw.get("duration", 0.0),
            total_agents=raw.get("total_agents", 0),
            total_frames=raw.get("total_frames", 0),
            config_snapshot=raw.get("config_snapshot", {}),
        )

    # ── Frame loading (priority: parquet > json > csv) ───────

    def _load_frames(self) -> None:
        pq_path = self._dir / "telemetry.parquet"
        json_path = self._dir / "telemetry.json"
        csv_path = self._dir / "telemetry.csv"

        if pq_path.exists() and _HAS_PARQUET:
            self._load_parquet(pq_path)
        elif json_path.exists():
            self._load_json(json_path)
        elif csv_path.exists():
            self._load_csv(csv_path)
        else:
            raise FileNotFoundError(f"No telemetry data in {self._dir}")

    def _load_parquet(self, path: Path) -> None:
        table = pq.read_table(str(path))
        df = table.to_pydict()
        n_agents = self.metadata.total_agents
        n_rows = len(df["time"])

        for i in range(n_rows):
            self._frames.append(self._row_to_frame(df, i, n_agents))

    def _load_json(self, path: Path) -> None:
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            n = len(entry["positions"])
            frame = TelemetryFrame(
                time=entry["time"],
                positions=np.array(entry["positions"], dtype=np.float64),
                energies=np.array(entry["energies"], dtype=np.float64),
                adjacency=np.zeros((n, n), dtype=np.uint8),  # not stored in JSON
                connected_components=entry.get("connected_components", [list(range(n))]),
                spectral_gap=entry["spectral_gap"],
                consensus_variance=entry["consensus_variance"],
                packet_drop_rate=entry["packet_drop_rate"],
                latency=entry["latency"],
                regime_state={int(k): v for k, v in entry.get("regime_state", {}).items()},
                adaptive_parameters={
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in entry.get("adaptive_parameters", {}).items()
                },
                drone_failure_flags=np.array(entry["drone_failure_flags"], dtype=bool),
            )
            self._frames.append(frame)

    def _load_csv(self, path: Path) -> None:
        n = self.metadata.total_agents
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                positions = np.array(
                    [[float(row.get(f"pos_x_{i}", 0)), float(row.get(f"pos_y_{i}", 0))]
                     for i in range(n)], dtype=np.float64,
                )
                energies = np.array(
                    [float(row.get(f"energy_{i}", 0)) for i in range(n)], dtype=np.float64,
                )
                failures = np.array(
                    [row.get(f"failed_{i}", "True") == "True" for i in range(n)], dtype=bool,
                )
                regime = {i: row.get(f"regime_{i}", "UNKNOWN") for i in range(n)}

                frame = TelemetryFrame(
                    time=float(row["time"]),
                    positions=positions,
                    energies=energies,
                    adjacency=np.zeros((n, n), dtype=np.uint8),
                    connected_components=[list(range(n))],
                    spectral_gap=float(row["spectral_gap"]),
                    consensus_variance=float(row["consensus_variance"]),
                    packet_drop_rate=float(row["packet_drop_rate"]),
                    latency=float(row["latency"]),
                    regime_state=regime,
                    adaptive_parameters={},
                    drone_failure_flags=failures,
                )
                self._frames.append(frame)

    # ── Parquet row → frame ──────────────────────────────────

    def _row_to_frame(self, df: dict, idx: int, n: int) -> TelemetryFrame:
        positions = np.array(
            [[df[f"pos_x_{i}"][idx], df[f"pos_y_{i}"][idx]] for i in range(n)],
            dtype=np.float64,
        )
        energies = np.array(
            [df[f"energy_{i}"][idx] for i in range(n)], dtype=np.float64,
        )
        failures = np.array(
            [df[f"failed_{i}"][idx] for i in range(n)], dtype=bool,
        )
        regime = {i: df.get(f"regime_{i}", ["UNKNOWN"] * len(df["time"]))[idx] for i in range(n)}

        return TelemetryFrame(
            time=df["time"][idx],
            positions=positions,
            energies=energies,
            adjacency=np.zeros((n, n), dtype=np.uint8),
            connected_components=[list(range(n))],
            spectral_gap=df["spectral_gap"][idx],
            consensus_variance=df["consensus_variance"][idx],
            packet_drop_rate=df["packet_drop_rate"][idx],
            latency=df["latency"][idx],
            regime_state=regime,
            adaptive_parameters={},
            drone_failure_flags=failures,
        )

    # ── Validation ───────────────────────────────────────────

    def _validate(self) -> None:
        expected = self.metadata.total_frames
        actual = len(self._frames)
        if expected > 0 and actual != expected:
            import warnings
            warnings.warn(
                f"Frame count mismatch: metadata={expected}, loaded={actual}",
                stacklevel=2,
            )
        # Chronological check
        for i in range(1, len(self._frames)):
            if self._frames[i].time < self._frames[i - 1].time:
                raise ValueError(
                    f"Non-chronological frames at index {i}: "
                    f"{self._frames[i-1].time} > {self._frames[i].time}"
                )
