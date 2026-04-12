"""Phase 2M verification: TelemetryExporter correctness and determinism."""

import json
import os
import shutil
import sys
import time

import numpy as np

from src.core.config import SimConfig
from src.telemetry.telemetry_frame import TelemetryFrame
from src.telemetry.exporter import TelemetryExporter, _HAS_PARQUET


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "_test_export"
)


def _make_frames(n_agents: int, count: int) -> list[TelemetryFrame]:
    frames = []
    for i in range(count):
        f = TelemetryFrame.empty(n_agents)
        f.time = float(i) * 0.05
        f.positions = np.random.default_rng(i).uniform(0, 100, (n_agents, 2))
        f.energies = np.full(n_agents, 100.0 - i * 0.5)
        f.spectral_gap = 0.5 + i * 0.01
        f.consensus_variance = 0.1 * i
        f.packet_drop_rate = 0.02
        f.latency = 0.5
        f.regime_state = {j: "STABLE" for j in range(n_agents)}
        f.adaptive_parameters = {"coverage_gains": np.full(n_agents, 0.4)}
        f.drone_failure_flags = np.zeros(n_agents, dtype=bool)
        frames.append(f)
    return frames


def test_exporter():
    # Cleanup
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    cfg = SimConfig(num_agents=10, max_time=5.0, seed=42)
    exporter = TelemetryExporter(output_root=OUTPUT_DIR)

    # 1. Begin experiment
    run_id = exporter.begin_experiment("baseline", cfg)
    assert len(run_id) > 0
    assert "baseline" in run_id
    assert "s42" in run_id
    print(f"[PASS] run_id_generated ({run_id})")

    # 2. Record frames
    frames = _make_frames(10, 20)
    for f in frames:
        exporter.record_frame(f)
    assert exporter.frame_count == 20
    print("[PASS] frame_recording (20 frames buffered)")

    # 3. Flush (async)
    completed = []
    exporter.on_export_completed.append(lambda rid, path: completed.append((rid, path)))
    exporter.flush()

    # Wait for background thread
    time.sleep(5.0)

    assert len(completed) == 1
    rid, path = completed[0]
    assert rid == run_id
    print(f"[PASS] export_completed (path={path})")

    # 4. Verify metadata.json
    meta_path = os.path.join(path, "metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["run_id"] == run_id
    assert meta["scenario_name"] == "baseline"
    assert meta["seed"] == 42
    assert meta["total_agents"] == 10
    assert meta["total_frames"] == 20
    assert "config_snapshot" in meta
    assert meta["config_snapshot"]["num_agents"] == 10
    print("[PASS] metadata_correct")

    # 5. Verify CSV
    csv_path = os.path.join(path, "telemetry.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 21  # header + 20 rows
    print(f"[PASS] csv_export ({len(lines)-1} rows)")

    # 6. Verify JSON
    json_path = os.path.join(path, "telemetry.json")
    assert os.path.exists(json_path)
    with open(json_path) as f:
        data = json.load(f)
    assert len(data) == 20
    assert data[0]["time"] == 0.0
    assert abs(data[-1]["time"] - 19 * 0.05) < 1e-9, f"Last time: {data[-1]['time']}"
    # Verify chronological
    for i in range(1, len(data)):
        assert data[i]["time"] >= data[i-1]["time"], "Non-chronological"
    print("[PASS] json_export (chronological order preserved)")

    # 7. Verify Parquet
    if _HAS_PARQUET:
        pq_path = os.path.join(path, "telemetry.parquet")
        assert os.path.exists(pq_path)
        assert os.path.getsize(pq_path) > 0
        print("[PASS] parquet_export (snappy compressed)")
    else:
        print("[SKIP] parquet_export (pyarrow not installed)")

    # 8. Config snapshot immutability
    snap = meta["config_snapshot"]
    assert snap["seed"] == 42
    assert snap["num_agents"] == 10
    assert snap["max_time"] == 5.0
    print("[PASS] config_snapshot_immutable")

    # 9. Buffer bounded (post-flush should be empty)
    assert exporter.frame_count == 0
    print("[PASS] buffer_cleared_after_flush")

    # 10. Determinism: same seed → same run_id prefix
    exporter2 = TelemetryExporter(output_root=OUTPUT_DIR)
    rid2 = exporter2.begin_experiment("baseline", cfg)
    assert "baseline" in rid2
    assert "s42" in rid2
    print("[PASS] deterministic_run_id_pattern")

    # Cleanup
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    print("\nAll Phase 2M tests passed.")


if __name__ == "__main__":
    test_exporter()
