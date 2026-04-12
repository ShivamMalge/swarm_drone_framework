"""Phase 2N verification: Replay system end-to-end."""

import json
import os
import shutil
import sys
import time

import numpy as np

from src.core.config import SimConfig
from src.telemetry.telemetry_frame import TelemetryFrame
from src.telemetry.exporter import TelemetryExporter
from src.replay.replay_loader import ReplayLoader
from src.replay.replay_engine import ReplayEngine, PlaybackState


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "_test_replay"
)


def _make_frames(n_agents: int, count: int) -> list[TelemetryFrame]:
    frames = []
    rng = np.random.default_rng(99)
    for i in range(count):
        f = TelemetryFrame.empty(n_agents)
        f.time = float(i) * 0.05
        f.positions = rng.uniform(0, 100, (n_agents, 2))
        f.energies = np.full(n_agents, max(100.0 - i * 2.0, 0.0))
        f.spectral_gap = max(0.5 - i * 0.02, 0.0)
        f.consensus_variance = 0.1 * i
        f.packet_drop_rate = 0.02
        f.latency = 0.5
        f.regime_state = {j: "STABLE" for j in range(n_agents)}
        if i > 15:
            f.regime_state = {j: "FRAGMENTED" for j in range(n_agents)}
        f.adaptive_parameters = {"coverage_gains": np.full(n_agents, 0.4)}
        f.drone_failure_flags = np.zeros(n_agents, dtype=bool)
        if i > 20:
            f.drone_failure_flags[0] = True
        frames.append(f)
    return frames


def test_replay():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # ── 1. Export test data ───────────────────────────────────
    cfg = SimConfig(num_agents=5, max_time=5.0, seed=99)
    exporter = TelemetryExporter(output_root=OUTPUT_DIR)
    run_id = exporter.begin_experiment("test_scenario", cfg)

    frames = _make_frames(5, 30)
    for f in frames:
        exporter.record_frame(f)
    exporter.flush()
    time.sleep(3.0)

    # Find run dir
    run_dir = os.path.join(OUTPUT_DIR, f"run_{run_id}")
    assert os.path.exists(run_dir), f"Run dir missing: {run_dir}"
    print(f"[PASS] export_created ({run_id})")

    # ── 2. Load via ReplayLoader ─────────────────────────────
    loader = ReplayLoader(run_dir)
    loaded = loader.load()
    assert len(loaded) == 30
    print(f"[PASS] loader_frames (count={len(loaded)})")

    # ── 3. Frame ordering preserved ──────────────────────────
    for i in range(1, len(loaded)):
        assert loaded[i].time >= loaded[i - 1].time
    print("[PASS] chronological_order")

    # ── 4. Metadata consistency ──────────────────────────────
    assert loader.metadata.run_id == run_id
    assert loader.metadata.scenario_name == "test_scenario"
    assert loader.metadata.seed == 99
    assert loader.metadata.total_agents == 5
    assert loader.metadata.total_frames == 30
    print("[PASS] metadata_consistency")

    # ── 5. ReplayEngine: play / pause / seek ─────────────────
    engine = ReplayEngine(loaded)
    assert engine.state == PlaybackState.STOPPED
    assert engine.total_frames == 30

    engine.play()
    assert engine.state == PlaybackState.PLAYING

    f1 = engine.next_frame()
    assert f1 is not None
    assert f1.time == 0.0
    assert engine.index == 1
    print("[PASS] engine_play_next")

    engine.pause()
    assert engine.state == PlaybackState.PAUSED
    f_none = engine.next_frame()
    assert f_none is None
    print("[PASS] engine_pause")

    engine.seek(10)
    assert engine.index == 10
    engine.play()
    f10 = engine.next_frame()
    assert abs(f10.time - 0.5) < 1e-9
    print("[PASS] engine_seek")

    engine.stop()
    assert engine.index == 0
    assert engine.state == PlaybackState.STOPPED
    print("[PASS] engine_stop_reset")

    # ── 6. Speed clamping ────────────────────────────────────
    engine.set_speed(0.1)
    assert engine.speed == 0.25
    engine.set_speed(100.0)
    assert engine.speed == 8.0
    engine.set_speed(2.0)
    assert engine.speed == 2.0
    print("[PASS] speed_clamping")

    # ── 7. No frame mutation ─────────────────────────────────
    orig_time = loaded[5].time
    engine.seek(5)
    engine.play()
    _ = engine.next_frame()
    assert loaded[5].time == orig_time
    print("[PASS] no_frame_mutation")

    # ── 8. Event detection ───────────────────────────────────
    engine2 = ReplayEngine(loaded)
    events = engine2.detect_events()
    regime_events = [e for e in events if e["type"] == "regime_transition"]
    death_events = [e for e in events if e["type"] == "agent_death"]
    assert len(regime_events) > 0, "Expected regime transition events"
    assert len(death_events) > 0, "Expected agent death events"
    print(f"[PASS] event_detection (regime={len(regime_events)}, deaths={len(death_events)})")

    # ── 9. runs_index.json ───────────────────────────────────
    idx_path = os.path.join(OUTPUT_DIR, "runs_index.json")
    assert os.path.exists(idx_path)
    with open(idx_path) as f:
        idx = json.load(f)
    assert len(idx) >= 1
    assert idx[-1]["run_id"] == run_id
    print("[PASS] runs_index_json")

    # ── 10. Determinism: reload → identical frames ───────────
    loader2 = ReplayLoader(run_dir)
    loaded2 = loader2.load()
    for a, b in zip(loaded, loaded2):
        assert a.time == b.time
        assert np.array_equal(a.positions, b.positions)
        assert np.array_equal(a.energies, b.energies)
    print("[PASS] deterministic_reload")

    # Cleanup
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    print("\nAll Phase 2N tests passed.")


if __name__ == "__main__":
    test_replay()
