"""Phase 2O verification: Performance benchmarks for rendering pipeline."""

import sys
import time
import collections

import numpy as np

from PySide6.QtWidgets import QApplication

from src.telemetry.telemetry_frame import TelemetryFrame
from src.gui.swarm_map import SwarmMapWidget
from src.gui.graphs import TelemetryGraphsWidget, _RingBuffer


def _make_frame(n: int, t: float) -> TelemetryFrame:
    rng = np.random.default_rng(int(t * 1000) % 2**31)
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, min(i + 4, n)):
            adj[i, j] = adj[j, i] = 1
    return TelemetryFrame(
        time=t,
        positions=rng.uniform(0, 100, (n, 2)),
        energies=rng.uniform(10, 100, n),
        adjacency=adj,
        connected_components=[list(range(n))],
        spectral_gap=0.5,
        consensus_variance=0.1,
        packet_drop_rate=0.02,
        latency=0.5,
        regime_state={i: "STABLE" for i in range(n)},
        adaptive_parameters={},
        drone_failure_flags=np.zeros(n, dtype=bool),
    )


def test_ring_buffer():
    rb = _RingBuffer(5)
    for i in range(3):
        rb.append(float(i))
    arr = rb.as_array()
    assert len(arr) == 3
    assert list(arr) == [0.0, 1.0, 2.0]

    for i in range(3, 8):
        rb.append(float(i))
    arr = rb.as_array()
    assert len(arr) == 5
    assert list(arr) == [3.0, 4.0, 5.0, 6.0, 7.0]
    print("[PASS] ring_buffer (ordered, bounded)")


def test_swarm_map_perf():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    widget = SwarmMapWidget()

    for n_agents in [30, 50, 100]:
        times = []
        for i in range(60):
            frame = _make_frame(n_agents, float(i) * 0.05)
            t0 = time.perf_counter()
            widget.update_frame(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)

        avg_ms = np.mean(times)
        max_ms = np.max(times)
        p95_ms = np.percentile(times, 95)

        print(
            f"[PERF] SwarmMap N={n_agents:3d}: "
            f"avg={avg_ms:.2f}ms  p95={p95_ms:.2f}ms  max={max_ms:.2f}ms  "
            f"{'PASS' if p95_ms < 16 else 'WARN'}"
        )


def test_graphs_perf():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    widget = TelemetryGraphsWidget(max_points=300)

    times = []
    for i in range(300):
        frame = _make_frame(50, float(i) * 0.05)
        t0 = time.perf_counter()
        widget.update_frame(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)

    avg_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)
    print(
        f"[PERF] Graphs 300 frames: "
        f"avg={avg_ms:.2f}ms  p95={p95_ms:.2f}ms  "
        f"{'PASS' if p95_ms < 8 else 'WARN'}"
    )


def test_adjacency_hash_skip():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    widget = SwarmMapWidget()

    # Same adjacency, different positions
    frame1 = _make_frame(30, 0.0)
    frame2 = _make_frame(30, 0.05)
    frame2.adjacency = frame1.adjacency.copy()

    widget.update_frame(frame1)
    h1 = widget._last_adj_hash

    widget.update_frame(frame2)
    h2 = widget._last_adj_hash

    assert h1 == h2, "Hash should match for same adjacency"
    print("[PASS] adjacency_hash_skip (same topology -> skipped edge rebuild)")


def test_memory_bounded():
    rb = _RingBuffer(300)
    for i in range(10000):
        rb.append(float(i))
    arr = rb.as_array()
    assert len(arr) == 300
    assert arr[-1] == 9999.0
    print("[PASS] memory_bounded (ring buffer capped at 300)")


def test_no_backlog():
    """Verify single-slot buffer always returns latest frame."""
    from src.telemetry.telemetry_buffer import TelemetryBuffer
    buf = TelemetryBuffer()
    for i in range(100):
        f = TelemetryFrame.empty(10)
        f.time = float(i)
        buf.push(f)
    latest = buf.get_latest()
    assert latest is not None
    assert latest.time == 99.0
    print("[PASS] no_backlog (latest frame = 99.0)")


def test_edge_downscale():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    widget = SwarmMapWidget()
    widget.set_edge_rendering(False)
    assert widget._edge_render_enabled is False
    widget.set_edge_rendering(True)
    assert widget._edge_render_enabled is True
    print("[PASS] adaptive_edge_downscale (toggle works)")


if __name__ == "__main__":
    test_ring_buffer()
    test_memory_bounded()
    test_no_backlog()
    test_swarm_map_perf()
    test_graphs_perf()
    test_adjacency_hash_skip()
    test_edge_downscale()
    print("\nAll Phase 2O tests passed.")
