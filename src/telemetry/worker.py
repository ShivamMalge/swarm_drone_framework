"""
SimulationWorker — QThread executing the DES kernel in fixed-timestep
deterministic mode.

Architecture contract:
  - Runs ONLY inside a QThread; GUI thread is never blocked.
  - Advances the kernel via ``kernel.run(until=t+dt)`` — fixed steps.
  - Extracts read-only telemetry after every step via TelemetryEmitter.
  - Pushes frames into TelemetryBuffer (single-slot, non-blocking).
  - NEVER modifies kernel state, injects events, or alters agent logic.
  - NEVER imports GUI elements or calls GUI methods.
  - All randomness lives inside the kernel RNG streams; worker loop
    is fully deterministic.

Pipeline:
  DES Engine → Worker → TelemetryEmitter → TelemetryBuffer → (GUI reads)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import time

from PySide6.QtCore import QThread, Signal

from src.core.config import SimConfig
from src.simulation import Phase1Simulation
from src.telemetry.telemetry_emitter import TelemetryEmitter
from src.telemetry.telemetry_buffer import TelemetryBuffer
from src.telemetry.exporter import TelemetryExporter

if TYPE_CHECKING:
    pass


class SimulationWorker(QThread):
    """
    Background worker that steps the DES kernel at fixed ``frame_dt``
    intervals and streams telemetry into a shared buffer.

    Parameters
    ----------
    simulation_config : SimConfig
        Frozen, immutable simulation configuration.
    telemetry_buffer : TelemetryBuffer
        Single-slot buffer consumed by the GUI main thread.
    frame_dt : float
        Fixed simulation time increment per step (default 0.05).
    """

    # ── Signals (consumed by GUI via slots) ──────────────────
    simulation_started = Signal()
    simulation_paused = Signal()
    simulation_resumed = Signal()
    simulation_stopped = Signal()
    simulation_finished = Signal()

    def __init__(
        self,
        simulation_config: SimConfig,
        telemetry_buffer: TelemetryBuffer,
        frame_dt: float = 0.05,
        exporter: TelemetryExporter | None = None,
    ) -> None:
        super().__init__()
        self._cfg = simulation_config
        self._buffer = telemetry_buffer
        self._frame_dt = frame_dt
        self._exporter = exporter

        # ── Internal mutable control state ───────────────────
        self._running: bool = False
        self._paused: bool = False
        self._current_time: float = 0.0
        self._step_requested: bool = False
        self._speed_multiplier: float = 1.0

        # ── Simulation + emitter (created here, used only in run()) ─
        self._sim: Phase1Simulation | None = None
        self._emitter: TelemetryEmitter | None = None

    # ── QThread entry point ──────────────────────────────────

    def run(self) -> None:
        """
        Core execution loop.  Advances the kernel in fixed steps of
        ``frame_dt``, extracts a TelemetryFrame after each step, and
        pushes it into the buffer.

        Time progression: 0.00 → 0.05 → 0.10 → 0.15 → …
        """
        # Build simulation and emitter inside the worker thread
        self._sim = Phase1Simulation(self._cfg)
        self._sim.seed_events()
        self._emitter = TelemetryEmitter(self._sim, self._cfg)
        self._current_time = 0.0
        self._running = True

        self.simulation_started.emit()

        max_time = self._cfg.max_time

        while self._running and self._current_time < max_time:

            # ── Pause gate ───────────────────────────────────
            if self._paused and not self._step_requested:
                QThread.msleep(5)
                continue

            # ── Deterministic fixed-step advance ─────────────
            target_time = min(self._current_time + self._frame_dt, max_time)

            try:
                self._sim.kernel.run(until=target_time)
            except Exception as exc:
                print(f"[SimulationWorker] Kernel error at t={target_time:.4f}: {exc}")
                self._running = False
                break

            self._current_time = target_time

            # ── Read-only telemetry extraction ───────────────
            frame = self._emitter.extract_frame()
            self._buffer.push(frame)
            if self._exporter is not None:
                self._exporter.record_frame(frame)

            if self._step_requested:
                self._step_requested = False

            # ── CPU yield / Speed pacing ─────────────────────
            sleep_ms = int(max(1.0, 16.0 / self._speed_multiplier))
            QThread.msleep(sleep_ms)

        # ── Natural completion ───────────────────────────────
        self._running = False
        self.simulation_finished.emit()

    # ── Control methods (called from GUI thread) ─────────────

    def start_simulation(self) -> None:
        """Begin (or restart) the worker thread."""
        if not self.isRunning():
            self._running = True
            self._paused = False
            self.start()  # QThread.start() → calls run()

    def pause_simulation(self) -> None:
        """Pause the simulation loop (non-destructive)."""
        self._paused = True
        self.simulation_paused.emit()

    def resume_simulation(self) -> None:
        """Resume from pause."""
        self._paused = False
        self.simulation_resumed.emit()

    def stop_simulation(self) -> None:
        """Gracefully terminate the worker loop and wait."""
        self._running = False
        self._paused = False
        self.wait()
        self.simulation_stopped.emit()

    def reset_simulation(self) -> None:
        """Stop, discard state, and prepare for a fresh run."""
        self.stop_simulation()
        self._sim = None
        self._emitter = None
        self._current_time = 0.0
        self._step_requested = False
        self._buffer.clear()

    def step_simulation(self) -> None:
        """Execute exactly one step if paused."""
        if self._paused:
            self._step_requested = True

    def set_speed(self, speed: float) -> None:
        """Adjust pacing multiplier without affecting DES kernel determinism."""
        self._speed_multiplier = max(0.1, speed)

    # ── Disturbance Injection (Phase 2K) ─────────────────────

    def set_jamming(self, enabled: bool) -> None:
        if self._sim and hasattr(self._sim, "interference"):
            # Toggling could enable/disable the interference block entirely
            pass

    def set_interference(self, strength: float) -> None:
        if self._sim and hasattr(self._sim, "interference"):
            self._sim.interference.psi_max = strength * self._cfg.psi_max

    def set_energy_drain(self, enabled: bool) -> None:
        # State toggle for drain logic
        pass

    def set_drain_rate(self, rate: float) -> None:
        # Safely mutating frozen config for future reads, ideally restore on reset
        object.__setattr__(self._cfg, "p_idle", rate)

    def set_connectivity_drop(self, enabled: bool) -> None:
        drop = 0.8 if enabled else 0.05
        object.__setattr__(self._cfg, "p_drop_base", drop)

    # ── Experiment Runner (Phase 2L) ─────────────────────────

    experiment_finished = Signal()

    _EXPERIMENT_CONFIGS = {
        "baseline": {},
        "percolation_collapse": {"connectivity_drop": True},
        "energy_cascade": {"drain_rate": 5.0},
        "jamming_attack": {"interference": 0.7},
    }

    def run_experiment(self, name: str) -> None:
        """Reset, apply scenario config, and start simulation."""
        self.reset_simulation()

        # Begin export session
        if self._exporter is not None:
            self._exporter.begin_experiment(name, self._cfg)

        cfg = self._EXPERIMENT_CONFIGS.get(name, {})

        if cfg.get("connectivity_drop"):
            self.set_connectivity_drop(True)
        if "drain_rate" in cfg:
            self.set_drain_rate(cfg["drain_rate"])
        if "interference" in cfg:
            self.set_interference(cfg["interference"])

        self.start_simulation()

    def run_custom_scenario(self, scenario_config) -> None:
        """Run a visually built scenario configuration (Phase 2P)."""
        self.reset_simulation()
        
        # Override Base Config
        self._cfg = scenario_config.to_sim_config(self._cfg)

        run_name = f"{scenario_config.name}_{int(time.time())}"
        
        if self._exporter is not None:
            self._exporter.begin_experiment(run_name, self._cfg)
            
        # Optional: Add any extra scenario hooks here (e.g. interference zones)
        
        self.start_simulation()

    def stop_experiment(self) -> None:
        """Stop current experiment, flush export, and restore defaults."""
        self.stop_simulation()
        if self._exporter is not None:
            self._exporter.flush()
        self.set_connectivity_drop(False)
        self.set_interference(0.0)
        self.set_drain_rate(self._cfg.p_idle)

    # ── Read-only inspection (safe from any thread) ──────────

    @property
    def current_time(self) -> float:
        """Current simulation clock value."""
        return self._current_time

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_running(self) -> bool:
        return self._running
