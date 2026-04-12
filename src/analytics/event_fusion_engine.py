"""
event_fusion_engine.py — Deterministic cross-analyzer event fusion.

Consumes per-frame metrics from all analytical layers (2Q–2X) and
emits high-level SystemEvent objects.  All logic is pure-functional
with hash-skip optimisation and bounded memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.analytics.system_event import SystemEvent
from src.analytics.percolation_analyzer import PercolationMetrics
from src.analytics.spectral_analyzer import SpectralMetrics
from src.analytics.energy_cascade_analyzer import EnergyMetrics
from src.analytics.consensus_analyzer import ConsensusMetrics
from src.analytics.anomaly_detector import AnomalyMetrics
from src.analytics.swarm_health import HealthMetrics
from src.telemetry.telemetry_frame import TelemetryFrame


@dataclass
class _FusedSnapshot:
    """Lightweight struct holding the values the engine evaluates."""
    perc_ratio: float
    perc_state: str
    lambda2_norm: float
    spec_state: str
    energy_norm: float
    energy_rate: float
    energy_state: str
    consensus_var: float
    consensus_state: str
    anomaly_count: int
    health_score: float
    health_state: str


class EventFusionEngine:
    """
    Stateless-per-hash fusion engine.

    * Hash-skip:  identical metric snapshots → zero work.
    * Dedup window:  suppresses repeated events of the same type
      within a sliding window of 20 events.
    * Bounded storage:  event buffer capped at 200 entries.
    """

    # ── Thresholds (configurable but fixed at runtime) ──────
    PERC_COLLAPSE_THRESHOLD = 0.5
    SPEC_INSTABILITY_THRESHOLD = 0.1
    ENERGY_DRAIN_THRESHOLD = -0.5
    CONSENSUS_DIVERGE_THRESHOLD = 0.5
    ANOMALY_SPIKE_THRESHOLD = 3
    HEALTH_CRITICAL_THRESHOLD = 0.3
    RECOVERY_PERC_THRESHOLD = 0.7

    def __init__(self) -> None:
        self._last_hash: int = -1
        self._last_events: list[SystemEvent] = []
        self._recent_types: deque[str] = deque(maxlen=20)
        self._event_buffer: deque[SystemEvent] = deque(maxlen=200)

    # ── Public API ──────────────────────────────────────────

    @property
    def events(self) -> deque[SystemEvent]:
        return self._event_buffer

    def reset(self) -> None:
        self._last_hash = -1
        self._last_events = []
        self._recent_types.clear()
        self._event_buffer.clear()

    def analyze_frame(
        self,
        frame: TelemetryFrame,
        perc: PercolationMetrics,
        spec: SpectralMetrics,
        eng: EnergyMetrics,
        cns: ConsensusMetrics,
        anom: AnomalyMetrics,
        health: HealthMetrics,
        frame_index: int,
    ) -> list[SystemEvent]:
        """Return newly fused events for this frame (may be empty)."""

        # Build deterministic hash from the values we actually inspect
        snap = _FusedSnapshot(
            perc_ratio=round(perc.connectivity_ratio, 4),
            perc_state=perc.state,
            lambda2_norm=round(spec.lambda2_normalized, 4),
            spec_state=spec.state,
            energy_norm=round(eng.normalized_energy, 4),
            energy_rate=round(eng.d_mean_energy_dt, 4),
            energy_state=eng.state,
            consensus_var=round(cns.consensus_variance, 4),
            consensus_state=cns.state,
            anomaly_count=anom.anomaly_count,
            health_score=round(health.health_score, 4),
            health_state=health.state,
        )
        h = hash((
            snap.perc_ratio, snap.perc_state,
            snap.lambda2_norm, snap.spec_state,
            snap.energy_norm, snap.energy_rate, snap.energy_state,
            snap.consensus_var, snap.consensus_state,
            snap.anomaly_count,
            snap.health_score, snap.health_state,
        ))

        if h == self._last_hash:
            return self._last_events

        self._last_hash = h
        t = frame.time

        raw: list[SystemEvent] = []

        # ── Rule 1: Network collapse ────────────────────────
        if (snap.perc_ratio < self.PERC_COLLAPSE_THRESHOLD
                and snap.energy_rate < 0):
            sev = min(1.0, (self.PERC_COLLAPSE_THRESHOLD - snap.perc_ratio)
                      / self.PERC_COLLAPSE_THRESHOLD + 0.4)
            raw.append(self._make("collapse", sev, 0.9, t, frame_index))

        # ── Rule 2: Spectral instability ────────────────────
        if (snap.lambda2_norm < self.SPEC_INSTABILITY_THRESHOLD
                and snap.consensus_var > self.CONSENSUS_DIVERGE_THRESHOLD):
            sev = 0.7 + 0.3 * (1.0 - snap.lambda2_norm
                                / max(self.SPEC_INSTABILITY_THRESHOLD, 1e-6))
            raw.append(self._make("instability", sev, 0.85, t, frame_index))

        # ── Rule 3: Energy cascade ──────────────────────────
        if snap.energy_state == "CASCADE":
            sev = 0.8
            raw.append(self._make("cascade", sev, 0.9, t, frame_index))

        # ── Rule 4: Anomaly spike ───────────────────────────
        if snap.anomaly_count >= self.ANOMALY_SPIKE_THRESHOLD:
            sev = min(1.0, 0.5 + snap.anomaly_count * 0.05)
            raw.append(self._make("anomaly_spike", sev, 0.75, t, frame_index))

        # ── Rule 5: Health critical ─────────────────────────
        if snap.health_score < self.HEALTH_CRITICAL_THRESHOLD:
            sev = 1.0 - snap.health_score
            raw.append(self._make("health_critical", sev, 0.95, t, frame_index))

        # ── Rule 6: Recovery ────────────────────────────────
        if (snap.perc_ratio > self.RECOVERY_PERC_THRESHOLD
                and snap.health_state == "HEALTHY"):
            raw.append(self._make("recovery", 0.3, 0.8, t, frame_index))

        # ── Deduplicate & store ─────────────────────────────
        self._last_events = self._deduplicate(raw)
        return self._last_events

    # ── Internal helpers ────────────────────────────────────

    @staticmethod
    def _make(type_: str, severity: float, confidence: float,
              t: float, idx: int) -> SystemEvent:
        return SystemEvent(
            type=type_,
            severity=max(0.0, min(1.0, severity)),
            confidence=max(0.0, min(1.0, confidence)),
            timestamp=t,
            frame_index=idx,
            source="fusion",
        )

    def _deduplicate(self, events: list[SystemEvent]) -> list[SystemEvent]:
        filtered: list[SystemEvent] = []
        for e in events:
            if e.type in self._recent_types:
                continue
            self._recent_types.append(e.type)
            self._event_buffer.append(e)
            filtered.append(e)
        return filtered
