"""
spectral_alarm.py — Phase 2AB Spectral Alarm (lambda-2 Collapse Warning).

Standalone early-warning engine that monitors algebraic connectivity and
emits structured alarm states with hysteresis.  Reuses the same
dense/sparse eigenvalue strategy as Phase 2R but applies alarm-specific
thresholds, confidence scoring, and delta tracking.

Hardened patch applied:
  - numerical epsilon stabilisation
  - normalised lambda2 for cross-swarm-size consistency
  - spectral margin metric
  - rate-based confidence score
  - event emission via SystemEvent schema
  - upper-triangle hash optimisation
  - performance guardrail logging

Observer-only.  No mutation of simulation state.  Deterministic.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.analytics.system_event import SystemEvent

_log = logging.getLogger(__name__)


@dataclass
class SpectralAlarmResult:
    lambda2: float              # raw Fiedler value (>= 0, no NaN)
    lambda2_normalized: float   # lambda2 / active_agents
    spectral_margin: float      # lambda2 - ENTER_CRITICAL
    state: str                  # "STABLE" | "WEAKENING" | "CRITICAL"
    delta_lambda2: float        # change since previous frame
    confidence: float           # [0, 1] rate-based confidence
    compute_ms: float           # wall-clock compute time
    events: list[SystemEvent]   # emitted events (may be empty)


class SpectralAlarm:
    """
    Real-time spectral stability monitor.

    Thresholds (with hysteresis):
        STABLE     : lambda2 >= 0.6   (exit only if >= 0.65)
        WEAKENING  : 0.3 <= lambda2   (enter at < 0.6, leave to STABLE >= 0.65)
        CRITICAL   : lambda2 < 0.3    (enter at < 0.3, leave only if > 0.4)
    """

    # ── Class-level thresholds ──────────────────────────────
    ENTER_CRITICAL  = 0.3
    EXIT_CRITICAL   = 0.4
    ENTER_WEAKENING = 0.6
    EXIT_WEAKENING  = 0.65

    # Numerical tolerances
    _EPSILON_EIGVAL = 1e-8
    _EPSILON_DELTA  = 1e-6

    PERF_WARN_MS = 12.0

    def __init__(self) -> None:
        self._state: str = "STABLE"
        self._prev_state: str = "STABLE"
        self._last_hash: int = -1
        self._last_lambda2: float = 0.0
        self._prev_lambda2: float = 0.0
        self._last_result: SpectralAlarmResult | None = None

    # ── Public API ──────────────────────────────────────────

    def reset(self) -> None:
        self._state = "STABLE"
        self._prev_state = "STABLE"
        self._last_hash = -1
        self._last_lambda2 = 0.0
        self._prev_lambda2 = 0.0
        self._last_result = None

    def analyze(
        self,
        adjacency: np.ndarray,
        alive_mask: np.ndarray,
        timestamp: float = 0.0,
        frame_index: int = 0,
    ) -> SpectralAlarmResult:
        """
        Analyse a single frame.

        Parameters
        ----------
        adjacency   : (N, N) uint8 or float adjacency matrix.
        alive_mask  : (N,) bool — True for alive agents.
        timestamp   : simulation time (for event emission).
        frame_index : frame counter (for event emission).
        """
        # ── Hash-skip (upper-triangle only, zero-copy) ──────
        triu = np.triu(adjacency)
        h = hash(triu.data.tobytes() + alive_mask.tobytes())
        if h == self._last_hash and self._last_result is not None:
            return self._last_result
        self._last_hash = h

        # ── Compute lambda2 ─────────────────────────────────
        t0 = _time.perf_counter()
        self._prev_lambda2 = self._last_lambda2
        active_agents = int(alive_mask.sum())

        lam2 = self._compute_lambda2(adjacency, alive_mask)

        # Numerical stabilisation
        if not np.isfinite(lam2):
            lam2 = 0.0
        if abs(lam2) < self._EPSILON_EIGVAL:
            lam2 = 0.0
        lam2 = max(0.0, lam2)

        self._last_lambda2 = lam2
        compute_ms = (_time.perf_counter() - t0) * 1000.0

        if compute_ms > self.PERF_WARN_MS:
            _log.warning("spectral_alarm compute %.1fms > %.0fms guardrail",
                         compute_ms, self.PERF_WARN_MS)

        # ── Delta (epsilon-filtered) ────────────────────────
        delta = lam2 - self._prev_lambda2
        if abs(delta) < self._EPSILON_DELTA:
            delta = 0.0

        # ── Normalisation & margin ──────────────────────────
        lam2_norm = lam2 / max(1, active_agents)
        margin = lam2 - self.ENTER_CRITICAL

        # ── Hysteresis state machine ────────────────────────
        self._prev_state = self._state

        if self._state == "STABLE":
            if lam2 < self.ENTER_CRITICAL:
                self._state = "CRITICAL"
            elif lam2 < self.ENTER_WEAKENING:
                self._state = "WEAKENING"

        elif self._state == "WEAKENING":
            if lam2 < self.ENTER_CRITICAL:
                self._state = "CRITICAL"
            elif lam2 >= self.EXIT_WEAKENING:
                self._state = "STABLE"

        elif self._state == "CRITICAL":
            if lam2 > self.EXIT_CRITICAL:
                self._state = "WEAKENING"
                if lam2 >= self.EXIT_WEAKENING:
                    self._state = "STABLE"

        # ── Confidence (rate-based) ─────────────────────────
        conf = min(1.0, abs(delta) / max(lam2, 1e-6))
        conf = max(0.0, min(1.0, conf))

        # ── Events (only on state transitions) ──────────────
        events: list[SystemEvent] = []
        if self._state != self._prev_state:
            if self._state == "CRITICAL":
                events.append(SystemEvent(
                    type="spectral_collapse_imminent",
                    severity=max(0.0, min(1.0, 1.0 - lam2_norm * 10)),
                    confidence=conf,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    source="spectral",
                ))
            elif self._state == "WEAKENING" and self._prev_state == "STABLE":
                events.append(SystemEvent(
                    type="spectral_instability_detected",
                    severity=max(0.0, min(1.0, 0.5 + (self.ENTER_WEAKENING - lam2))),
                    confidence=conf,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    source="spectral",
                ))
            elif (self._state in ("STABLE", "WEAKENING")
                  and self._prev_state == "CRITICAL"):
                events.append(SystemEvent(
                    type="spectral_recovered",
                    severity=max(0.0, min(1.0, lam2_norm)),
                    confidence=conf,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    source="spectral",
                ))

        result = SpectralAlarmResult(
            lambda2=lam2,
            lambda2_normalized=lam2_norm,
            spectral_margin=margin,
            state=self._state,
            delta_lambda2=delta,
            confidence=conf,
            compute_ms=compute_ms,
            events=events,
        )
        self._last_result = result
        return result

    # ── Eigenvalue computation ──────────────────────────────

    @staticmethod
    def _compute_lambda2(adjacency: np.ndarray,
                         alive_mask: np.ndarray) -> float:
        adj = adjacency[alive_mask][:, alive_mask].astype(np.float64)
        n = adj.shape[0]

        if n <= 1:
            return 0.0

        degree = np.sum(adj, axis=1)

        if np.any(degree == 0):
            return 0.0

        if n > 80:
            D = sp.diags(degree, format="csr")
            A = sp.csr_matrix(adj)
            L = D - A
            try:
                eigvals = spla.eigsh(L, k=2, which="SA",
                                     return_eigenvectors=False)
                # eigvalsh returns sorted ascending
                eigvals = np.sort(eigvals)
                lam2 = float(eigvals[1])
            except Exception:
                L_dense = np.diag(degree) - adj
                eigvals = np.sort(np.linalg.eigvalsh(L_dense))
                lam2 = float(eigvals[1])
        else:
            L_dense = np.diag(degree) - adj
            eigvals = np.sort(np.linalg.eigvalsh(L_dense))
            lam2 = float(eigvals[1])

        return max(0.0, lam2)
