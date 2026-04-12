"""
spectral_analyzer.py — Phase 2R Spectral Stability Analyzer.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from PySide6.QtCore import QObject, Signal

from src.telemetry.telemetry_frame import TelemetryFrame


@dataclass
class SpectralMetrics:
    lambda2: float
    lambda2_normalized: float
    spectral_margin: float
    dl2_dt: float
    state: str


class SpectralAnalyzer(QObject):
    """
    Computes and tracks the algebraic connectivity (Fiedler value, λ₂) 
    of the dynamic swarm communication graph.
    """

    spectral_instability_detected = Signal(float)  # timestamp
    spectral_recovered = Signal(float)             # timestamp

    def __init__(
        self,
        collapse_threshold: float = 0.1,
        recovery_threshold: float = 0.15,
        history_size: int = 300,
    ) -> None:
        super().__init__()
        self._collapse_threshold = collapse_threshold
        self._recovery_threshold = recovery_threshold
        self._history_size = history_size

        self._in_collapse_state = False
        self._last_adj_hash: int = -1
        self._last_lambda2: float = 0.0
        self._last_time: float = -1.0

        self.time_history = collections.deque(maxlen=history_size)
        self.lambda2_history = collections.deque(maxlen=history_size)

    def analyze_frame(self, frame: TelemetryFrame) -> SpectralMetrics:
        """Calculate algebraic connectivity and maintain tracking history."""
        # Detect topology changes using hash mapping
        adj_hash = hash(frame.adjacency.data.tobytes())
        alive_mask = ~frame.drone_failure_flags
        
        # Check cache
        if adj_hash == self._last_adj_hash and self._last_time >= 0:
            lambda2 = self._last_lambda2
        else:
            self._last_adj_hash = adj_hash
            lambda2 = self._compute_lambda2(frame.adjacency, alive_mask)
            self._last_lambda2 = lambda2

        # Rate of change logic
        dt = frame.time - self._last_time if self._last_time >= 0 else 1.0
        if dt <= 0:
            dt = 1e-6
            
        if len(self.lambda2_history) > 0:
            dl2_dt = (lambda2 - self.lambda2_history[-1]) / dt
        else:
            dl2_dt = 0.0

        self._last_time = frame.time

        # Append to history
        self.time_history.append(frame.time)
        self.lambda2_history.append(lambda2)

        # Hysteresis and Collapse Signals
        if not self._in_collapse_state and lambda2 <= self._collapse_threshold:
            self._in_collapse_state = True
            self.spectral_instability_detected.emit(frame.time)
        elif self._in_collapse_state and lambda2 >= self._recovery_threshold:
            self._in_collapse_state = False
            self.spectral_recovered.emit(frame.time)

        if lambda2 > 0.5:
            state = "STRONG"
        elif lambda2 > 0.1:
            state = "WEAKENING"
        else:
            state = "CRITICAL"

        alive_count = int(alive_mask.sum())
        lambda2_norm = lambda2 / max(1, alive_count)

        return SpectralMetrics(
            lambda2=lambda2,
            lambda2_normalized=lambda2_norm,
            spectral_margin=lambda2 - self._collapse_threshold,
            dl2_dt=dl2_dt,
            state=state,
        )

    def _compute_lambda2(self, adjacency: np.ndarray, alive_mask: np.ndarray) -> float:
        """
        Compute the second smallest eigenvalue (Fiedler value) of the graph Laplacian.
        """
        # Exclude failed agents and explicitly cast to float to prevent uint8 underflow
        adj = adjacency[alive_mask][:, alive_mask].astype(np.float64)
        n = adj.shape[0]

        if n <= 1:
            return 0.0

        degree = np.sum(adj, axis=1)
        
        # If any node has degree 0, graph is disconnected immediately -> λ₂ = 0
        if np.any(degree == 0):
            return 0.0

        # For N > 80, attempt sparse formulation (optimized)
        if n > 80:
            # Create sparse Laplacian
            D = sp.diags(degree, format="csr")
            A = sp.csr_matrix(adj)
            L = D - A
            
            try:
                # We want the 2 smallest algebraic eigenvalues: k=2.
                # which='SA' means Smallest Algebraic. 
                # Note: 'SA' can occasionally struggle if exact zero. 
                # A small shift could help, but spla handles it ok generally.
                # Shift-invert is 'SM' with sigma=0, but 'SA' is often sufficient for L.
                eigvals = spla.eigsh(L, k=2, which='SA', return_eigenvectors=False)
                # eigvals are returned in ascending order
                lambda2 = eigvals[1]
            except Exception:
                # Fallback to dense if sparse solver fails (e.g., convergence issues)
                L_dense = np.diag(degree) - adj
                eigvals = np.linalg.eigvalsh(L_dense)
                lambda2 = eigvals[1]
        else:
            L_dense = np.diag(degree) - adj
            eigvals = np.linalg.eigvalsh(L_dense)
            lambda2 = eigvals[1]

        # Numerical stability: clamp small noisy negatives near zero
        return float(max(0.0, lambda2))
