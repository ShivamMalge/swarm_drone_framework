"""
anomaly_detector.py — Phase 2X Agent-Level Anomaly Detection Layer.
Contains adaptive thresholds, temporal persistence, and clustering.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
import numpy as np
import scipy.sparse.csgraph as csgraph
import scipy.sparse as sparse

from PySide6.QtCore import QObject, Signal

from src.telemetry.telemetry_frame import TelemetryFrame

@dataclass
class AnomalyMetrics:
    scores: np.ndarray        # (N,) float64 [0, 1]
    severity: np.ndarray      # (N,) float64 [0, 1]
    class_labels: np.ndarray  # (N,) uint8 (0=Normal, 1=Suspicious, 2=Anomaly, 3=Dead)
    cluster_count: int
    largest_cluster: int
    anomaly_count: int

class AnomalyDetector(QObject):
    """
    Vectorized detector isolating independent agent deviances mapping
    orthogonal bounds strictly dynamically avoiding looping natively mapping limits.
    """

    anomaly_detected = Signal(object) # AnomalyMetrics

    def __init__(self) -> None:
        super().__init__()
        
        self._last_hash = -1
        self._last_metrics: AnomalyMetrics | None = None

        self.k_threshold = 3.0
        self.window_size = 10
        self.persistence_req = 3
        
        self.spatial_hist = None
        self.energy_hist = None
        self.consensus_hist = None
        self.degree_hist = None
        self.hist_idx = 0
        
        self.consecutive_anomalies = None
        
        self._ema_scores = None

    def analyze_frame(self, frame: TelemetryFrame) -> AnomalyMetrics:
        N = len(frame.positions)
        if N == 0:
            return AnomalyMetrics(np.array([]), np.array([]), np.array([]), 0, 0, 0)
            
        alive_mask = ~frame.drone_failure_flags
        
        # 6. Hash skip extension (pos, adj, energies)
        h_data = (
            frame.adjacency.data.tobytes() +
            frame.positions.tobytes() +
            frame.energies.tobytes() +
            frame.agent_states.tobytes() +
            alive_mask.tobytes()
        )
        current_hash = hash(h_data)
        
        if current_hash == self._last_hash and self._last_metrics is not None:
            return self._last_metrics
            
        self._last_hash = current_hash
        
        # Allocation handling / Zero copy
        if self.spatial_hist is None or self.spatial_hist.shape[0] != N:
            self.spatial_hist = np.zeros((N, self.window_size), dtype=np.float64)
            self.energy_hist = np.zeros((N, self.window_size), dtype=np.float64)
            self.consensus_hist = np.zeros((N, self.window_size), dtype=np.float64)
            self.degree_hist = np.zeros((N, self.window_size), dtype=np.float64)
            self.consecutive_anomalies = np.zeros(N, dtype=int)
            self.hist_idx = 0
            
        adj = frame.adjacency.astype(np.float64)
        degrees = np.sum(adj, axis=1)
        safe_deg = np.where(degrees > 0, degrees, 1.0)
        adj_norm = adj / safe_deg[:, np.newaxis]
        
        # Raw Metrics computation
        pos = frame.positions
        neighbor_pos = adj_norm @ pos
        neighbor_pos[degrees == 0] = pos[degrees == 0]
        dist = np.sqrt(np.sum((pos - neighbor_pos) ** 2, axis=1))
        
        eng = frame.energies
        neighbor_eng = adj_norm @ eng
        neighbor_eng[degrees == 0] = eng[degrees == 0]
        eng_diff = np.abs(eng - neighbor_eng)
        
        alive_deg = degrees[alive_mask]
        avg_deg = np.mean(alive_deg) if len(alive_deg) > 0 else 0.0
        deg_diff = np.maximum(0.0, avg_deg - degrees)
        
        cns = frame.agent_states
        alive_cns = cns[alive_mask]
        mean_cns = np.mean(alive_cns) if len(alive_cns) > 0 else 0.0
        cns_diff = np.abs(cns - mean_cns)
        
        # Buffer Updates
        idx = self.hist_idx % self.window_size
        self.spatial_hist[:, idx] = dist
        self.energy_hist[:, idx] = eng_diff
        self.degree_hist[:, idx] = deg_diff
        self.consensus_hist[:, idx] = cns_diff
        self.hist_idx += 1
        
        # 1. Adaptive Thresholds
        std_spatial = np.std(self.spatial_hist, axis=1)
        std_energy = np.std(self.energy_hist, axis=1)
        std_degree = np.std(self.degree_hist, axis=1)
        std_cns = np.std(self.consensus_hist, axis=1)
        
        t_sp = np.maximum(self.k_threshold * std_spatial, 1e-4) # strict minimum
        t_en = np.maximum(self.k_threshold * std_energy, 1e-4)
        t_dg = np.maximum(self.k_threshold * std_degree, 1e-4)
        t_cn = np.maximum(self.k_threshold * std_cns, 1e-4)
        
        # Normalization
        score_spatial = np.clip(dist / t_sp, 0.0, 1.0)
        score_energy = np.clip(eng_diff / t_en, 0.0, 1.0)
        score_degree = np.clip(deg_diff / t_dg, 0.0, 1.0)
        score_consensus = np.clip(cns_diff / t_cn, 0.0, 1.0)
        
        # Combine
        raw_scores = (0.3 * score_spatial +
                      0.3 * score_consensus +
                      0.2 * score_energy +
                      0.2 * score_degree)
                      
        raw_scores[~alive_mask] = 0.0
        
        if self._ema_scores is None or len(self._ema_scores) != N:
            self._ema_scores = raw_scores.copy()
        else:
            self._ema_scores = 0.7 * self._ema_scores + 0.3 * raw_scores
            
        scores = self._ema_scores
        
        # Classification & 2. Persistence Filter
        class_labels = np.zeros(N, dtype=np.uint8)
        class_labels[scores > 0.3] = 1 # Suspicious
        
        is_anom = scores > 0.6
        self.consecutive_anomalies[is_anom] += 1
        self.consecutive_anomalies[~is_anom] = 0
        
        valid_anom = self.consecutive_anomalies >= self.persistence_req
        class_labels[valid_anom] = 2
        
        class_labels[~alive_mask] = 3
        
        # 3. Clustered Anomaly Detection
        cluster_count = 0
        largest_cluster = 0
        anomaly_mask = valid_anom & alive_mask
        
        if np.any(anomaly_mask):
            anom_adj = adj[anomaly_mask][:, anomaly_mask]
            sparse_adj = sparse.csr_matrix(anom_adj)
            n_comp, labels = csgraph.connected_components(sparse_adj, directed=False)
            cluster_count = n_comp
            if n_comp > 0:
                _, counts = np.unique(labels, return_counts=True)
                largest_cluster = int(np.max(counts))
                
        anomaly_count = int(np.sum(valid_anom))
        
        severity = scores.copy()
        severity[class_labels == 2] += 0.2
        severity = np.clip(severity, 0.0, 1.0)
        
        metrics = AnomalyMetrics(
            scores=scores,
            severity=severity,
            class_labels=class_labels,
            cluster_count=cluster_count,
            largest_cluster=largest_cluster,
            anomaly_count=anomaly_count
        )
        
        self._last_metrics = metrics
        if anomaly_count > 0 or np.any(class_labels == 1):
            self.anomaly_detected.emit(metrics)
            
        return metrics
