import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from src.telemetry.telemetry_frame import TelemetryFrame
from src.analytics.percolation_analyzer import PercolationMetrics
from src.analytics.anomaly_detector import AnomalyMetrics
from src.analytics.system_event import SystemEvent
from src.analytics.percolation_shock_analyzer import ShockMetrics
from src.gui.percolation_shock_overlay import shock_brushes
from src.analytics.energy_heatmap_mapper import EnergyHeatmapResult
from src.gui.energy_heatmap_panel import EnergyHeatmapOverlay

class NetworkViewer(QWidget):
    """
    Renders the drone swarm communication topology.
    - Nodes styled by connectivity / degree.
    - Edges drawn via high-performance interleaved NaNs.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

        self._last_adj_hash: int = -1
        
        # Color palettes
        self._color_dead = pg.mkBrush(100, 100, 100, 150)
        self._color_isolated = pg.mkBrush(255, 100, 100, 200) # Red
        # Component colors
        self._comp_colors = [
            pg.mkBrush(88, 166, 255, 200),  # Blue
            pg.mkBrush(63, 185, 80, 200),   # Green
            pg.mkBrush(210, 153, 34, 200),  # Orange
            pg.mkBrush(163, 113, 247, 200), # Purple
            pg.mkBrush(248, 81, 73, 200),   # Red-ish
        ]

        # Cached structures
        self._edge_x: np.ndarray = np.array([])
        self._edge_y: np.ndarray = np.array([])
        self._cached_brushes = None
        self._cached_sizes = None

        # Event integration states
        self._active_events: list[SystemEvent] = []
        self._shock: ShockMetrics | None = None
        self._energy_heatmap: EnergyHeatmapResult | None = None
        self._heatmap_overlay = EnergyHeatmapOverlay()
        self._pulse_edges = False
        self._dim_nodes = False

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#0d1117')
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        
        # Scatter and edges
        self.edge_item = pg.PlotDataItem(pen=pg.mkPen(color=(139, 148, 158, 100), width=1))
        self.node_item = pg.ScatterPlotItem(pxMode=True)
        
        self.plot.addItem(self.edge_item)
        self.plot.addItem(self.node_item)
        
        layout.addWidget(self.plot)

    def set_fused_events(self, events: list[SystemEvent]) -> None:
        """Accept fused events and derive visual overrides."""
        self._active_events = events
        types = {e.type for e in events}
        self._pulse_edges = "instability" in types or "collapse" in types
        self._dim_nodes = "cascade" in types or "health_critical" in types

    def update_shock(self, shock: ShockMetrics | None) -> None:
        """Store latest shock metrics for overlay during next render."""
        self._shock = shock

    def update_energy_heatmap(self, heatmap: EnergyHeatmapResult | None) -> None:
        """Store latest energy heatmap metrics for overlay."""
        self._energy_heatmap = heatmap

    def trigger_spectral_instability(self):
        self._pulse_edges = True

    def clear_spectral_instability(self):
        self._pulse_edges = False

    def trigger_energy_cascade(self):
        self._dim_nodes = True

    def clear_energy_cascade(self):
        self._dim_nodes = False

    def update_frame(self, frame: TelemetryFrame, perc_metrics: PercolationMetrics | None = None, anom_metrics: AnomalyMetrics | None = None) -> None:
        N = len(frame.positions)
        if N == 0:
            return

        alive_mask = ~frame.drone_failure_flags
        adj = frame.adjacency.astype(np.float64)
        adj_hash = hash(adj.tobytes() + alive_mask.tobytes())

        if adj_hash != self._last_adj_hash:
            self._last_adj_hash = adj_hash
            
            # Recalculate topology visuals
            # Degrees
            degrees = np.sum(adj, axis=1)
            
            # Edges
            # Exclude dead agents from edges
            alive_adj = adj.copy()
            alive_adj[frame.drone_failure_flags, :] = 0
            alive_adj[:, frame.drone_failure_flags] = 0
            
            row, col = np.triu(alive_adj, 1).nonzero()
            
            # We construct normalized index arrays, coordinates dynamically filled later
            self._edge_pairs = (row, col)

            # Node Coloring
            brushes = np.empty(N, dtype=object)
            sizes = np.empty(N, dtype=np.float64)
            
            # By default base (Stable = Green)
            brushes.fill(self._comp_colors[1])
            sizes.fill(8.0)
            
            if perc_metrics is not None and frame.connected_components:
                comps = frame.connected_components
                if comps:
                    # LCC is typically the largest list
                    sizes_list = [len(c) for c in comps]
                    lcc_idx = np.argmax(sizes_list) if sizes_list else 0
                    
                    for i, comp in enumerate(comps):
                        color_idx = 1 if i == lcc_idx else ((i % (len(self._comp_colors)-1)) + 1)
                        brush = self._comp_colors[color_idx]
                        for node in comp:
                            if alive_mask[node]:
                                brushes[node] = brush
            
            # Degree mappings
            sizes = 6.0 + (degrees * 1.5)
            # Cap sizes globally for improved visual clarity
            sizes = np.clip(sizes, 8.0, 26.0)

            # Isolate / Dead
            isolated = (degrees == 0) & alive_mask
            brushes[isolated] = self._color_isolated
            brushes[frame.drone_failure_flags] = self._color_dead
            
            self._cached_brushes = brushes
            self._cached_sizes = sizes

        # Prepare node positions
        pos_x = frame.positions[:, 0]
        pos_y = frame.positions[:, 1]
        
        # Build edge paths based on latest positions
        row, col = self._edge_pairs
        E = len(row)
        if E > 0:
            x_flat = np.empty(E * 3, dtype=np.float64)
            y_flat = np.empty(E * 3, dtype=np.float64)
            
            x_flat[0::3] = pos_x[row]
            x_flat[1::3] = pos_x[col]
            x_flat[2::3] = np.nan
            
            y_flat[0::3] = pos_y[row]
            y_flat[1::3] = pos_y[col]
            y_flat[2::3] = np.nan
            
            if self._pulse_edges:
                self.edge_item.setPen(pg.mkPen(color=(248, 81, 73, 180), width=2, style=Qt.PenStyle.DashLine))
            else:
                self.edge_item.setPen(pg.mkPen(color=(139, 148, 158, 80), width=1))
                
            self.edge_item.setData(x=x_flat, y=y_flat)
        else:
            self.edge_item.setData(x=np.array([]), y=np.array([]))

        # Node properties mutation (events)
        N = len(pos_x)
        colors = np.zeros((N, 4), dtype=np.ubyte)
        final_sizes = self._cached_sizes.copy() if self._cached_sizes is not None else np.zeros(N)
        
        # Populate base colors
        if self._cached_brushes is not None:
            for i, b in enumerate(self._cached_brushes):
                c = b.color()
                colors[i] = (c.red(), c.green(), c.blue(), c.alpha())
                
        if self._dim_nodes:
            colors[:, 3] = 50

        # Apply Heatmap overlay
        if self._energy_heatmap is not None:
            alive = ~frame.drone_failure_flags
            colors, final_sizes = self._heatmap_overlay.update_overlay(
                self._energy_heatmap, final_sizes, alive
            )

        # Apply anomaly overlay
        if anom_metrics is not None and len(anom_metrics.class_labels) == N:
            anom_idx = np.where(anom_metrics.class_labels == 2)[0]
            susp_idx = np.where(anom_metrics.class_labels == 1)[0]
            
            colors[susp_idx] = (210, 153, 34, 255)
            colors[anom_idx] = (248, 81, 73, 255)

        # Apply shock overlay (highest visual priority when active)
        if (self._shock is not None
                and self._shock.shock_active
                and len(self._shock.shock_normalized) == N):
            alive = ~frame.drone_failure_flags
            s_brushes = shock_brushes(self._shock.shock_normalized, alive)
            for i, b in enumerate(s_brushes):
                c = b.color()
                colors[i] = (c.red(), c.green(), c.blue(), c.alpha())
        
        self.node_item.setData(
            x=pos_x,
            y=pos_y,
            brush=colors,
            size=final_sizes,
            pen=pg.mkPen(None)
        )
