
import csv
import os
from typing import Optional, Dict, List

class KernelLogger:
    """
    Kernel-side logger for system-level validation experiments.
    Writes simulation metrics to CSV files incrementally.
    """
    
    def __init__(self, test_mode: Optional[str] = None, output_dir: str = "logs", filename_suffix: str = ""):
        self.test_mode = test_mode
        self.output_dir = output_dir
        self.filename_suffix = filename_suffix
        self.file_handle = None
        self.writer = None
        self.headers = []
        
        if self.test_mode:
            self._initialize_log()

    def _initialize_log(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        filename = f"experiment_{self.test_mode}{self.filename_suffix}.csv"
        # Prepend experiment index for consistency with requirements
        if self.test_mode == "percolation":
            filename = f"experiment_1_percolation{self.filename_suffix}.csv"
            self.headers = ["time", "true_lambda_2", "avg_local_lambda_proxy", "active_network_edges", "count_regime_stable", "count_regime_fragmented"]
        elif self.test_mode == "thermodynamics":
            filename = f"experiment_2_thermodynamics{self.filename_suffix}.csv"
            self.headers = ["time", "active_drone_count", "total_system_energy", "avg_voronoi_cell_area", "avg_kinematic_velocity"]
        elif self.test_mode == "stability":
            filename = f"experiment_3_stability{self.filename_suffix}.csv"
            self.headers = ["time", "avg_network_delay", "max_velocity_cmd_run_A", "max_velocity_cmd_run_B", "clamped_events_count"]
            
        filepath = os.path.join(self.output_dir, filename)
        self.file_handle = open(filepath, mode='w', newline='')
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.headers)
        self.writer.writeheader()
        self.file_handle.flush()

    def log_snapshot(self, metrics: Dict[str, float]):
        """Write a snapshot of metrics to the CSV file."""
        if self.writer and self.test_mode:
            # Filter metrics to match headers
            row = {k: 0.0 for k in self.headers}
            for k, v in metrics.items():
                if k in self.headers:
                    row[k] = v
            self.writer.writerow(row)
            self.file_handle.flush()

    def close(self):
        """Close the file handle."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            self.writer = None
