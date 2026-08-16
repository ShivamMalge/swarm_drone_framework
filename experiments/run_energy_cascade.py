import sys
import os
import csv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import SimConfig
from src.simulation import Phase1Simulation

def run():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    print("Starting Thermodynamics Experiment...")
    
    # Run A: Baseline (Unconstrained, no theta safe)
    config_a = SimConfig(
        num_agents=50,
        grid_width=100,
        grid_height=100,
        test_mode="thermodynamics",
        theta_safe_enabled=False,
        coverage_enabled=True,
        log_dir=log_dir,
        max_time=300.0,
        seed=42
    )
    sim_a = Phase1Simulation(config_a, log_suffix="_run_A")
    print("  -> Running Baseline (Theta Safe OFF)...")
    sim_a.run()
    sim_a.close_loggers()

    # Run B: Proposed Framework
    config_b = SimConfig(
        num_agents=50,
        grid_width=100,
        grid_height=100,
        test_mode="thermodynamics",
        theta_safe_enabled=True,
        coverage_enabled=True,
        log_dir=log_dir,
        max_time=300.0,
        seed=42
    )
    sim_b = Phase1Simulation(config_b, log_suffix="_run_B")
    print("  -> Running Proposed (Theta Safe ON)...")
    sim_b.run()
    sim_b.close_loggers()
    
    # Merge CSVs
    file_a = os.path.join(log_dir, "experiment_2_thermodynamics_run_A.csv")
    file_b = os.path.join(log_dir, "experiment_2_thermodynamics_run_B.csv")
    output_file = os.path.join(log_dir, "experiment_2_thermodynamics_merged.csv")
    
    if os.path.exists(file_a) and os.path.exists(file_b):
        data_a = {}
        with open(file_a, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_a[row['time']] = row
                
        with open(file_b, 'r', newline='') as f_in, open(output_file, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            fieldnames = [
                "time", 
                "baseline_active_drones", "framework_active_drones",
                "baseline_total_energy", "framework_total_energy"
            ]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            
            for row_b in reader:
                t = row_b['time']
                row_a = data_a.get(t, {})
                
                merged_row = {
                    "time": t,
                    "baseline_active_drones": row_a.get("active_drone_count", 0.0),
                    "framework_active_drones": row_b.get("active_drone_count", 0.0),
                    "baseline_total_energy": row_a.get("total_system_energy", 0.0),
                    "framework_total_energy": row_b.get("total_system_energy", 0.0)
                }
                writer.writerow(merged_row)
        
        print(f"Final merged CSV generated: {output_file}")

if __name__ == "__main__":
    run()
