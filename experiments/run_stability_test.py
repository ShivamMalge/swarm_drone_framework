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
    
    # Run A: Theta Safe Disabled
    config_a = SimConfig(
        num_agents=30,
        latency_mean=2.0,
        latency_min=1.0,
        test_mode="stability",
        theta_safe_enabled=False,
        log_dir=log_dir,
        max_time=200.0,
        seed=123 # Fixed seed for comparison
    )
    sim_a = Phase1Simulation(config_a, log_suffix="_run_A")
    print("Starting Stability Test - Run A (Theta Safe OFF)...")
    sim_a.run()
    sim_a.close_loggers()

    # Run B: Theta Safe Enabled
    config_b = SimConfig(
        num_agents=30,
        latency_mean=2.0,
        latency_min=1.0,
        test_mode="stability",
        theta_safe_enabled=True,
        log_dir=log_dir,
        max_time=200.0,
        seed=123 # Identical seed
    )
    sim_b = Phase1Simulation(config_b, log_suffix="_run_B")
    print("Starting Stability Test - Run B (Theta Safe ON)...")
    sim_b.run()
    sim_b.close_loggers()
    
    # Merge CSVs
    file_a = os.path.join(log_dir, "experiment_3_stability_run_A.csv")
    file_b = os.path.join(log_dir, "experiment_3_stability_run_B.csv")
    output_file = os.path.join(log_dir, "experiment_3_stability.csv")
    
    if os.path.exists(file_a) and os.path.exists(file_b):
        data_a = {}
        with open(file_a, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_a[row['time']] = row
                
        with open(file_b, 'r', newline='') as f_in, open(output_file, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            fieldnames = ["time", "avg_network_delay", "max_velocity_cmd_run_A", "max_velocity_cmd_run_B", "clamped_events_count"]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            
            for row_b in reader:
                t = row_b['time']
                row_a = data_a.get(t, {})
                
                merged_row = {
                    "time": t,
                    "avg_network_delay": row_b['avg_network_delay'],
                    "max_velocity_cmd_run_A": row_a.get('max_velocity_cmd_run_A', 0.0),
                    "max_velocity_cmd_run_B": row_b.get('max_velocity_cmd_run_B', 0.0),
                    "clamped_events_count": row_b['clamped_events_count']
                }
                writer.writerow(merged_row)
        
        print(f"Final merged CSV generated: {output_file}")

if __name__ == "__main__":
    run()
