import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import SimConfig
from src.simulation import Phase1Simulation

def run():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    config = SimConfig(
        num_agents=200,
        grid_width=150,
        grid_height=150,
        comm_radius=20.0,
        psi_max=0.05,
        test_mode="percolation",
        log_dir=log_dir,
        max_time=500.0
    )
    
    sim = Phase1Simulation(config)
    print("Starting Percolation Experiment (N=200)...")
    sim.run()
    sim.close_loggers()
    print(f"Done. CSV generated in {log_dir}/")

if __name__ == "__main__":
    run()
