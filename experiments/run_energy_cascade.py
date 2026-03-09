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
        num_agents=50,
        grid_width=100,
        grid_height=100,
        test_mode="thermodynamics",
        log_dir=log_dir,
        max_time=300.0
    )
    
    sim = Phase1Simulation(config)
    print("Starting Thermodynamics Experiment...")
    sim.run()
    sim.close_loggers()
    print(f"Done. CSV generated in {log_dir}/")

if __name__ == "__main__":
    run()
