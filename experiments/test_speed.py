import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import SimConfig
from src.simulation import Phase1Simulation

def run_ablation():
    drop_rates = np.linspace(0.1, 0.9, 9)
    results = {"Unconstrained": [], "Proposed": []}
    
    for p_drop in drop_rates:
        print(f"Running p_drop = {p_drop:.1f}...")
        
        # Unconstrained
        cfg_u = SimConfig(num_agents=30, p_drop=float(p_drop), theta_safe_enabled=False, max_time=150.0)
        sim_u = Phase1Simulation(cfg_u)
        sim_u.run()
        l2_u = np.mean([entry["spectral_gap"] for entry in sim_u.connectivity_log])
        results["Unconstrained"].append((p_drop, l2_u))
        
        # Proposed
        cfg_p = SimConfig(num_agents=30, p_drop=float(p_drop), theta_safe_enabled=True, max_time=150.0)
        sim_p = Phase1Simulation(cfg_p)
        sim_p.run()
        l2_p = np.mean([entry["spectral_gap"] for entry in sim_p.connectivity_log])
        results["Proposed"].append((p_drop, l2_p))
        
    print(f"Unconstrained: {results['Unconstrained']}")
    print(f"Proposed: {results['Proposed']}")

if __name__ == "__main__":
    run_ablation()
