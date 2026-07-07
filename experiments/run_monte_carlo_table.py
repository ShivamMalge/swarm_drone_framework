import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import SimConfig
from src.simulation import Phase1Simulation

def run_mc(runs=50):
    print(f"Running {runs} Monte Carlo simulations...")
    
    results = {
        "Unconstrained": {"lambda_2": [], "energy_decay": [], "survival": []},
        "Static Bounded": {"lambda_2": [], "energy_decay": [], "survival": []},
        "Proposed": {"lambda_2": [], "energy_decay": [], "survival": []}
    }
    
    for i in range(runs):
        seed = 1000 + i
        print(f"Run {i+1}/{runs} - Seed {seed}")
        
        # Unconstrained RGG (theta_safe_enabled = False)
        config_u = SimConfig(
            num_agents=100,
            grid_width=100,
            grid_height=100,
            comm_radius=20.0,
            p_drop=0.1,
            test_mode="none",
            theta_safe_enabled=False,
            log_dir="logs",
            max_time=2000.0,
            seed=seed
        )
        sim_u = Phase1Simulation(config_u)
        sim_u.run()
        
        l2_u = np.mean([entry["spectral_gap"] for entry in sim_u.connectivity_log])
        init_energy_u = config_u.num_agents * config_u.energy_initial
        rem_energy_u = sim_u.summary()["total_energy_remaining"]
        tod_u = sim_u.summary().get("time_of_death")
        surv_u = tod_u if tod_u is not None else config_u.max_time
        decay_u = (init_energy_u - rem_energy_u) / (config_u.num_agents * surv_u) if surv_u > 0 else 0
        
        results["Unconstrained"]["lambda_2"].append(l2_u)
        results["Unconstrained"]["energy_decay"].append(decay_u)
        results["Unconstrained"]["survival"].append(surv_u)
        
        # Proposed (theta_safe_enabled = True)
        config_p = SimConfig(
            num_agents=100,
            grid_width=100,
            grid_height=100,
            comm_radius=20.0,
            p_drop=0.1,
            test_mode="none",
            theta_safe_enabled=True,
            log_dir="logs",
            max_time=2000.0,
            seed=seed
        )
        sim_p = Phase1Simulation(config_p)
        sim_p.run()
        
        l2_p = np.mean([entry["spectral_gap"] for entry in sim_p.connectivity_log])
        init_energy_p = config_p.num_agents * config_p.energy_initial
        rem_energy_p = sim_p.summary()["total_energy_remaining"]
        tod_p = sim_p.summary().get("time_of_death")
        surv_p = tod_p if tod_p is not None else config_p.max_time
        decay_p = (init_energy_p - rem_energy_p) / (config_p.num_agents * surv_p) if surv_p > 0 else 0
        
        results["Proposed"]["lambda_2"].append(l2_p)
        results["Proposed"]["energy_decay"].append(decay_p)
        results["Proposed"]["survival"].append(surv_p)
        
        # Static Bounded RGG (theta_safe_enabled = True, test_mode = "static_bounded")
        config_s = SimConfig(
            num_agents=100,
            grid_width=100,
            grid_height=100,
            comm_radius=20.0,
            p_drop=0.1,
            test_mode="static_bounded",
            theta_safe_enabled=True,
            log_dir="logs",
            max_time=2000.0,
            seed=seed
        )
        sim_s = Phase1Simulation(config_s)
        sim_s.run()
        
        l2_s = np.mean([entry["spectral_gap"] for entry in sim_s.connectivity_log])
        init_energy_s = config_s.num_agents * config_s.energy_initial
        rem_energy_s = sim_s.summary()["total_energy_remaining"]
        tod_s = sim_s.summary().get("time_of_death")
        surv_s = tod_s if tod_s is not None else config_s.max_time
        decay_s = (init_energy_s - rem_energy_s) / (config_s.num_agents * surv_s) if surv_s > 0 else 0
        
        results["Static Bounded"]["lambda_2"].append(l2_s)
        results["Static Bounded"]["energy_decay"].append(decay_s)
        results["Static Bounded"]["survival"].append(surv_s)
        
    print("\n--- Final Results (50 Runs) ---")
    
    # Calculate means and 95% CIs
    for mode in ["Unconstrained", "Static Bounded", "Proposed"]:
        l2_mean = np.mean(results[mode]["lambda_2"])
        l2_std = np.std(results[mode]["lambda_2"])
        l2_ci = 1.96 * l2_std / np.sqrt(runs)
        
        ed_mean = np.mean(results[mode]["energy_decay"])
        ed_std = np.std(results[mode]["energy_decay"])
        ed_ci = 1.96 * ed_std / np.sqrt(runs)
        
        surv_mean = np.mean(results[mode]["survival"])
        surv_std = np.std(results[mode]["survival"])
        surv_ci = 1.96 * surv_std / np.sqrt(runs)
        
        print(f"[{mode}] Spectral Connectivity (lambda_2): {l2_mean:.2f} +/- {l2_ci:.2f}")
        print(f"[{mode}] Energy Decay Rate: {ed_mean:.2f} +/- {ed_ci:.2f}")
        print(f"[{mode}] Mean Swarm Survival: {surv_mean:.0f} +/- {surv_ci:.0f}")

if __name__ == "__main__":
    run_mc(50)
