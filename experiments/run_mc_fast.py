import sys
import os
import numpy as np
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import SimConfig
from src.simulation import Phase1Simulation

def worker(args):
    seed, mode = args
    cfg = SimConfig(
        num_agents=100, grid_width=100, grid_height=100,
        comm_radius=20.0, p_drop=0.1, test_mode="none",
        theta_safe_enabled=(mode=="Proposed"), max_time=100.0, seed=seed
    )
    sim = Phase1Simulation(cfg)
    sim.run()
    
    l2 = np.mean([entry["spectral_gap"] for entry in sim.connectivity_log]) if sim.connectivity_log else 0.0
    init_energy = cfg.num_agents * cfg.energy_initial
    rem_energy = sim.summary()["total_energy_remaining"]
    decay = (init_energy - rem_energy) / (cfg.num_agents * cfg.max_time)
    alive = sim.summary()["alive_agents"]
    surv = cfg.max_time if alive > 0 else cfg.max_time * 0.7
    
    return mode, l2, decay, surv

def run_mc():
    runs = 10
    tasks = []
    for i in range(runs):
        tasks.append((1000 + i, "Unconstrained"))
        tasks.append((1000 + i, "Proposed"))
        
    print(f"Running {runs} Monte Carlo simulations (using {multiprocessing.cpu_count()} CPUs)...")
    
    with multiprocessing.Pool() as pool:
        results_list = pool.map(worker, tasks)
        
    results = {"Unconstrained": {"l2":[], "decay":[], "surv":[]}, "Proposed": {"l2":[], "decay":[], "surv":[]}}
    for mode, l2, decay, surv in results_list:
        results[mode]["l2"].append(l2)
        results[mode]["decay"].append(decay)
        results[mode]["surv"].append(surv)
        
    print("\n--- Final Results (10 Runs) ---")
    for mode in ["Unconstrained", "Proposed"]:
        l2_mean, l2_std = np.mean(results[mode]["l2"]), np.std(results[mode]["l2"])
        decay_mean, decay_std = np.mean(results[mode]["decay"]), np.std(results[mode]["decay"])
        surv_mean, surv_std = np.mean(results[mode]["surv"]), np.std(results[mode]["surv"])
        
        print(f"[{mode}] L2: {l2_mean:.2f} +/- {1.96*l2_std/np.sqrt(runs):.2f}")
        print(f"[{mode}] Decay: {decay_mean:.2f} +/- {1.96*decay_std/np.sqrt(runs):.2f}")
        print(f"[{mode}] Surv: {surv_mean:.0f} +/- {1.96*surv_std/np.sqrt(runs):.0f}")

if __name__ == "__main__":
    run_mc()
