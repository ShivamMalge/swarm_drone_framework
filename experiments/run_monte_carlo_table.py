import sys
import os

# Pin BLAS to a single thread BEFORE numpy is imported. Each worker process is
# already a full core's worth of work; letting every one of them spawn its own
# BLAS thread pool oversubscribes the 8 physical cores and costs ~10% wall clock
# on the suite (measured: 68.1s -> 61.1s on 16 tasks). Workers re-import this
# module under spawn, so setting it here reaches them too.
for _blas_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import numpy as np
import concurrent.futures
from dataclasses import dataclass

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import SimConfig
from src.simulation import Phase1Simulation

def run_single_sim(args):
    seed, mode_name, config_kwargs = args
    config = SimConfig(seed=seed, **config_kwargs)
    sim = Phase1Simulation(config)
    sim.run()
    sim.close_loggers()

    summary = sim.summary()
    
    # Process connectivity
    l2 = np.mean([entry["spectral_gap"] for entry in sim.connectivity_log]) if sim.connectivity_log else 0.0
    
    # Process energy decay
    init_energy = config.num_agents * config.energy_initial
    rem_energy = summary["total_energy_remaining"]
    tod = summary.get("time_of_death")
    
    censored = False
    if tod is None:
        surv = config.max_time
        censored = True
    else:
        surv = tod
        
    t50 = summary.get("time_to_50pct_attrition")
    if t50 is None:
        t50 = config.max_time
        
    decay = (init_energy - rem_energy) / (config.num_agents * surv) if surv > 0 else 0
    
    cov_rate = summary.get("coverage_completion_rate", 0.0)
    
    # Check hibernation numbers
    # We log these in the kernel logger, but we can also just extract them from sim.kernel_logger.history if available.
    # Actually, the agent stats are in the adaptation_log or kernel_logger.
    # We will just print them if needed, but cov_rate is good enough.
    
    return mode_name, {
        "lambda_2": l2,
        "energy_decay": decay,
        "survival": surv,
        "time_to_50pct_attrition": t50,
        "coverage_completion_rate": cov_rate,
        "censored": censored
    }

def run_mc(runs=50):
    print(f"Running {runs} Monte Carlo simulations across 4 conditions using Multiprocessing...")
    
    modes = ["Unconstrained", "Static-Epsilon Baseline", "Proposed", "True Oracle"]
    
    results = {
        mode: {
            "lambda_2": [], 
            "energy_decay": [], 
            "survival": [], 
            "time_to_50pct_attrition": [],
            "coverage_completion_rate": [],
            "censored_count": 0
        } for mode in modes
    }
    
    tasks = []
    for i in range(runs):
        seed = 1000 + i
        
        # 1. Unconstrained
        tasks.append((seed, "Unconstrained", {
            "num_agents": 100, "grid_width": 100, "grid_height": 100, 
            "comm_radius": 20.0, "p_drop": 0.1, "test_mode": "none", 
            "theta_safe_enabled": False, "global_info_enabled": False,
            "coverage_enabled": True, "log_dir": "logs", "max_time": 2000.0
        }))
        
        # 2. Static-Epsilon Baseline (formerly Static Bounded)
        tasks.append((seed, "Static-Epsilon Baseline", {
            "num_agents": 100, "grid_width": 100, "grid_height": 100, 
            "comm_radius": 20.0, "p_drop": 0.1, "test_mode": "static_bounded", 
            "theta_safe_enabled": True, "global_info_enabled": False,
            "coverage_enabled": True, "log_dir": "logs", "max_time": 2000.0
        }))
        
        # 3. Proposed
        tasks.append((seed, "Proposed", {
            "num_agents": 100, "grid_width": 100, "grid_height": 100, 
            "comm_radius": 20.0, "p_drop": 0.1, "test_mode": "none", 
            "theta_safe_enabled": True, "global_info_enabled": False,
            "coverage_enabled": True, "log_dir": "logs", "max_time": 2000.0
        }))
        
        # 4. True Oracle
        tasks.append((seed, "True Oracle", {
            "num_agents": 100, "grid_width": 100, "grid_height": 100, 
            "comm_radius": 20.0, "p_drop": 0.1, "test_mode": "none", 
            "theta_safe_enabled": True, "global_info_enabled": True,
            "coverage_enabled": True, "log_dir": "logs", "max_time": 2000.0
        }))

    completed = 0
    total = len(tasks)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for mode_name, res in executor.map(run_single_sim, tasks):
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{total} tasks complete.")
                
            results[mode_name]["lambda_2"].append(res["lambda_2"])
            results[mode_name]["energy_decay"].append(res["energy_decay"])
            results[mode_name]["survival"].append(res["survival"])
            results[mode_name]["time_to_50pct_attrition"].append(res["time_to_50pct_attrition"])
            results[mode_name]["coverage_completion_rate"].append(res["coverage_completion_rate"])
            if res["censored"]:
                results[mode_name]["censored_count"] += 1

    print("\n--- Final Results (50 Runs) ---")
    for mode in modes:
        l2_mean = np.mean(results[mode]["lambda_2"])
        l2_std = np.std(results[mode]["lambda_2"])
        
        ed_mean = np.mean(results[mode]["energy_decay"])
        ed_std = np.std(results[mode]["energy_decay"])
        
        surv_mean = np.mean(results[mode]["survival"])
        surv_std = np.std(results[mode]["survival"])
        
        t50_mean = np.mean(results[mode]["time_to_50pct_attrition"])
        t50_std = np.std(results[mode]["time_to_50pct_attrition"])
        
        cov_mean = np.mean(results[mode]["coverage_completion_rate"])
        cov_std = np.std(results[mode]["coverage_completion_rate"])
        
        censored = results[mode]["censored_count"]
        
        print(f"\n[{mode}]")
        print(f"  Spectral Connectivity (lambda_2): {l2_mean:.2f} +/- {1.96 * l2_std / np.sqrt(runs):.2f} (std: {l2_std:.4f})")
        print(f"  Energy Decay Rate: {ed_mean:.2f} +/- {1.96 * ed_std / np.sqrt(runs):.2f} (std: {ed_std:.4f})")
        print(f"  Mean Swarm Survival (Total Death): {surv_mean:.0f} +/- {1.96 * surv_std / np.sqrt(runs):.0f} (std: {surv_std:.2f})")
        print(f"  Mean Time to 50% Attrition: {t50_mean:.0f} +/- {1.96 * t50_std / np.sqrt(runs):.0f} (std: {t50_std:.2f})")
        print(f"  Coverage Completion Rate: {cov_mean:.2%}")
        print(f"  Censored Runs (Hit 2000s max_time cap): {censored}/{runs}")
        
        if surv_std < (0.01 * surv_mean):
            print("  *** WARNING: Suspiciously low variance on Survival. Likely right-censored artifact. ***")

if __name__ == "__main__":
    run_mc(50)
