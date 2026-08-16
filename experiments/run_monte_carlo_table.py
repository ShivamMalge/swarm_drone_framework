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
    
    # The former "Static-Epsilon Baseline" arm (test_mode="static_bounded") was
    # removed: it produced bit-identical positions and energies to "Proposed" on
    # every seed tested (audit F-02). That flag reaches exactly one line
    # (agent_core.py:522), suppressing the dynamic gossip_epsilon bound -- and
    # gossip_epsilon has no causal path to any physical outcome (audit F-10),
    # so the two arms were the same simulation reported twice.
    modes = ["Unconstrained", "Proposed", "True Oracle"]
    
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
        
        # 2. Proposed
        tasks.append((seed, "Proposed", {
            "num_agents": 100, "grid_width": 100, "grid_height": 100, 
            "comm_radius": 20.0, "p_drop": 0.1, "test_mode": "none", 
            "theta_safe_enabled": True, "global_info_enabled": False,
            "coverage_enabled": True, "log_dir": "logs", "max_time": 2000.0
        }))
        
        # 3. True Oracle. Billed all-to-all: global awareness implies global
        # bandwidth, and this study is about coordination under JOINT
        # communication and energy constraints. The per-neighbour accounting
        # is reported separately by run_oracle_sensitivity.py.
        tasks.append((seed, "True Oracle", {
            "num_agents": 100, "grid_width": 100, "grid_height": 100,
            "comm_radius": 20.0, "p_drop": 0.1, "test_mode": "none",
            "theta_safe_enabled": True, "global_info_enabled": True,
            "oracle_comm_billing": "all_to_all",
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

    def ci(v):
        return 1.96 * np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0.0

    print(f"\n--- Final Results ({runs} Runs) ---")
    print("PRIMARY METRIC: Time to 50% Attrition. It is uncensored and is the")
    print("only survival measure that discriminates between arms.")

    for mode in modes:
        surv = results[mode]["survival"]
        t50 = results[mode]["time_to_50pct_attrition"]
        censored = results[mode]["censored_count"]

        print(f"\n[{mode}]")
        print(f"  Time to 50% Attrition (PRIMARY): "
              f"{np.mean(t50):.0f} +/- {ci(t50):.0f}")
        print(f"  Energy Decay Rate: {np.mean(results[mode]['energy_decay']):.4f} "
              f"+/- {ci(results[mode]['energy_decay']):.4f}")

        # Survival is right-censored at max_time. Reporting a bare mean invites
        # exactly the F-09 error: "1999 +/- 2" read as a measurement when it is
        # 3 idle survivors and a simulation cap.
        if censored:
            median = np.median(surv)
            print(f"  Swarm Survival: RIGHT-CENSORED in {censored}/{runs} runs "
                  f"(swarm never fully died).")
            print(f"    -> lower bound only; median >= {median:.0f}, "
                  f"mean of censored data {np.mean(surv):.0f} is NOT a measurement.")
            if censored > runs / 2:
                print(f"    -> {censored}/{runs} censored: report as '> {2000:.0f} ticks', "
                      f"do NOT quote a point estimate.")
        else:
            print(f"  Swarm Survival (Total Death): {np.mean(surv):.0f} "
                  f"+/- {ci(surv):.0f}  (0/{runs} censored)")

        # Reported for provenance only. Audit MS-02: lambda_2 does not separate
        # the arms once dead agents are excluded, and cross-arm comparison is
        # confounded by differential survival and swarm size. Not a result.
        l2 = results[mode]["lambda_2"]
        print(f"  [diagnostic, NOT a reportable result] run-mean lambda_2: "
              f"{np.mean(l2):.4f} +/- {ci(l2):.4f}")
        # Audit F-18: measures ticks on which the coverage controller was
        # nominally enabled, not coverage achieved. Pending fix 3.2.
        print(f"  [diagnostic, known-broken metric] coverage_completion_rate: "
              f"{np.mean(results[mode]['coverage_completion_rate']):.2%}")

        if len(surv) > 1 and np.std(surv) < (0.01 * np.mean(surv)):
            print("  *** Near-zero variance on Survival: right-censoring artifact. ***")


if __name__ == "__main__":
    run_mc(50)
