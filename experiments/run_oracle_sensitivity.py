"""
Oracle communication-accounting sensitivity study.

The oracle receives global state every tick. What that awareness is billed for
is a modelling choice, and it decides whether the oracle behaves as an upper
bound. This script runs both accountings over identical seeds and reports them
side by side rather than asserting either one.

  all_to_all     -- billed for every other living agent (N-1 recipients).
                    Global awareness implies global bandwidth. Primary model.
  per_neighbour  -- billed only for living agents inside comm_radius, exactly
                    as the decentralized arms are. Isolates coordination
                    quality from communication cost.

Usage:
    python experiments/run_oracle_sensitivity.py [n_seeds]
"""

import sys
import os

for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import concurrent.futures

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import SimConfig
from src.simulation import Phase1Simulation

BASE = dict(num_agents=100, grid_width=100, grid_height=100, comm_radius=20.0,
            p_drop=0.1, coverage_enabled=True, log_dir="logs", max_time=2000.0)

ARMS = [
    ("Unconstrained", dict(test_mode="none", theta_safe_enabled=False,
                           global_info_enabled=False)),
    ("Proposed", dict(test_mode="none", theta_safe_enabled=True,
                      global_info_enabled=False)),
    ("Oracle [all_to_all]", dict(test_mode="none", theta_safe_enabled=True,
                                 global_info_enabled=True,
                                 oracle_comm_billing="all_to_all")),
    ("Oracle [per_neighbour]", dict(test_mode="none", theta_safe_enabled=True,
                                    global_info_enabled=True,
                                    oracle_comm_billing="per_neighbour")),
]


def run_one(task):
    seed, name, kwargs = task
    cfg = SimConfig(seed=seed, **kwargs)
    sim = Phase1Simulation(cfg)
    sim.run()
    sim.close_loggers()
    s = sim.summary()

    tod = s["time_of_death"]
    censored = tod is None
    surv = cfg.max_time if censored else tod

    t50 = s["time_to_50pct_attrition"]
    t50_censored = t50 is None
    if t50_censored:
        t50 = cfg.max_time

    init_e = cfg.num_agents * cfg.energy_initial
    decay = (init_e - s["total_energy_remaining"]) / (cfg.num_agents * surv) if surv > 0 else 0.0
    l2 = np.mean([e["spectral_gap"] for e in sim.connectivity_log]) if sim.connectivity_log else 0.0

    return name, dict(survival=surv, censored=censored, t50=t50,
                      t50_censored=t50_censored, decay=decay, lambda_2=l2,
                      alive_end=int(sim.alive_mask.sum()))


def main(n_seeds=10):
    tasks = [(1000 + i, name, dict(BASE, **kw))
             for i in range(n_seeds) for name, kw in ARMS]

    acc = {name: [] for name, _ in ARMS}
    with concurrent.futures.ProcessPoolExecutor() as ex:
        for name, res in ex.map(run_one, tasks):
            acc[name].append(res)

    def ci95(v):
        return 1.96 * np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0.0

    print(f"\nOracle communication-accounting sensitivity -- {n_seeds} seeds, "
          f"N={BASE['num_agents']}, max_time={BASE['max_time']:.0f}\n")
    hdr = (f"{'arm':<24} {'lambda_2':>14} {'decay/tick':>14} "
           f"{'survival':>15} {'cens':>6} {'t50':>13} {'alive_end':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name, _ in ARMS:
        r = acc[name]
        surv = [x["survival"] for x in r]
        t50 = [x["t50"] for x in r]
        dec = [x["decay"] for x in r]
        l2 = [x["lambda_2"] for x in r]
        alive = [x["alive_end"] for x in r]
        nc = sum(x["censored"] for x in r)
        n50 = sum(x["t50_censored"] for x in r)
        print(f"{name:<24} {np.mean(l2):7.3f}+/-{ci95(l2):<5.3f} "
              f"{np.mean(dec):7.4f}+/-{ci95(dec):<5.4f} "
              f"{np.mean(surv):8.0f}+/-{ci95(surv):<5.0f} "
              f"{nc:>3d}/{len(r):<2d} {np.mean(t50):6.0f}+/-{ci95(t50):<5.0f} "
              f"{np.mean(alive):10.1f}")
        if n50:
            print(f"{'':<24} (t50 censored in {n50}/{len(r)} runs -- "
                  f"swarm never lost 50%)")

    print("\nCensoring: 'cens' counts runs where the swarm never fully died, so "
          "survival is recorded as max_time and the mean is a lower bound.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
