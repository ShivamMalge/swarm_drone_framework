# Decentralized Swarm Autonomy Framework (DSAF)

A lightweight, deterministic Discrete-Event Simulation (DES) framework for evaluating decentralized drone swarms under severe communication and energy constraints. 

This repository contains the official codebase for the paper: **Decentralized Swarm Autonomy Framework for Cyber-Physical Systems under Communication and Energy Constraints**.

---

## 🎯 Project Overview

Modern cyber-physical systems (CPS) often rely on decentralized swarms for robust spatial operations. However, existing simulators frequently assume perfect continuous-time communication and abstract away real thermodynamic energy costs. 

**DSAF** is engineered to test swarms under realistic, contested conditions. It enforces strict "Fog-of-War" constraints (a Random Geometric Graph whose per-transmission survival probability combines a baseline drop rate, exogenous interference, and distance-dependent path loss) and models thermodynamic energy costs. Its central mechanism is a **Stability-Constrained Adaptation Pipeline** -- a static safety box constraint plus a bisection search against a locally computed delay-tolerant bound, followed by EMA smoothing. The paper makes no claim of a mathematical stability guarantee: the pipeline is a heuristic bounding layer, and the paper reports measured behaviour, including its negative results.

### Key Features
1. **Enforced Fog-of-War:** Agents operate exclusively on stale, asynchronous neighbor data stored in a `LocalMap`. Zero access to global topology arrays.
2. **Emergent Halt Under Isolation:** No velocity is ever clamped to zero. When belief eviction empties an isolated agent's map, the localized Voronoi coverage law lands on a stationary fixed point (an isolated agent's cell centroid is its own position), so fragmented remnants halt and coast at idle-plus-transmission cost. This emergent mechanism -- not the adaptation pipeline -- produces the measured 5.2x reduction in mean energy decay versus the unconstrained baseline.
3. **Bounded Heuristic Clamping:** A two-stage projector intercepts every parameter proposal: a static box clamp (the stage responsible for the energy result) followed by a 5-step bisection against a locally computed consensus step-size bound. The bound demonstrably prevents numerically unbounded consensus divergence, and demonstrably does not affect energy or attrition -- both facts are reported in the paper.
4. **Deterministic Reproducibility:** The DES kernel is driven by independent, pre-seeded PRNG streams, ensuring exact stochastic replay of packet drops and delays.

---

## 📂 Repository Structure

```text
swarm_drone_framework/
│
├── src/                        # Core Simulator Engine
│   ├── adaptation/             # Hybrid Supervisor & Bisection Projector
│   ├── agent/                  # Decentralized Agent Logic & Fog-of-War Map
│   ├── communication/          # RGG, Packet Drop Physics, Latency Models
│   ├── coordination/           # Gossip Consensus, Energy-Aware Auctions, Voronoi
│   ├── core/                   # Event Queue, Config, and DES Kernel
│   └── regime/                 # Local Spectral Proxy & Variance Trackers
│
├── experiments/                # Executable Simulation Scripts
│   ├── run_monte_carlo_table.py  # 50-run suite (Generates paper statistics)
│   ├── run_energy_cascade.py     # Simulates high node attrition
│   └── run_percolation.py        # Environmental jamming tests
│
├── manuscript/                 # LaTeX Source for the IEEE Paper
└── README.md                   # This file
```

---

## ⚙️ Installation & Setup

DSAF is written in pure Python, prioritizing rapid algorithmic experimentation and deterministic execution.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShivamMalge/swarm_drone_framework.git
   cd swarm_drone_framework
   ```

2. **Install dependencies:**
   The framework relies heavily on `numpy` for localized mathematical processing and `scipy` for KD-Tree spatial partitioning.
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running Experiments

The experiments reproduce the quantitative metrics presented in the paper. All simulations are deterministic given a specific PRNG seed.

### Generating Table III (Monte Carlo Baselines)
To run the primary 50-seed Monte Carlo evaluation comparing the Unconstrained RGG, the Centralized Oracle, and the Proposed Framework:

```bash
python experiments/run_monte_carlo_table.py
```
*Note: A 50-run suite executes 150 total simulations (~20 million discrete events) and takes roughly 3-4 minutes on an 8-core machine.*

**Expected Output Metrics (means over 50 seeds; see the paper for confidence intervals and caveats):**
*   **Unconstrained RGG:** energy decay ~0.26 units/tick; total swarm death at ~407 ticks.
*   **Centralized Oracle (all-to-all comm. cost):** the shortest-lived arm (~39 ticks) -- global awareness is billed as global bandwidth, and that cost consumes its coordination benefit.
*   **Proposed Framework:** energy decay ~0.05 units/tick (5.2x below baseline); all 50 runs right-censored at the 2000-tick horizon. Time to 50% attrition is statistically indistinguishable from the unconstrained baseline (~128 vs ~127 ticks) -- the framework does not delay half-swarm loss, and the paper says so.

---

## 🔬 Future Work
As outlined in the manuscript, future development of this framework will focus on:
* **Embedded Deployment:** Translating the discrete-event physics kernel into memory-safe environments (Rust/C++) to overcome Python's GIL and scale to $N > 10,000$ agents.
* **Hardware-in-the-Loop (HITL):** Validating the asynchronous gossip protocols over physical RF transceivers.

---

## 📖 Citation

If you utilize this simulation framework for academic research, please cite our paper:

```bibtex
@inproceedings{malge2026decentralized,
  title={Decentralized Swarm Autonomy Framework for Cyber-Physical Systems under Communication and Energy Constraints},
  author={Malge, Shivam and Hegde, Prajwal Narendra and K R, Koushik},
  booktitle={IEEE Student Conference},
  year={2026}
}
```
