# Decentralized Swarm Autonomy Framework (DSAF)

A lightweight, deterministic Discrete-Event Simulation (DES) framework for evaluating decentralized drone swarms under severe communication and energy constraints. 

This repository contains the official codebase for the paper: **Decentralized Swarm Autonomy Framework for Cyber-Physical Systems under Communication and Energy Constraints**.

---

## 🎯 Project Overview

Modern cyber-physical systems (CPS) often rely on decentralized swarms for robust spatial operations. However, existing simulators frequently assume perfect continuous-time communication and abstract away real thermodynamic energy costs. 

**DSAF** is engineered to test swarms under realistic, contested conditions. It enforces strict "Fog-of-War" constraints (probabilistic packet drops via a Random Geometric Graph) and models thermodynamic energy cascades. The core innovation of this framework is the **Stability-Constrained Adaptation Protocol**, which allows agents to adapt their behavior heuristically while mathematically guaranteeing they do not induce kinematic oscillations that drain the swarm's battery.

### Key Features
1. **Enforced Fog-of-War:** Agents operate exclusively on stale, asynchronous neighbor data stored in a `LocalMap`. Zero access to global topology arrays.
2. **Thermodynamic Braking:** When the local spectral proxy detects fragmentation or chaotic variance, agents mathematically clamp their velocity to `0.0`. By physically halting, agents eliminate kinetic energy penalties and coast on the baseline transmission cost ($0.05$ units/tick), successfully matching the survival rate of a centralized oracle.
3. **Bounded Heuristic Clamping:** A 5-step local Bisection Search intercepts behavioral mutations from the Hybrid Supervisor, projecting them against local stability boundaries to prevent kinematic divergence.
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
│   ├── coordination/           # Gossip Consensus, SSI Auctions, Voronoi
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
*Note: A 50-run suite executes 150 total simulations (evaluating millions of discrete events) and takes approximately 35-40 minutes on standard hardware.*

**Expected Output Metrics:**
*   **Unconstrained RGG:** Rapid kinematic oscillation leading to early death ($\sim 527$ ticks).
*   **Centralized Oracle:** Perfect survival ($\sim 1999$ ticks) with baseline energy drain.
*   **Proposed Framework:** Matches the Oracle ($\sim 1999$ ticks) purely through decentralized thermodynamic braking.

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
  author={Malge, Shivam and Hegde, Prajwal Narendra and K R, Koushik and Shruthi},
  booktitle={IEEE Student Conference},
  year={2026}
}
```
