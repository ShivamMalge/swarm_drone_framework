---
trigger: always_on
---

# 📜 Architectural Integrity Rules  
## Resilient Decentralized Swarm Drone Autonomy Framework  

---

## 0. Foundational Principle

This project implements a **fully decentralized, event-driven swarm autonomy simulation framework** based on a frozen theoretical specification (Chapters 1–8).

The theory and system architecture are **immutable**.

No implementation may violate:

- Decentralization purity  
- Event-driven execution model  
- Fog-of-war observability constraints  
- Stability-constrained adaptive tuning  
- Energy monotonic decay  
- Θ_safe parameter manifold enforcement  

---

# 1. Decentralization Rules

## 1.1 No Global State Access by Agents

Agents must NEVER access:

- Global adjacency matrix `A(t)`  
- Global Laplacian `L`  
- True algebraic connectivity `λ₂`  
- Global energy vector  
- Global task list  
- Global regime distribution  
- True packet drop mask  
- Global metrics logger  

Agents may only use:

- Local state `x_i`  
- Local energy `E_i`  
- Stale neighbor states  
- Local telemetry buffer  
- Local spectral proxy `λ̂₂,i`  
- Local neighbor list `𝒩_i`  

Violation invalidates the architecture.

---

## 1.2 No Centralized Task Assignment

Task allocation must:

- Be auction-based  
- Be decentralized  
- Be gossip-driven  
- Be tolerant to packet drops  
- Support rebid on timeout  

There must NEVER exist:

- Global task dispatcher  
- Central auctioneer node  
- Master controller  

---

## 1.3 No Centralized Control Loops

There must be:

- No global planner  
- No global controller  
- No global optimization routine  

All intelligence must reside inside each `AgentCore`.

---

# 2. Event-Driven Execution Rules

## 2.1 All State Changes Must Be Event-Triggered

No continuous time stepping loops allowed.

All updates must occur via:

- Min-heap event queue  
- Explicitly scheduled events  
- Simulation clock jumps  

No hidden state mutation.

---

## 2.2 Communication Must Be Scheduled

Messages must:

- Go through Communication Engine  
- Be subjected to RGG connectivity check  
- Undergo Bernoulli drop sampling  
- Undergo exponential latency sampling  
- Be injected into kernel event queue  

No direct agent-to-agent function calls allowed.

---

## 2.3 Silent Discard Enforcement

Dropped packets:

- Must not generate acknowledgments  
- Must not trigger retry loops  
- Must not inform sender of failure  

---

# 3. Fog-of-War Integrity Rules

## 3.1 Partial Observability Enforcement

Each agent must operate under:

- Stale belief buffers  
- Delayed neighbor states  
- No access to interference field `ψ(q,t)`  
- No knowledge of full topology  

If any agent computes true `λ₂` → violation.  
If any agent accesses full adjacency → violation.

---

# 4. Energy Model Rules

## 4.1 Monotonic Energy Decay

Energy `E_i` must:

- Strictly decrease over time  
- Never increase  
- Decrease with:
  - Movement  
  - Communication  
  - Computation (if modeled)  

No regeneration allowed.

---

## 4.2 Energy-Based Death

If `E_i ≤ 0`:

- Agent must be permanently removed  
- No revival  
- No resurrection  
- No negative energy allowed  

---

# 5. Stability & Adaptation Rules

## 5.1 Θ_safe Enforcement

All parameter updates must:

1. Start from `θ_nominal`  
2. Add `Δθ` (heuristic or RL)  
3. Pass through:
   - Simplex projection  
   - Box clamp  
   - Spectral radius check  
4. If unstable:
   - Run bisection loop  

No parameter may be applied without Θ_safe validation.

---

## 5.2 Spectral Radius Constraint

Closed-loop condition must enforce:

ρ(A_cl) < 1


If violated → must be corrected before application.

---

## 5.3 RL Is a Meta-Optimizer Only

RL may:

- Suggest `Δθ`  
- Suggest weight adjustments  

RL may NOT:

- Directly output control actions  
- Replace deterministic controller  
- Bypass Θ_safe projector  

---

# 6. Regime Detection Rules

## 6.1 No Global Regime Awareness

Each agent must compute:

- Local `λ̂₂` proxy  
- Local telemetry average  
- Local regime classification `𝓡_i`  

No global regime synchronization allowed.

---

## 6.2 Hysteresis & Dwell Enforcement

State transitions must:

- Pass hysteresis band  
- Satisfy dwell time `τ_dwell`  
- Prevent Zeno behavior  

No instantaneous flipping.

---

# 7. Theoretical Constraints

## 7.1 Percolation Threshold Respect

Connectivity collapse must follow:
p ≈ (c log N)/N

No artificial graph healing below threshold.

---

## 7.2 Gossip Mixing Bound Awareness

Consensus convergence must scale realistically:
O(N² log ε⁻¹)

No instant consensus allowed.

---

## 7.3 Energy Cascade Reality

Energy cascade effects must:

- Be emergent  
- Be irreversible  
- Follow depletion logic  

No artificial balancing allowed.

---

# 8. Code Structure Rules

## 8.1 Modular Architecture Enforcement

Code must mirror architecture:

- `simulation_kernel/`  
- `environment/`  
- `communication/`  
- `agent_core/`  
- `coordination/`  
- `regime_detection/`  
- `adaptation/`  
- `metrics/`  

No monolithic files allowed.

---

## 8.2 Clear Interface Boundaries

Modules must communicate through:

- Defined interfaces  
- Event messages  
- Explicit data structures  

No circular imports.

---

# 9. Implementation Discipline

## 9.1 Phase-Wise Development Only

Do not implement:

- RL  
- Adaptation  
- Regime detection  

Until previous phases are stable.

---

## 9.2 No Architecture Drift

The system must not:

- Introduce shortcuts  
- Add convenience global objects  
- Simplify away stochastic modeling  
- Replace event-driven logic with loops  

---

# 10. Research Integrity Rule

This project is:

- A research-grade decentralized systems simulator  
- Not a toy visualization  
- Not a simplified swarm demo  

All implementation must preserve publishable-level rigor.