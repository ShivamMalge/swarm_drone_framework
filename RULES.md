# RULES.md — Swarm Autonomy Framework (STRICT EXECUTION CONTRACT)

---

## 1. SYSTEM IDENTITY

This project is a Decentralized Swarm Autonomy Framework built as a research-grade cyber-physical system simulator.

It is NOT:
- a visualization project
- a toy simulation
- a UI-first system

It IS:
- a deterministic Discrete Event Simulation (DES) system
- a decentralized multi-agent coordination framework
- a research-grade experimentation platform

---

## 2. CORE ARCHITECTURE PRINCIPLE (NON-NEGOTIABLE)

The system follows a strict layered architecture:

DES Kernel (Core Engine)
→ Coordination Algorithms
→ Regime Detection
→ Adaptive Control
→ Metrics & Telemetry
→ Dashboard (Observer ONLY)

HARD RULES:
- GUI MUST NEVER interact with kernel state directly
- Kernel MUST remain pure and deterministic
- Dashboard MUST be read-only observer
- NO bidirectional coupling between GUI and simulation

---

## 3. DISCRETE EVENT SIMULATION (DES) RULES

- Simulation MUST be event-driven (NO loops like "for agent in agents")
- Use min-heap event queue
- Time advances ONLY via event timestamps
- Events include:
  - MSG_TRANSMIT
  - MSG_DELIVER
  - KINEMATIC_UPDATE
  - REGIME_UPDATE
  - AUCTION_RESOLVE

STRICT:
- NO real-time stepping
- NO frame-based simulation logic
- NO hidden state mutations

---

## 4. DETERMINISM & REPRODUCIBILITY

- Use "numpy.random.SeedSequence"
- Every subsystem must have independent RNG stream
- Simulation MUST be reproducible exactly

If same seed → EXACT SAME OUTPUT

---

## 5. FOG-OF-WAR CONSTRAINT (CRITICAL)

Agents MUST NEVER access:
- global adjacency matrix
- global state tensor
- true λ₂ (spectral value)
- other agents' real-time states

Agents ONLY use:
- LocalMap (delayed + partial info)
- stale neighbor data
- TTL-based belief expiration

Violation = ARCHITECTURE FAILURE

---

## 6. COMMUNICATION PIPELINE RULES

Pipeline MUST follow:
1. RGG Connectivity Check
2. Bernoulli Packet Drop
3. Exponential Latency

SILENT DISCARD RULE:
- Dropped packets MUST NOT notify sender
- NO ACK / retry logic allowed

---

## 7. ENERGY MODEL CONSTRAINT

- Energy MUST be strictly monotonic decreasing
- NO regeneration allowed

Energy consumption includes:
- movement cost
- communication cost
- idle cost

If:
E_i <= 0
→ Agent MUST be permanently removed

---

## 8. COORDINATION ALGORITHMS (STRICT)

Only allowed:
- Gossip Consensus
- Voronoi Coverage (Lloyd-based)
- CBBA Auction Allocation

Constraints:
- MUST operate on LocalMap only
- MUST tolerate stale data
- MUST handle packet loss + delay

---

## 9. REGIME DETECTION SYSTEM

System MUST classify into:
- Stable
- Intermittent
- Marginal
- Fragmented
- Energy Cascade
- Latency Oscillation

Inputs:
- spectral proxy (λ̂₂)
- neighbor density
- staleness
- consensus variance

---

## 10. ADAPTIVE CONTROL RULES

- RL / heuristic adaptation allowed ONLY via:
  Θ_safe projection

MUST include:
- simplex projection
- box constraints
- spectral radius validation (ρ < 1)

STRICT:
Unsafe parameters MUST be rejected or projected

---

## 11. TELEMETRY ARCHITECTURE

Pipeline:
DES Kernel
→ Telemetry Emitter
→ Telemetry Buffer
→ Simulation Worker (QThread)
→ Telemetry Bridge
→ GUI

RULES:
- Kernel NEVER blocked by GUI
- GUI runs on main thread only
- Simulation runs on worker thread

---

## 12. THREADING MODEL

- Use PySide6 QThread
- NO blocking calls in GUI thread
- NO shared mutable state without control

Preferred:
- signals/slots
- lock-free buffers

---

## 13. ARCHITECTURE DIAGRAM INTERPRETATION RULE

All diagrams in "/architecture" are AUTHORITATIVE REFERENCES

They define:
- execution order
- event causality
- module boundaries

AI MUST:
- follow diagrams strictly
- convert them into code logic
- NOT ignore them

If conflict occurs:
→ diagrams override assumptions

---

## 14. CODE DESIGN RULES

- Modular structure REQUIRED
- No monolithic files
- Follow existing repo structure strictly

Each module MUST have:
- single responsibility
- clear input/output
- no hidden dependencies

---

## 15. PERFORMANCE REQUIREMENTS

Target:
- real-time simulation + visualization
- 60 FPS rendering

Use:
- NumPy vectorization
- PyQtGraph
- batch updates

---

## 16. FORBIDDEN ACTIONS 🚫

DO NOT:
- redesign architecture
- simplify system logic
- introduce centralized control
- bypass fog-of-war
- couple GUI with kernel
- replace DES with loops
- remove stochastic models

---

## 17. DEVELOPMENT MODE

AI must operate in:
→ STRICT EXECUTION MODE

Meaning:
- Follow instructions EXACTLY
- Do NOT improvise architecture
- Do NOT downgrade complexity
- Do NOT simplify math or logic

---

## 18. SOURCE OF TRUTH

Order of priority:
1. rules.md
2. architecture diagrams
3. existing repo structure
4. methodology documents

---

## 19. OUTPUT EXPECTATIONS

All outputs must be:
- production-grade
- modular
- architecture-compliant
- ready to integrate

---

## 20. CURRENT OBJECTIVE

We are implementing:
→ Phase 2 — Forensic Telemetry Dashboard

AI must:
- extend system WITHOUT breaking kernel
- build observer layer only
- maintain strict separation

---

## 21. FINAL DIRECTIVE

This is a research-grade system, not a demo.

Every decision must:
- preserve determinism
- preserve decentralization
- preserve physical realism
- preserve architectural purity

Failure to follow rules = invalid output
