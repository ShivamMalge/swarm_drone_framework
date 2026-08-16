# Audit Findings — Swarm Drone Autonomy Framework

**Date:** 2026-08-16
**Scope:** `src/**`, `experiments/**`, repo-root scripts, `manuscript/final_manuscript.tex`, `draft 11.pdf`, `logs/`, `plots/`, `tests/`
**Method:** full source read + direct execution of probe experiments. Every claim below marked **[VERIFIED]** was reproduced by running code, not inferred from reading.

---

## 0. Executive summary

The most serious problem in this repository is **not** the two already-known fabrications. It is that the paper's headline result — Table III, "Proposed Framework matches the Centralized Oracle" — is produced by three arms of a four-arm ablation that are **not actually different experiments**, measuring a quantity (`λ₂`) that is **computed over dead agents**, with a survival number that is a **right-censoring artifact describing 3 surviving drones out of 100**.

Ranked by severity:

| # | Finding | Status |
|---|---|---|
| **F-01** | `manuscript/fig1/2/3.png` are byte-identical to the **pre-fix** plots — the fabricated thermodynamics figure is still embedded in the paper | **[VERIFIED]** md5 match |
| **F-02** | "Static-Epsilon Baseline" and "Proposed" produce **bit-identical** positions and energies on every seed tested | **[VERIFIED]** |
| **F-03** | Reported `λ₂` is computed over **all 100 positions including corpses**; alive-only λ₂ is 0.0 while the code logs 0.29 | **[VERIFIED]** |
| **F-04** | The manuscript's stated causal mechanism ("clamps velocity to zero") is **false** — survivors have `velocity_scale = 1.000` | **[VERIFIED]** |
| **F-05** | Fig. 2's "Scaled Local Proxy λ̂₂" is **mean neighbour count ÷ 100**, not the Eq. (2) proxy | **[VERIFIED]** |
| **F-06** | `experiments/run_mc_fast.py:26` fabricates survival: `surv = max_time if alive>0 else max_time*0.7` | **[VERIFIED]** |
| **F-07** | The false-audit sentence exists **only in the `.tex`**, added after the last PDF compile | **[VERIFIED]** |
| **F-08** | 19 of 106 runnable tests fail; 23 more test files cannot be collected | **[VERIFIED]** |

---

## 1. Status of the 8 previously-known issues

Several were **already partially fixed in the uncommitted working tree** by earlier work. This audit reports what is true *now*, with fresh line references. Do not assume the briefing list is current.

| # | Known issue | Current status |
|---|---|---|
| 1 | `static_bounded` is not an oracle | **STILL TRUE.** `src/agent/agent_core.py:522` — `dynamic_bounds = {...} if self.test_mode != "static_bounded" else {}`. Skips one bound; grants no global information. A *real* oracle (`global_info_enabled`) has since been added, but Table III still labels the fake one "Centralized Oracle (Static Bound)". See **F-02**, which is worse than originally described. |
| 2 | `coverage_enabled` never True | **PARTIALLY FIXED.** Default is still `False` (`src/core/config.py:108`), but `run_monte_carlo_table.py` (lines 83, 91, 99, 107) and `run_percolation.py:22` now set `True`. **`experiments/run_stability_test.py` still does not** — Figure 1's experiment runs with Algorithm 3 disabled. |
| 3 | No 50%-attrition metric | **FIXED.** `src/simulation.py:157-162` `_check_death_conditions`, surfaced at `summary()` line 742 and reported at `run_monte_carlo_table.py:150`. **[VERIFIED]** returns `t50=105.0`/`106.0`. See **F-09** for what this reveals. |
| 4 | Survival is right-censored | **INSTRUMENTATION FIXED, PAPER NOT.** `run_monte_carlo_table.py:29-33, 152-155` now counts censored runs and warns. The manuscript still reports `1999 ± 2` with no censoring note. |
| 5 | Braking is shutdown, not coasting | **CONFIRMED AND WORSE.** See **F-04** — it is neither. |
| 6 | Algorithm 1 missing box-clamp step | **STILL TRUE**, and this is the *load-bearing* step. See **F-11**. |
| 7 | Only 3 auction tasks | **FIXED, BUT A NEW DEFECT REPLACES IT.** `src/simulation.py:516` now uses `exponential(15.0)` Poisson spawning → **[VERIFIED]** 129 tasks per 2000-tick run. However only **26 are ever won**; 122 remain unallocated. See **F-13**. |
| 8 | Time axes mislabelled `(s)` | **FIXED.** `experiments/plot_results.py:24` `DT = 0.1`, applied at lines 34, 64, 87. |

---

## 2. Fabricated / stand-in data still present

### F-01 — The manuscript still embeds the fabricated figures **[VERIFIED]**

`experiments/plot_results.py` was repaired in the working tree (the `np.maximum(0, max_energy - time*0.5)` synthesis is gone; `plot_thermodynamics()` now reads `framework_total_energy` / `baseline_total_energy` from the CSV at lines 95-96). The regenerated PNGs sit in `plots/`.

**But the figures compiled into the paper were never updated.** md5 comparison:

```
fig1.png  manuscript=81216ba4bb12d8e2a89d2dab15caf13e
   OLD (committed) plots/kinematic_stability.png = 81216ba4... -> match: YES
   NEW (regenerated)                             = 44d65f5c... -> match: no
fig2.png  manuscript=3ddc72f3d040fa64e66b5270178e1c5c
   OLD (committed) plots/spectral_stability.png  = 3ddc72f3... -> match: YES
   NEW (regenerated)                             = e405fce4... -> match: no
fig3.png  manuscript=21344ea8fd9750d40fe778e94dfb9848
   OLD (committed) plots/thermodynamic_decay.png = 21344ea8... -> match: YES
   NEW (regenerated)                             = 48cf4e5e... -> match: no
```

`manuscript/fig3.png` (Figure 4 in the paper, `\ref{fig:thermo}`) is **byte-identical to the plot generated by the fabricating code**. Timestamps agree: `manuscript/fig*.png` = Jul 4 12:45; `plots/*.png` = Jul 8 10:40.

There is no build step linking `plots/*.png` → `manuscript/fig*.png`. The mapping is manual and undocumented, which is precisely how the fabrication survived its own fix.

`manuscript/fig6_ablation.png` exists but is referenced nowhere in the `.tex` — an orphan from an earlier draft.

### F-06 — `run_mc_fast.py` fabricates survival **[VERIFIED]**

`experiments/run_mc_fast.py:26`:

```python
surv = cfg.max_time if alive > 0 else cfg.max_time * 0.7
```

Survival is **never measured**. If any agent lives, survival is declared to be exactly `max_time`; otherwise it is declared to be exactly 70 % of `max_time`. The `0.7` is an invented constant. Line 24 has the same defect for decay:

```python
decay = (init_energy - rem_energy) / (cfg.num_agents * cfg.max_time)
```

— divided by `max_time` regardless of when the swarm actually died. This script prints a table in the same format as `run_monte_carlo_table.py`, so its output is indistinguishable from a real result.

### F-15 — Figure 1 is annotated with events that do not exist **[VERIFIED]**

`experiments/plot_results.py:42-45` draws two labelled vertical lines:

```python
plt.axvline(x=50 * DT, ...); plt.text(52*DT, 2.0, 'Obstacle Avoidance Trigger', ...)
plt.axvline(x=150 * DT, ...); plt.text(152*DT, 2.0, 'Severe Packet Loss Spike', ...)
```

- `grep -rni "obstacle" src/ experiments/` returns **only this line**. There is no obstacle-avoidance behaviour anywhere in the codebase.
- `experiments/run_stability_test.py` sets no `p_drop` or `psi` schedule; both are constant for the whole run. There is no packet-loss spike.

The code comment even concedes the guess: `# Adjust annotations for physical time (assuming original tick indices were 50 and 150)`. These are invented narrative events drawn onto a research figure.

---

## 3. Self-checks that report success without verifying

### F-16 — `diagnostics_audit.py` logs hardcoded `True`

Three of ten checks pass a literal `True` as the success argument and can never fail:

- **line 110** — `self.log_result("Task Resolution", True, "Tasks consumed strictly via spatial arrival (r_task threshold).")`. Nothing about task consumption is inspected; `sim.run()` is called and the result discarded.
- **line 131** — `self.log_result("Stability Constraints", True, f"Projection manifold active. Recorded {total_projections} projection events.")`. Reports the count but asserts nothing about it; passes identically at 0 events.
- **line 146** — `self.log_result("Communication Overhead", True, "No global broadcasts detected; all gossip is RGG-localized.")`. Nothing is measured at all — the function body is two comments and the log call.

Two more are misleading rather than vacuous:

- **line 168** — `duration < 30.0` is a wall-clock performance check, reported as `"Stress Simulation ... Stable under load."` A slow CI machine fails it; a numerically broken simulation passes it.
- **line 88** — the Fog-of-War check tests `hasattr(agent, attr)` for `['kernel','orchestrator','all_agents','global_state']`, none of which any version of `AgentCore` has ever defined. It cannot fail, and it does not detect the real global-information channel that now exists (`oracle_centroid`, set externally at `src/simulation.py:232`).
- **line 41** — `events1 = sim1.kernel.peak_queue_size  # Just a proxy for now, but we can do better` — assigned, never used.

### F-08 — The real test suite is red and largely uncollectable **[VERIFIED]**

```
19 failed, 87 passed in 263.22s
... plus 23 test files that ERROR on collection: ModuleNotFoundError: No module named 'PySide6'
```

`PySide6>=6.5` is in `requirements.txt`, so the environment does not match the declared requirements — the entire GUI/analytics half of the suite has not been executed in this environment at all.

Of the 19 failures, these matter most:

- **`tests/test_safety_projector.py`** — all 4 tests raise `TypeError: project_to_theta_safe() missing 1 required positional argument: 'theta_nominal'`. The signature changed and the tests were never updated. **Algorithm 1 — the paper's headline contribution — currently has zero passing test coverage.**
- **`tests/test_no_global_access.py::test_no_global_state_reference_in_source`** — fails: `AgentCore source contains forbidden reference: 'lambda_2'` (matches `self._lambda_2_proxy` at `agent_core.py:153`). The repo's own fog-of-war guard is red.
- **`tests/test_packet_drop.py`** — 4 failures including `test_zero_drop_never_drops`. The drop model gained a distance path-loss term (F-20) and the contract tests were never updated: with `p_drop = 0`, packets are still dropped.
- **`tests/test_scalability_n100.py::test_drop_rate_within_tolerance`** — `Drop rate 0.642 outside expected range` (expected ≤ 0.55). Confirms the effective drop rate is far above nominal.
- **`tests/test_replay.py`** — `TypeError: TelemetryFrame.__init__() missing 1 required positional argument: 'agent_states'` (see F-25).
- `tests/test_regime_detection.py` (5), `tests/test_hybrid_supervisor.py` (2), `tests/test_auction.py::test_dropout_robustness` (1).

---

## 4. Core engine: findings that invalidate reported results

### F-02 — "Static-Epsilon Baseline" and "Proposed" are the same simulation **[VERIFIED]**

`run_monte_carlo_table.py` distinguishes arm 2 from arm 3 solely by `test_mode="static_bounded"` vs `"none"`. That flag reaches exactly one line, `src/agent/agent_core.py:522`:

```python
dynamic_bounds = {"gossip_epsilon": safe_bound} if self.test_mode != "static_bounded" else {}
```

Probe result (N=100, `max_time=600`, all else identical):

```
seed 1000: positions identical=True  energies identical=True  max|dpos|=0.000e+00  alive 4 vs 4
seed 1001: positions identical=True  energies identical=True  max|dpos|=0.000e+00  alive 4 vs 4
seed 1002: positions identical=True  energies identical=True  max|dpos|=0.000e+00  alive 3 vs 3
```

**Bit-identical**, every seed. The reason is F-10: `gossip_epsilon` influences only `consensus_state`, which influences only the variance-dependent regime branches, which are never the deciding branch in these runs.

The manuscript reads this identity as a scientific result (§V-A): *"the Proposed Framework perfectly matches the Centralized Oracle's maximum swarm survival (1999 ticks) and baseline energy decay rate (0.05 units/tick)"*. It matches because it **is the same run**.

### F-03 — `λ₂` is computed over dead agents **[VERIFIED]**

`src/simulation.py:566-569`:

```python
all_pos = self._get_all_positions()          # -> np.array([a.position for a in self.agents]) — ALL agents
metrics = compute_connectivity_metrics(all_pos, self.config.comm_radius)
```

`_get_all_positions()` (line 680) iterates `self.agents` unconditionally. Dead agents retain their final position and continue to contribute graph edges forever.

Probe (seed 1000, `theta_safe=False`, `max_time=800`):

```
alive at end: 0/100
lambda_2 over ALL 100 positions (what the code logs) = 0.2919  LCC=100
lambda_2 over 0 ALIVE positions only                 = 0.0000  LCC=0
```

The entire swarm is dead and the framework reports `λ₂ = 0.29` with a largest connected component of 100. **Table III's Spectral Connectivity column measures the geometric arrangement of corpses.** This also explains why the column barely varies across conditions (0.25 / 0.29 / 0.29).

The same defect corrupts the percolation log. `logs/experiment_1_percolation.csv` final rows:

```
491.0, 0.0,                  19.428..., 1102, 0, 7, ...
496.0, 1.1907141939104804e-14,19.428..., 1101, 0, 7, ...
```

`active_network_edges` **rises to 1101** while only 7 of 200 agents are alive.

Three further problems in the same call:
- `self.config.comm_radius` (static 20.0) is used, ignoring `tx_power_scale`, which the adaptation layer drives up to 2.0×. The reported graph is not the graph the swarm actually uses.
- Packet drop and interference are ignored entirely — this is the *ideal* geometric graph, not the realised communication graph.
- `src/metrics/connectivity_metrics.py:105` uses `np.linalg.eigvals` on a symmetric Laplacian, then `np.sort(np.real(...))[1]`. `eigvalsh` is correct here; `eigvals` returns unordered, complex-typed results. A correct implementation already exists in this repo at `src/analytics/spectral_analyzer.py:112-156` (uses `eigvalsh`, excludes dead agents, returns 0.0 for disconnected graphs) — **it is simply not the one the paper pipeline calls.**

### F-04 — The stated braking mechanism does not occur **[VERIFIED]**

Manuscript §V-A: *"the Proposed Framework's Θ_safe bounded heuristic **clamps the drones' velocity to zero** when it detects chaotic local variances."*

Probe, end of a 2000-tick Proposed run (seed 1000):

```
alive=3/100 at t=2000.0
  agent 15: strategy=CONNECTIVITY_RECOVERY regime=FRAGMENTED vel_scale=1.000 cov_gain=0.500 auction=0.000 bcast=1.000 E=32.08
  agent 61: strategy=CONNECTIVITY_RECOVERY regime=FRAGMENTED vel_scale=1.000 cov_gain=0.500 auction=0.000 bcast=1.000 E=4.37
  agent 67: strategy=CONNECTIVITY_RECOVERY regime=FRAGMENTED vel_scale=1.000 cov_gain=0.500 auction=0.000 bcast=1.000 E=13.73
```

`velocity_scale = 1.000` — **not clamped to zero.** The survivors are in `CONNECTIVITY_RECOVERY`, not `ENERGY_CONSERVATION`. No velocity clamping happens.

**The actual mechanism** is a degenerate case of the coverage law, and it is an artifact:

1. `HybridSupervisor.propose_parameters` for `CONNECTIVITY_RECOVERY` sets `coverage_gain = 0.0` (`hybrid_supervisor.py:84`).
2. With `theta_safe_enabled=True`, the **box clamp** (`safety_projector.py:62-68`, bounds `(0.5, 2.0)`) raises it to **0.5**.
3. `0.5 > 0.05`, so `compute_velocity` takes the coverage branch (`agent_core.py:216`).
4. An isolated agent has no neighbours, so `compute_local_centroid` returns `agent_pos.copy()` (`voronoi_coverage.py:49`).
5. `velocity = k·(centroid − position) = k·0 = **0**`. The agent stops — because its own position is trivially its own Voronoi centroid.

With `theta_safe_enabled=False`, `coverage_gain` stays `0.0`, the branch is skipped, and the agent falls through to `agent_core.py:238-243`:

```python
else:
    # Simple random walk placeholder (deterministic backbone)
    direction = self._rng.standard_normal(2)
    ...
    return direction * self._v_max * 0.5 * self.velocity_scale
```

**The baseline burns its battery on an acknowledged Phase-1 random-walk placeholder.** Probe 1 confirms the branch selection for every strategy. So Table III's 527-vs-1999 gap is: *a placeholder random walk versus a coverage controller whose fixed point for an isolated agent is "do not move."* It is not "thermodynamic braking", not stability-constrained adaptation, and not a fair ablation.

The manuscript's supporting narrative — *"the Unconstrained swarm engages in rapid, erratic 'kinematic oscillation' based on blowing-up mathematical errors"* — has no mechanism in the code. Every supervisor proposal is a bounded constant in `[0, 2]`; nothing can blow up.

### F-09 — Survival numbers describe 3 drones, and both arms lose half the swarm at the same time **[VERIFIED]**

```
[Unconstrained]  events=111727  alive=0/100    ToD=434.0  t50=105.0
[Static-Epsilon] events=136935  alive=3/100    ToD=None   t50=106.0
[Proposed]       events=136935  alive=3/100    ToD=None   t50=106.0
```

- **Time to 50 % attrition is 105 vs 106 ticks** — a 1 % difference. The two conditions are indistinguishable over the half-life of the swarm.
- The entire 527 → 1999 gap lives in the tail, after 97 % of the swarm is already dead.
- `ToD=None` means the run was **censored**; `run_monte_carlo_table.py:29-33` then substitutes `surv = config.max_time`. "Mean Swarm Survival 1999 ± 2 ticks" is the statement *"3 of 100 drones sat still until the simulation clock ran out"*, with `±2` reflecting nothing but the logging interval.

### F-10 — ~~The consensus subsystem has no causal effect on any reported metric~~ **[CORRECTED 2026-08-16 — the original explanation below is WRONG]**

> **CORRECTION.** The finding that the bisection has no measurable effect (F-02's bit-identity) stands, but the causal explanation given below does not. Instrumenting `RegimeClassifier.classify` over 10,221 classifications (3 seeds) shows the variance-dependent branches are reached **25.80 % of the time** (`LATENCY_OSCILLATION` 23.68 %, variance-`MARGINAL` 2.12 %) — `FRAGMENTED` does *not* always fire first, and consensus is **not** causally inert.
>
> The real mechanism is a **duplicated bound**: the projector genuinely moves `gossip_epsilon` (mean 0.0159 with bisection ON vs 0.0430 with it OFF), but `gossip_consensus.py:69` independently recomputes `safe_bound = 0.99/(d_i(τ_max+1))` and clamps on **98.5 % of calls**, so the epsilon actually applied is identical to three significant figures in both arms (0.003897 vs 0.003901). Algorithm 1's dynamic bound is overwritten downstream by a second implementation of itself — and the two disagree on τ discretisation (`ceil` vs `int`, F-24). The bisection is **redundant**, not unreachable.
>
> Resolution: option (b) of `fixes_phases.md` §2.1 — the internal clamp is removed and the projector made the sole enforcement point. Full evidence there.

### F-10 (original text, superseded) — The consensus subsystem has no causal effect on any reported metric **[VERIFIED]**

Probe 6:

```
dynamic-bound checks = 2588, bisection actually fired = 2581 (99.73%)
adaptation_log final: gossip_epsilon = 2.8991446613713206e-05
```

The Algorithm 1 bisection fires almost every time and drives `gossip_epsilon` from 0.05 to **2.9 × 10⁻⁵** — i.e. it switches consensus off. Yet F-02 shows positions and energies are unchanged whether the bound is applied or not. `consensus_state` propagates only to `compute_local_consensus_variance` → `mean_variance` → the `LATENCY_OSCILLATION` / `MARGINAL` branches of `RegimeClassifier.classify` (`classifier.py:65-71`), and those branches are never reached because `FRAGMENTED` (line 55, on density/staleness) fires first.

Consequence: **Contribution 1's "gossip-based consensus", Eq. (6), Eq. (2), and the ε-bound that Algorithm 1 exists to enforce are all causally inert** with respect to every number in the paper.

Root cause of the collapse to 2.9e-05: `safe_bound = 0.99 / (d_i · (τ_max + 1))` (`agent_core.py:518`). Reaching 2.9e-05 requires `d_i·(τ_max+1) ≈ 34,000` — which happens because of F-12.

### F-11 — Algorithm 1's missing Step 1 is the step that does all the work

The manuscript's Algorithm 1 pseudocode (`.tex` lines 177-199) contains only the bisection search. The implementation has **two** stages (`safety_projector.py`):

- **Step 1, lines 59-72** — static box clamp against `THETA_SAFE_BOUNDS` (lines 21-28). *Absent from the paper.*
- **Step 2, lines 74-98** — the bisection. *This is all the paper documents.*

Per F-04, Step 1 (`coverage_gain: 0.0 → 0.5`) is the sole cause of the entire Table III effect, and per F-10 Step 2 changes nothing observable. **The paper documents the inert half of its own algorithm and omits the load-bearing half.**

Two secondary mismatches:
- The pseudocode states `**Precondition:** θ_nominal ≤ θ_bounds ≤ θ_prop`. The code handles violation of that precondition at lines 82-83 (`if low >= bound: low = bound * 0.99`) — an unstated fallback that fires routinely.
- `hybrid_supervisor.py:92` — `theta["velocity_scale"] = 0.0 # Intent: Freeze. Clamped to 0.5`. **The comment is wrong.** `THETA_SAFE_BOUNDS["velocity_scale"] = (0.0, 1.5)`, so `0.0` passes through unclipped. (The equivalent comment at line 84 for `coverage_gain` *is* correct.)

### F-12 — `LocalMap` never evicts stale beliefs **[VERIFIED by grep]**

`src/agent/local_map.py:74` defines `remove_neighbor`; `grep -rn "remove_neighbor" src/ experiments/ *.py` finds **no caller anywhere**. There is no age-based eviction either.

Once agent *i* has ever received one message from agent *j*, *j* remains in *i*'s belief map for the rest of the run, at a frozen position, **including after *j* dies**. Consequences:

- `compute_neighbor_density` counts phantom neighbours → agents believe they are well-connected while isolated → the `FRAGMENTED` trigger fires late.
- `compute_information_staleness` grows without bound → `τ_max` explodes → the ε bound collapses (F-10).
- `compute_local_centroid` steers agents relative to dead agents' last known positions.
- `src/regime/local_proxies.py:26` documents a cleanup that does not exist: *"Stale neighbors inherently persist in LocalMap **until drop heuristics clean them**"*.

### F-13 — The auction mechanism degrades to inactivity **[VERIFIED]**

```
tasks spawned = 129, total auction wins = 26, tasks still unallocated at end = 122
auction wins timeline: first=5.5  last=183.05     (run length = 2000)
```

**80 % of tasks are never allocated, and no auction has been won for the final 91 % of the run.** Two compounding causes:

1. `auction_participation` is set to `0.0` by `CONSENSUS_PRIORITY`, `CONNECTIVITY_RECOVERY`, **and** `ENERGY_CONSERVATION` (`hybrid_supervisor.py:81, 86, 91`). `handle_auction_start`/`handle_auction_resolve` return immediately when it is `≤ 0.05` (`agent_core.py:397, 420`). Auctions are therefore disabled in **4 of 6 regimes**.
2. `prepare_broadcast` gossips **one uniformly random** active auction per transmission (`agent_core.py:311-314`). Unwon tasks are never removed from `active_auctions`, so the dictionary grows to 122+ entries. With `auction_timeout = 5.0` ticks and one broadcast per tick, the chance of gossiping the *relevant* bid before resolution falls to ≈ 5/122 ≈ 4 %. **The mechanism is self-throttling by construction.**

The manuscript (§V-C) attributes the 0.05 units/tick decay rate to the auction: *"Because the localized SSI auction distributes tasks based on remaining energy margins..."*. The auction is inactive during the entire period that produces that number.

### F-14 — Algorithm 2 does not describe `auction.py`

| Manuscript Algorithm 2 | `src/coordination/auction.py` |
|---|---|
| `bid_i(τ) ← ω_d‖p_i − p_τ‖₂ + ω_e(1/E_i)` | line 32: `return float(task_reward - dist)` — **energy appears nowhere in the bid** |
| `τ_target ← argmin_τ bid_i(τ)` (minimise cost) | `update_local_winner:48` maximises (`if incoming_bid_value > current_bid_value`) |
| Agent picks one `τ_target`, broadcasts that bid | Agent bids on **every** task (`handle_auction_start` per task) and gossips one at **random** |
| `if bid_i(τ_target) < min(B_nbrs) then T_won ← τ_target` | `resolve_local_winner:63` reads a cached `winner_id`; the agent never compares its own bid |
| — | `active_task_id` is a **single slot** (`agent_core.py:430`); winning a second task silently overwrites the first, abandoning it |

Energy enters only as a feasibility gate (`if dist > agent_energy: return -inf`, line 29), comparing a distance to an energy in inconsistent units.

### F-17 — The metrics logger mutates simulation state and is not reproducible-safe

`src/simulation.py` docstring line 547 calls `_handle_metrics_log` *"(centralized, read-only)"*. It is not:

- **line 574** — `self.interference.psi_max += self.config.interference_growth_rate * self.config.dt * 5`. The metrics handler drives the percolation experiment's physics.
- **lines 612 and 629** — `agent.compute_velocity()` is called for logging. That method has side effects (`agent_core.py:193-195`: `self.total_ticks += 1`, `self.coverage_ticks += 1`), so it inflates the `coverage_completion_rate` metric.
- **Worse:** in the random-walk branch, `compute_velocity` draws from the agent's RNG stream (`agent_core.py:239`: `self._rng.standard_normal(2)`). Calling it from the logger **consumes RNG draws and changes the subsequent trajectory.** In `thermodynamics` and `stability` test modes, a logged run and an unlogged run diverge. This directly contradicts Contribution 4, "Deterministic Reproducibility ... enabling exact replay of stochastic failures."

### F-18 — `coverage_completion_rate` measures nothing about coverage **[VERIFIED]**

`agent_core.py:194`: `if self.global_info_enabled or self.coverage_gain > 0.05: self.coverage_ticks += 1`.

It counts ticks on which the coverage controller was *nominally enabled*. It does not check `self._coverage_enabled`, does not measure area covered, and is unconditionally 100 % in oracle mode. Probe output: **92.07 % for the Unconstrained arm, which runs a random walk.** The name implies validation of Algorithm 3; the number validates nothing.

### F-19 — Percolation "λ̂₂" is neighbour count; Eq. (2)'s proxy is dead code **[VERIFIED]**

`src/simulation.py:584`:

```python
avg_local_lambda = sum(len(a._local_map.get_all_neighbors()) for a in alive_agents) / len(alive_agents)
```

Logged as `avg_local_lambda_proxy`, then plotted by `plot_results.py:68` as `df['avg_local_lambda_proxy'] / 100.0` with the legend `'Scaled Local Proxy ($\hat{\lambda}_2$)'`. **It is mean believed-neighbour count divided by an arbitrary 100.** Data confirms: the column runs 0.0 → 8.055 → 19.43, integer-ish neighbour counts, not eigenvalues.

Meanwhile the actual Eq. (2) quantity, `self._lambda_2_proxy` (`agent_core.py:476`), is `grep`-confirmed **write-only** — assigned at lines 153 and 476, read nowhere in the entire repository.

So the manuscript's §V-B narrative — the "+0.04 average overestimation", the "Disconnected Subgraph Paradox", the "triad of metrics" mitigation — is built on a curve that is not the quantity being discussed, and the quantity being discussed feeds no decision.

The companion `true_lambda_2` column is numerical noise for most of the run (`5.92e-15`, `0.0`, `1.19e-14`), because the graph over 200 positions at r=20 in a 150×150 box is nearly always disconnected → λ₂ = 0.

### F-21 — Eq. (2)'s guard is described backwards

Manuscript §IV-B: *"an explicit epsilon-guard (Var(t−1) ≤ 10⁻⁶) prevents undefined division, immediately returning a **fully-fragmented** proxy scalar"*.

`src/regime/local_proxies.py:82-84`:

```python
if prev_variance <= 1e-6 or current_variance <= 1e-6:
    # If consensus is already reached (or perfectly isolated), return a high/max proxy
    return 1.0
```

It returns a **maximum-connectivity** value, and the code comment says so. The paper's own §V-B ("their local consensus variance drops to zero, falsely generating a **high** spectral proxy") agrees with the code and contradicts §IV-B. The `.tex` is internally inconsistent. The guard also triggers on `current_variance`, which §IV-B does not mention.

### F-20 — Eq. (1) and the packet-drop prose do not describe `packet_drop.py`

Manuscript Eq. (1): `A_ij = 1 if ‖p_i − p_j‖ ≤ R̃_tx and U > p_drop`, with `R̃_tx = R_tx − ω_env`.

`src/communication/packet_drop.py:60-63`:

```python
path_loss_factor = max(0.0, 1.0 - (distance / tx_radius)**2)
p_survive = (1.0 - self._base_p_drop) * (1.0 - psi_value) * path_loss_factor
return bool(self._rng.random() >= p_survive)
```

- There is **no `R̃_tx = R_tx − ω_env`**. Interference multiplies survival probability; it never shortens the radius. The threshold used is `sender_tx_radius = comm_radius · tx_power_scale` (`agent_core.py:322`).
- The drop test is a **three-factor product**, not `U > p_drop`.
- §IV-B calls the distance term *"inverse-square law"*. `1 − (d/R)²` is not `1/d²`; it reaches **exactly zero at `d = R`**, so delivery probability at the nominal radius is 0. The effective communication radius is materially smaller than `R_tx`, which `tests/test_scalability_n100.py` catches (measured drop 0.642 vs nominal 0.2) and which nothing in the paper mentions.

§V-A also states *"The transmission radius (R_tx) was subjected to stochastic Gaussian attenuation"*. `src/simulation.py:55-57` constructs `InterferenceField(mode=FieldMode.CONSTANT, ...)`, and `evaluate()` returns `self.psi_max` unconditionally (`interference_field.py:69-70`). **ψ is a deterministic constant.** `GAUSSIAN_BLOB` exists but `Phase1Simulation` never selects it.

### F-22 — The O(N log N) KD-Tree claim is false for the communication layer **[VERIFIED by grep]**

Manuscript §III-C: *"dynamically updating the RGG adjacency matrix evaluates globally at O(N log N) using K-D Tree spatial partitioning"*.

`RGGBuilder.build_neighbor_lists` (the KD-Tree implementation) is called **only from `tests/test_rgg.py`** — never from the simulation. `CommunicationEngine.process_broadcasts` does a linear scan instead (`comm_engine.py:107, 109`):

```python
distances = np.linalg.norm(all_positions - sender_position, axis=1)
for nbr_id in range(len(all_positions)):
```

That is O(N) per broadcast → **O(N²) per tick**, executed once per agent per tick. `src/simulation.py:331-337` then repeats the same O(N) scan a second time to count neighbours for the energy charge. KD-Trees appear only in the metrics/telemetry observer paths.

### F-23 — The Dwell Time guarantee is not implemented **[VERIFIED by grep]**

Manuscript §III-B: *"the supervisor **strictly enforces** a fixed Dwell Time (τ_d = 5.0 ticks) before permitting subsequent transitions"*, and `src/core/config.py:26-29` claims it *"prevents Zeno behavior and guarantees that transient instability decays before the next switch"* citing Liberzon.

`grep -rn "dwell_time" src/` finds four uses: two constructor pass-throughs, one event-staggering divisor (`simulation.py:201`), and `simulation.py:541` — the **reschedule interval** for `REGIME_UPDATE`. Neither `RegimeClassifier` nor `HybridSupervisor` contains any elapsed-time check or last-transition timestamp. It is a polling period, not a dwell-time constraint. Nothing prevents the regime from flipping on consecutive updates.

### F-24 — Eq. (3) does not match the energy model

Eq. (3) charges `γ_c · R_tx² · N_msg`. `agent_core.py:288`:

```python
cost_per_msg = p_comm * (power_multiplier ** 2)   # power_multiplier = tx_radius / comm_radius_base
```

The implementation is `γ_c·(R_tx/R_base)²·N_msg` — a **normalised** ratio, bounded in `[1, 4]`. As written, Eq. (3) implies a factor of 400 at `R_tx = 20`. The scaling is quadratic in both, but the equation as printed is not the equation implemented.

Related: `τ_max` is discretised **inconsistently** in two places that implement the same bound — `math.ceil` in `gossip_consensus.py:65` versus `int()` (floor) in `agent_core.py:517`. The agent-side bound is therefore looser than the one the consensus function enforces.

`src/core/config.py:88` declares `comm_radius_max: float = 40.0`; `grep` finds no other reference. Dead configuration.

`src/core/event.py:26` declares `AUCTION_TIMEOUT = 5`; no handler is registered and it is never scheduled. Dead enum member.

### F-26 — Junk log files and a multiprocess write race

`KernelLogger._initialize_log` (`kernel_logger.py:23-43`) creates `experiment_{test_mode}.csv` with `self.headers = []` for any unrecognised `test_mode`. `run_monte_carlo_table.py` passes `test_mode="none"` and `"static_bounded"`, producing the empty artifacts `logs/experiment_none.csv` and `logs/experiment_static_bounded.csv` (both present, both 1 empty line).

Worse: all 200 tasks run under `ProcessPoolExecutor` (`run_monte_carlo_table.py:113`) and every one of them opens **the same path** in mode `'w'`. `run_single_sim` never calls `close_loggers()`, so 200 processes truncate and hold handles on two shared files concurrently.

Separately, `KernelLogger.log_snapshot:49` initialises missing keys to `0.0`, not blank. In `stability` mode only one of `max_velocity_cmd_run_A`/`_run_B` is ever written per run (`simulation.py:639-642`), so the other column is silently recorded as a real measurement of zero. The merge in `run_stability_test.py:70-77` happens to pick the right column from each file, so the final CSV is correct — but the per-run files are not, and nothing enforces the merge ordering.

---

## 5. Analytics layer (`src/analytics/`) — not previously reviewed

This layer is GUI-only; **no manuscript number flows through it**. It nonetheless contains the repo's *correct* λ₂ implementation (F-03) and several real defects.

### F-27 — Analyzer caches ignore agent deaths

`spectral_analyzer.py:59-67` and `percolation_analyzer.py:64-68` both key their cache on `hash(frame.adjacency.data.tobytes())` alone, then use `alive_mask = ~frame.drone_failure_flags` inside the computation. **If agents die without the adjacency changing, the cached λ₂ and connectivity ratio are stale.** The cache key must include `drone_failure_flags`.

`percolation_analyzer.py` additionally mutates the cached object in place (line 133 `metrics.d_ratio_dt = ...`, lines 154-158 `metrics.state = ...`) and returns it by reference, so any caller holding an earlier result sees it change underneath them.

`percolation_analyzer.py:127` reads `self._last_time` via `hasattr` — the attribute is never initialised in `__init__`.

### F-28 — "Research Metrics" aggregates a UI-smoothing artifact

`swarm_health.py:117-132` computes a real `score`, then returns `health_score=self._ema_health` (an 0.8/0.2 display smoother) alongside `raw_score=score`. `research_metrics.py:198-200` — the *research* export — consumes `health.health_score`, the smoothed value, for both `sum_health` and `min_health`. The exported research metric is a display artifact; the raw metric is discarded.

### F-29 — `time_to_stability` measures the opposite of its name

`research_metrics.py:146-152`: the `time_to_stability` field is assigned inside the **`else`** branch (the not-stable branch), and only after a stable period has started. It records the moment the swarm *left* stability, not the time taken to reach it.

`research_metrics.py:191` carries its own uncertainty in a comment: `m.anomaly.total_anomalies += anom.anomaly_count # Actually we want unique anomalies? "total" could be integral` — it accumulates a per-frame instantaneous count as if it were a cumulative total.

### F-30 — `swarm_health.py` sign conventions are guessed

`swarm_health.py:100`: `if eng.cascade_margin > -0.05: score -= 0.05`, with the comment `# Cascade happens when intensity exceeds threshold, margin > 0.` The threshold `-0.05` does not correspond to the stated condition. Similar unresolved sign reasoning at lines 86-89.

`energy_cascade_analyzer.py:124-125` captures `_initial_mean_energy` from the **first frame the analyzer happens to see**, not from the configured initial energy. Every `normalized_energy` value is relative to whenever the analyzer was attached.

---

## 6. Replay layer (`src/replay/`) — not previously reviewed

### F-25 — `ReplayLoader` raises on every load **[VERIFIED]**

```
$ python -c "from src.replay.replay_loader import ReplayLoader; ReplayLoader('outputs/run_20260412_102518_baseline_s42').load()"
EXCEPTION: TypeError TelemetryFrame.__init__() missing 1 required positional argument: 'agent_states'
```

`TelemetryFrame` requires `agent_states` (`telemetry_frame.py:79`, no default). None of the three loader paths supply it — `_load_json:117-133`, `_load_csv:153-166`, `_row_to_frame:184-197`. **Replay is completely non-functional**, and `tests/test_replay.py` fails with the same error. Contribution 4's "exact replay of stochastic failures" cannot be exercised at all.

### F-31 — Replay fabricates a contradictory graph

Even once F-25 is fixed, all three loader paths set:

```python
adjacency=np.zeros((n, n), dtype=np.uint8),          # fully DISCONNECTED
connected_components=[list(range(n))],               # fully CONNECTED
```

These two fields contradict each other. Downstream, `SpectralAnalyzer` reads the zero adjacency → every degree is 0 → returns `0.0` (`spectral_analyzer.py:126-127`); `PercolationAnalyzer` sees N singleton components → ratio ≈ 1/N → immediately emits `percolation_collapse_detected`. **Every replayed run will report total network collapse regardless of what actually happened**, while the correct `spectral_gap` scalar sits unused in the same frame.

The class docstring asserts *"Strictly read-only: no metric recomputation, no frame mutation"*, and `telemetry_frame.py:42-43` documents `adjacency` as *"Built from the RGG communication graph at extraction time"* — untrue after a round-trip.

---

## 7. Telemetry layer (`src/telemetry/`) — not previously reviewed

### F-32 — `TelemetryEmitter` reproduces the dead-agent λ₂ bug

`telemetry_emitter.py:77-88` builds the adjacency from **all** positions with no alive filtering, then computes `spectral_gap` from it. So `frame.spectral_gap` — the value exported to disk and reloaded on replay — has the same defect as F-03. (`SpectralAnalyzer` recomputes it correctly, giving two disagreeing λ₂ values in one system.) It also uses the static `cfg.comm_radius`, ignoring `tx_power_scale`.

### F-33 — `TelemetryFrame.empty()` marks every drone dead

`telemetry_frame.py:98`: `drone_failure_flags=np.ones(n, dtype=bool)`. An "empty" frame declares the entire swarm dead. The same fail-dead default recurs in `exporter.py:101` and `replay_loader.py:149`, so any missing or short column silently reads as a dead swarm rather than raising.

### F-34 — Export silently truncates and loses the config

`exporter.py:128`: `deque(maxlen=max_buffer)` with `max_buffer=10000`. Beyond 10 000 frames the **oldest** frames are silently discarded. `metadata["total_frames"]` (line 201) then records the truncated count, and `duration` (line 199) is computed from the truncated window — so `ReplayLoader._validate` compares against the already-wrong number and never warns.

`exporter.py:78`: `d.pop("regime", None)` strips `RegimeConfig` from `config_snapshot`. The exported metadata cannot reconstruct the run.

---

## 8. Scenario layer (`src/scenario/`) — not previously reviewed

### F-35 — Every scenario load crashes **[VERIFIED]**

```
$ python -c "... ScenarioConfig().to_sim_config(SimConfig()); Phase1Simulation(cfg)"
to_sim_config OK; type(cfg.regime) = <class 'dict'>
EXCEPTION: AttributeError 'dict' object has no attribute 'window_size'
```

`scenario_model.py:94` uses `dataclasses.asdict(base_cfg)`, which recursively converts the nested `RegimeConfig` into a plain `dict`. `SimConfig(**d)` then stores a `dict` in `.regime`, and `AgentCore.__init__` dereferences `reg_cfg.window_size` (`agent_core.py:109`) → `AttributeError`. The GUI's "run custom scenario" path (`main_window.py:358`) is dead.

### F-36 — `ScenarioConfig.tasks` is inert

`TaskParams.count` and `.distribution` (`scenario_model.py:35-38`) are serialised by `to_dict` and parsed by `from_dict`, but `to_sim_config` (lines 91-110) never reads them. Setting a task count in the scenario UI does nothing.

---

## 9. GUI layer (`src/gui/`) — not previously reviewed

### F-37 — `reset_analyzers()` permanently severs event wiring

`main_window.py:394-400` **replaces** all six analyzer objects:

```python
self.percolation_analyzer = PercolationAnalyzer()
self.spectral_analyzer = SpectralAnalyzer()
...
```

All the Qt signal connections were bound in `_connect_signals` (lines 369-389) to the *previous* objects. Nothing re-connects them. After the first replay reset, `percolation_collapse_detected`, `spectral_instability_detected`, `energy_cascade_detected`, all consensus signals, and the swarm-map visual triggers are **permanently dead** — the event log and timeline stay empty for the rest of the session while the panels continue to update, so the failure is invisible.

Line 408 also reaches into a private attribute of another object (`self.energy_heatmap_mapper._last_hash = -1`) rather than calling a reset method.

### F-38 — Status bar reports the smoothed health score

`main_window.py:531` displays `health_metrics.health_score` — the EMA (F-28) — not `raw_score`. The displayed value lags true state by ~5 frames, which matters for a "COLLAPSE" indicator.

---

## 10. `draft 11.pdf` vs `manuscript/final_manuscript.tex` — silent drift **[VERIFIED]**

The PDF does **not** match what the `.tex` would currently produce. The `.tex` is newer. Differences found by extracting and diffing the PDF text:

| Item | `draft 11.pdf` | `final_manuscript.tex` |
|---|---|---|
| **Authors** | 3 (Malge, Hegde, Koushik) | **4** — adds "Ms. Shruthi", Dept. of Information Science Engineering, ORCID `0000-0001-9876-5432` |
| **The false-audit sentence** | **absent** (`grep "audited"` → 0 hits, `"No such issues"` → 0 hits) | present at `.tex` line 311 |
| **§V-A explanation of the result** | *"By physically halting, the Proposed Framework eliminates its kinetic energy consumption, dropping its total drain to the baseline transmission cost (0.05 units/tick) **to safely coast until the network reconnects**, mirroring the mathematically perfect Oracle."* | replaced by *"This result was initially unexpected. We therefore audited the implementation for hidden state propagation, hard-coded parameters, and Monte Carlo aggregation artifacts. **No such issues were identified.** The observed convergence arises because, under the simulator's energy model, mobility dominates total energy expenditure..."* |
| **Figure numbering** | sequence diagram = Fig. 2, spectral = Fig. 3 | architecture figure inserted earlier; `\ref` numbering shifted |

**Treat `final_manuscript.tex` as the source of truth going forward.** Two flags:

1. **The false claim was introduced *after* the last compile.** The sentence "No such issues were identified" was added in the same edit that *removed* a different false claim. This is the same failure pattern the project has already hit twice.
2. **The PDF's own replaced text is also false.** *"safely coast until the network reconnects"* — the network never reconnects; the 3 survivors remain `FRAGMENTED` for the final ~1800 ticks (F-04). Neither version of the sentence is defensible.
3. The ORCID `0000-0001-9876-5432` follows an obvious placeholder pattern (`9876-5432`). Verify before submission.

---

## 11. Manuscript claims with no corresponding implementation

Claims in the `.tex` that map to **no code path anywhere in `src/`**:

| Location | Claim | Reality |
|---|---|---|
| §III-B | "fixed Dwell Time (τ_d = 5.0 ticks)" strictly enforced | No enforcement exists — F-23 |
| §III-C | RGG adjacency at O(N log N) via K-D Tree | Linear scan, O(N²)/tick — F-22 |
| §IV-B / Eq. (2) | Empirical local mixing proxy λ̂₂ | Computed, never read; the plotted curve is neighbour count — F-19 |
| §V-A | "stochastic Gaussian attenuation" of R_tx | `FieldMode.CONSTANT`, deterministic — F-20 |
| §V-A / Eq. (1) | `R̃_tx = R_tx − ω_env` | Not implemented — F-20 |
| §V-A | "Centralized Oracle (a theoretically perfect, global-information system)" | Row 2 is `static_bounded`, bit-identical to row 3 — F-02. A real oracle (`global_info_enabled`) now exists but was **not** used for Table III |
| §V-A | "The oracle exhibits a significantly tighter confidence interval (±40 ticks)" | Table III itself prints ±2 for that row. **The prose contradicts the table on the same page.** |
| §V-A | "over 15 million discrete events" | **[VERIFIED]** a single N=100/2000-tick run dispatches **111,727–136,935** events. Even 200 runs total ≈ 27 M. Per-experiment the figure is ~137 k |
| §V-A | "all quantitative experiments were averaged over 50 independent Monte Carlo runs ... 95 % CI" | **False for Figures 1-3.** Only `run_monte_carlo_table.py` is a Monte Carlo study. `run_percolation.py` = 1 run, N=**200**, seed 42. `run_energy_cascade.py` = 1 run, N=**50**, seed 42. `run_stability_test.py` = 1 run, N=**30**, seed 123 |
| §V-A | "N = 100 autonomous agents" | True only for Table III; Figures 1-3 use N = 30, 200, 50 |
| §V-B | "+0.04 overestimation across the 50 Monte Carlo runs (as visualized in Figure 2)" | Figure 2 is one N=200 run; the plotted quantity is not λ̂₂ — F-19 |
| §V-C | Auction "distributes tasks based on remaining energy margins" | Energy is not in the bid — F-14; the auction is inactive for 91 % of the run — F-13 |
| §V-C | "node attrition rates approaching 50 %" | `logs/experiment_2_thermodynamics.csv` ends at **4/50** framework and **0/50** baseline — 92 % and 100 % attrition |
| §V-D + Fig. 1 caption | "Kinematic variance (σ²)"; "bounds parameter mutations to Δμ < 0.04" | The plotted series is **max velocity magnitude**, not variance. The y-axis label itself conflates them: "Maximum Kinematic Variance / Velocity (v)". Mean Δμ **[VERIFIED]** = 54.243/3527 = **0.0154** ✓, but **max** Δμ = **0.0761** ✗ — "strictly bounds ... to Δμ < 0.04" is false for the maximum |
| §V-D | "An unconstrained optimizer exhibits severe oscillations with parameter variations exceeding ±0.3" | No such mechanism exists; all supervisor proposals are bounded constants in [0, 2]. Run A's plotted trace is flat at 1.0 |
| Fig. 1 "Local Stability Boundary" at y=2.0 | implied stability result | `v_max = 2.0`; `apply_movement:274-275` hard-clamps to it. The line is a physical constant, not a stability boundary, and neither trace can cross it |
| Abstract | "sustained network connectivity ... under 50 % node attrition" | Connectivity is measured over corpses — F-03 |
| `README.md:13` | "**mathematically guaranteeing** they do not induce kinematic oscillations" | No proof exists; the `.tex` correctly says "heuristic relaxations" and disclaims mathematical novelty. The README overclaims relative to the paper |

---

## 12. Summary of what must change before this paper is submittable

1. **Figures must be regenerated and re-embedded.** The fabricated Figure 4 is still in the PDF pipeline (F-01).
2. **Table III must be rebuilt.** Row 2 is not an oracle and is not a distinct experiment (F-02); the λ₂ column is invalid (F-03); the survival column is a censoring artifact describing 3 drones (F-09).
3. **The causal explanation in §V-A is wrong** and must be replaced with the real mechanism, which is less favourable: a coverage-controller fixed point versus a random-walk placeholder (F-04).
4. **The false-audit sentence must be deleted** (F-07). It is the single most damaging sentence in the paper.
5. **§V-A's methodology paragraph is false for three of four experiments** and must be corrected per-experiment (§11).
6. **Algorithm 1 must gain its Step 1**, since Step 1 is what produces the result and Step 2 is inert (F-11, F-10).
7. **Algorithm 2 must be rewritten to match `auction.py`**, or `auction.py` rewritten to match Algorithm 2 (F-14).
8. Equations (1), (2), (3) each need correction against their implementations (F-20, F-21, F-24).
9. The replay and scenario subsystems are non-functional (F-25, F-35) and the test suite is red (F-08); neither blocks the paper, but both should be fixed before the repository is cited as open-source evidence in the abstract's footnote.

Ordered, executable remediation is in `fixes_phases.md`. The Rust port is deliberately deferred to `rust_conversion_plan.md` and must not begin until `fixes_phases.md` is complete and verified.
