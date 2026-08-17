# Remediation Plan — Phased

**Companion to:** `audit_findings.md` (finding IDs `F-nn` refer to it)
**Rule for every item:** a fix is not done until its **Verify** step has been *executed* and its actual output pasted into this file under `Evidence`. "Implemented" without pasted evidence counts as not done.
**Rule for every result-changing item:** the corresponding `[MS-nn]` manuscript task is a separate checklist line. Code and paper are checked off independently, so neither can be silently forgotten.
**Rule on outcomes:** several fixes below will make the reported results *worse*. Report them as they come out. Do not re-tune parameters, re-pick seeds, or re-scope experiments to recover a favourable number.

---

## Phase 0 — Stop shipping fabricated material (BLOCKING)

Nothing else in this plan matters until Phase 0 is complete. These items are about the paper containing things that are not true right now.

### 0.1 — Delete the false audit sentence `[F-07]`

**File:** `manuscript/final_manuscript.tex`, line 311.
**Remove verbatim:**
> "This result was initially unexpected. We therefore audited the implementation for hidden state propagation, hard-coded parameters, and Monte Carlo aggregation artifacts. No such issues were identified."

Do **not** replace it with a softened audit claim. The honest replacement is written in 0.4 once the real mechanism is confirmed.

**Verify:** `grep -c "No such issues were identified" manuscript/final_manuscript.tex` → must print `0`.

**Evidence:** ✅ DONE
```
--- grep -c 'No such issues were identified' ---
0
--- grep -c 'audited' ---
0
--- grep -c 'initially unexpected' ---
0
```

### 0.2 — Regenerate every figure and re-embed it `[F-01]`

The plotting code is already fixed in the working tree; the paper still carries the pre-fix PNGs.

1. Regenerate the source data: `python experiments/run_percolation.py && python experiments/run_energy_cascade.py && python experiments/run_stability_test.py`
2. Regenerate plots: `python experiments/plot_results.py`
3. Copy into the manuscript with the **documented** mapping:
   - `plots/kinematic_stability.png` → `manuscript/fig1.png`
   - `plots/spectral_stability.png` → `manuscript/fig2.png`
   - `plots/thermodynamic_decay.png` → `manuscript/fig3.png`
4. Delete the orphan `manuscript/fig6_ablation.png` (referenced nowhere).
5. Add `experiments/publish_figures.py` that performs step 3 and prints each source→dest md5 pair, so this can never drift silently again.

**Verify:** run
```bash
for p in kinematic_stability:fig1 spectral_stability:fig2 thermodynamic_decay:fig3; do
  s=${p%%:*}; d=${p##*:}
  echo "$d $(md5sum plots/$s.png | cut -d' ' -f1) $(md5sum manuscript/$d.png | cut -d' ' -f1)"
done
```
Each line's two hashes must be **equal**. (They are currently all unequal — see `audit_findings.md` F-01.)

**Evidence:** ✅ DONE — implemented as `experiments/publish_figures.py` (copies + verifies; `--check` exits non-zero on drift).

Datasets regenerated first:
```
Starting Percolation Experiment (N=200)...
Done. CSV generated in logs/
Starting Thermodynamics Experiment...
  -> Running Baseline (Theta Safe OFF)...
  -> Running Proposed (Theta Safe ON)...
Final merged CSV generated: logs\experiment_2_thermodynamics.csv
Starting Stability Test - Run A (Theta Safe OFF)...
Starting Stability Test - Run B (Theta Safe ON)...
Final merged CSV generated: logs\experiment_3_stability.csv

Generated kinematic_stability.png
Generated spectral_stability.png
Generated thermodynamic_decay.png
```

Before publish (drift confirmed present):
```
figure                     source md5                         manuscript md5                     status
------------------------------------------------------------------------------------------------------------
fig1.png                   2c0a67c5ed5b31a513eddea2ce0e1002   81216ba4bb12d8e2a89d2dab15caf13e   DRIFT
fig2.png                   73aa17d930f781140b363edc9e2a695b   3ddc72f3d040fa64e66b5270178e1c5c   DRIFT
fig3.png                   43e748733ba212b4f33d79db8e65997a   21344ea8fd9750d40fe778e94dfb9848   DRIFT
------------------------------------------------------------------------------------------------------------
FAIL: 3 figure(s) out of sync. The manuscript is not showing current data.
```

After publish:
```
figure                     source md5                         manuscript md5                     status
------------------------------------------------------------------------------------------------------------
fig1.png                   2c0a67c5ed5b31a513eddea2ce0e1002   2c0a67c5ed5b31a513eddea2ce0e1002   MATCH
fig2.png                   73aa17d930f781140b363edc9e2a695b   73aa17d930f781140b363edc9e2a695b   MATCH
fig3.png                   43e748733ba212b4f33d79db8e65997a   43e748733ba212b4f33d79db8e65997a   MATCH
------------------------------------------------------------------------------------------------------------
OK: every manuscript figure is byte-identical to its regenerated source.
exit=0
```

Fabricated figure confirmed displaced:
```
--- fabricated fig3 hash (pre-fix, from git HEAD) ---
21344ea8fd9750d40fe778e94dfb9848
--- manuscript/fig3.png now ---
43e748733ba212b4f33d79db8e65997a
--- orphan figure removal ---
rm 'manuscript/fig6_ablation.png'
0
```

**Note:** these figures will need regenerating again after Phase 1 (1.1 and 1.4 change the plotted data). Phase 0's purpose here is only to stop shipping the fabricated Figure 4.

### 0.3 — Remove the fabricated-survival script `[F-06]`

**File:** `experiments/run_mc_fast.py`, lines 24 and 26.

`surv = cfg.max_time if alive > 0 else cfg.max_time * 0.7` invents its output. Either delete the file (preferred — `run_monte_carlo_table.py` supersedes it), or rewrite `worker()` to return `sim.summary()["time_of_death"]` with an explicit `censored` flag, exactly as `run_monte_carlo_table.py:29-33` now does.

**Verify:** `grep -rn "max_time \* 0.7" experiments/` → must print nothing.

**Evidence:** ✅ DONE — file deleted (`git rm -f experiments/run_mc_fast.py`); no references existed anywhere.
```
--- does anything import or reference run_mc_fast? ---
(no output above = no references)

--- the two fabricating lines, before deletion ---
    decay = (init_energy - rem_energy) / (cfg.num_agents * cfg.max_time)
    surv = cfg.max_time if alive > 0 else cfg.max_time * 0.7

rm 'experiments/run_mc_fast.py'
--- VERIFY: grep -rn 'max_time \* 0.7' experiments/ ---
(no output above = fabrication removed)
--- ls experiments/run_mc_fast.py ---
ls: cannot access 'experiments/run_mc_fast.py': No such file or directory
```

### 0.4 — `[MS-01]` Replace §V-A's causal explanation with the real one `[F-04]`

**File:** `manuscript/final_manuscript.tex`, §V-A, the paragraph beginning "As demonstrated in Table \ref{tab:baselines}".

**Delete** these claims, all of which the audit disproved by direct measurement:
- "the Proposed Framework's $\Theta_{safe}$ bounded heuristic clamps the drones' velocity to zero" — survivors have `velocity_scale = 1.000`.
- "the Unconstrained swarm engages in rapid, erratic 'kinematic oscillation' based on blowing-up mathematical errors" — no such mechanism exists.
- "mobility dominates total energy expenditure. Once the proposed controller clamps velocity during fragmentation..." — velocity is never clamped.

**Replace with** the measured mechanism (F-04): under `\theta_{safe}`, the box clamp raises `coverage_gain` from the proposed 0.0 to its lower bound 0.5, which keeps agents in the Voronoi-coverage branch; an isolated agent's local Voronoi centroid is its own position, so its commanded velocity is identically zero and it stops moving. Without `\theta_{safe}`, `coverage_gain` stays 0.0 and the agent falls through to a **random-walk fallback**, which continues to expend movement energy.

State plainly that the fallback is a placeholder controller, so the comparison is a lower bound on baseline behaviour rather than a competitive baseline. This is less favourable than the current text. That is the correct outcome.

**Verify:** re-read the edited paragraph against `audit_findings.md` F-04; every sentence must name a mechanism that exists in `src/`.

**Evidence:** ✅ DONE

Disproved claims removed:
```
--- disproved claims must all be 0 ---
clamps the drones' velocity to zero            0
kinematic oscillation                          1
blowing-up mathematical errors                 0
thermodynamic braking                          0
mobility dominates total energy expenditure    0
```
The single remaining `kinematic oscillation` hit is at line 60 (Introduction), a general statement about the literature, not a measured claim about this framework's baseline. Flagged for Phase 1 review — it motivates a failure mode this framework does not actually exhibit.

Replacement text as written to the file:
```
As demonstrated in Table \ref{tab:baselines}, the Proposed Framework perfectly matches the Centralized Oracle's maximum swarm survival ($1999$ ticks) and baseline energy decay rate ($0.05$ units/tick), vastly outperforming the Unconstrained RGG which crashes at $527$ ticks due to an exponential decay rate of $0.20$. The Proposed Framework does not achieve this by possessing global information. Tracing the executed control path shows that the difference arises from which velocity law each condition ends up running once the network fragments, and we report that mechanism directly rather than attributing the result to a stability property it does not exhibit.

Under fragmentation the Hybrid Supervisor proposes $\theta_{coverage} = 0$, an instruction to stop dispersing. With $\Theta_{safe}$ active, the static box constraint raises this proposal to the lower bound of its admissible interval, $\theta_{coverage} = 0.5$, so the agent remains in the localized Voronoi coverage law of Algorithm 3. For an agent whose belief buffer holds no live neighbours, the local Voronoi cell is the full communication disc and its centroid coincides with the agent's own position; the commanded velocity $v_i = k(c_{local} - p_i)$ is therefore identically zero. The agent halts, and its drain falls to the fixed idle and communication baseline. We note that the agent's velocity scale itself is never clamped: the halt is a fixed point of the coverage law under isolation, not an explicit velocity constraint.

With $\Theta_{safe}$ disabled, $\theta_{coverage} = 0$ passes through unmodified, the coverage law is bypassed, and the agent falls back to the framework's undirected random-walk controller, which continues to expend movement energy for the remainder of the run. This fallback is a baseline placeholder rather than a tuned unconstrained optimizer, so the Unconstrained row should be read as a lower bound on achievable baseline behaviour and not as a competitive comparison. The survival gap in Table \ref{tab:baselines} therefore measures the cost of continued undirected motion under fragmentation, and does not by itself demonstrate that the bisection projector confers a stability advantage.
```

**Scope note:** the paragraph's opening sentence still carries the numeric/label claims covered by `[MS-04]`, `[MS-05]`, `[MS-06]` ("perfectly matches the Centralized Oracle", `1999`, `527`). Those are Phase 1 items and were deliberately left untouched.

### 0.5 — Verify the True Oracle is wired end-to-end `[gates 1.2 Option A]`

**Result: all four specified checks PASS. A fifth, unrequested property fails — see the verdict below.**

**Evidence:** ✅ checks 1-4 pass

```
========================================================================
CHECK 1 -- does global_info_enabled reach AgentCore at runtime?
========================================================================
config.global_info_enabled      = True
agents[0].global_info_enabled   = True
all 100 agents have it True     = True
control (flag False) agents[0]  = False

========================================================================
CHECK 2 -- which branch of compute_velocity actually executes?
========================================================================
ORACLE MODE (global_info_enabled=True) branch counts:
   ORACLE centroid (line 205)       2103
   local Voronoi (line 216)           65
   active_task (line 197)              2
CONTROL  (global_info_enabled=False) branch counts:
   local Voronoi (line 216)        12929
   active_task (line 197)            314

oracle-branch executions in oracle mode = 2103   NON-ZERO: True
oracle-branch executions in control    = 0

========================================================================
CHECK 3 -- does the oracle path touch LocalMap / gossip / RGG / drop sampler?
========================================================================
                                  ORACLE      CONTROL
compute_gossip_update() calls     0           13143        (gossip_epsilon consumer)
LocalMap neighbour beliefs total  0           1957       
LocalMap active_auctions total    0           275        
comm_engine.total_sent            0           100363       (PacketDropSampler consumer)
comm_engine.total_dropped         0           68780      
comm_engine.total_delivered       0           31583      
auction_results (wins)            2           26         
RGGBuilder.build_neighbor_lists   : not called in EITHER mode (see audit F-22);
                                    comm_engine.py:107 uses np.linalg.norm scan

========================================================================
CHECK 4 -- same seed, oracle True vs False: are final positions identical?
========================================================================
seed 1000: positions identical=False  energies identical=False  max|dpos|=71.2200  alive 0 vs 4
seed 1001: positions identical=False  energies identical=False  max|dpos|=73.6580  alive 0 vs 4
seed 1002: positions identical=False  energies identical=False  max|dpos|=76.3165  alive 0 vs 3

========================================================================
SUPPLEMENTARY -- what regime does the oracle swarm believe it is in?
========================================================================
oracle  regimes: {}
control regimes: {'FRAGMENTED': 4}
oracle  strategies: {}
```

**Check 3, line references confirming isolation of the oracle path:**

| Subsystem | Bypass site | Runtime proof |
|---|---|---|
| `PacketDropSampler` | `simulation.py:286-301` returns before `comm_engine.process_broadcasts`, the sole caller of `should_drop` (`comm_engine.py:120`) | `total_sent = 0` |
| `LocalMap` | never populated: `receive_message`/`process_inbox` run only from `_handle_msg_deliver` (`simulation.py:369-378`), which is never scheduled because no `MSG_DELIVER` event is created | `0` neighbour beliefs, `0` active auctions |
| `gossip_epsilon` | `simulation.py:406-411` assigns the true global mean directly instead of calling `handle_consensus_update`, the sole caller of `compute_gossip_update` (`agent_core.py:381`) | `0` calls |
| `RGGBuilder` | not used in either mode (audit F-22) | — |

One caveat on strict isolation: `simulation.py:284` still calls `agent.prepare_broadcast()` in oracle mode, which reads `self._local_map.active_auctions` and may draw from `self._rng` (`agent_core.py:304, 312`). The map is always empty so the read is inert, but the RNG draw is not — it perturbs the oracle's agent streams. Worth cleaning up in Phase 1 alongside 1.5.

**⚠️ VERDICT — 1.2 Option A cannot be relied on as written.**

The oracle is genuinely wired and genuinely global. It is **not an upper bound**, because it is charged for an all-to-all broadcast every tick:

```
[oracle ] consume_comm_energy calls=  2157  mean recipients/call=  93.19  total recipient-charges=  201006  ToD=86.0  alive=0
[control] consume_comm_energy calls= 11213  mean recipients/call=   8.95  total recipient-charges=  100363  ToD=None  alive=4
```

`simulation.py:289-290` charges the oracle `alive_count - 1 ≈ 99` recipients per broadcast, versus ~9 for the decentralized path (neighbours inside `tx_radius`, `simulation.py:331-337`) — a **10.4× per-transmission energy penalty**. The oracle suffers total swarm death at **t=86**; the decentralized framework is still alive at t=400.

Putting this row in Table III as "a theoretically perfect, global-information system serving as an upper bound" would produce the claim that the decentralized framework outlives a perfect oracle by more than 4×. That is the same shape of too-good-to-be-true result that started this audit, and it would be an artifact of the energy accounting, not a finding.

**Options for Phase 1, needs a decision before 1.2 proceeds:**
- **(i)** Charge the oracle the same per-neighbour cost as everyone else (global *information* is free, but *transmission* is billed identically). Makes it a true upper bound on coordination quality. Requires justifying why a centralized system gets free global state.
- **(ii)** Keep the all-to-all charge and relabel the row honestly — not "upper bound" but "centralized coordination with realistic all-to-all communication cost". The result then becomes a legitimate finding: centralization does not pay for itself under this energy model. This is defensible and arguably more interesting.
- **(iii)** Drop the oracle row entirely; report Unconstrained vs Proposed only, and delete the oracle language from §V-A.

I have not chosen. This is a modelling decision, not a bug fix.

---

## Phase 0.6 — Oracle accounting decision + algorithmic performance work

Authorised 2026-08-16. Ordered before the Phase 1 metric fixes so every subsequent suite re-run is cheap.

### 0.6.1 — Oracle RNG determinism leak `[author-flagged]`

`simulation.py:284` called `prepare_broadcast()` in oracle mode and discarded the message, but that method draws from the agent RNG (`agent_core.py:304, 312`). Fixed by evaluating the oracle branch **before** `prepare_broadcast()`.

**Evidence:** ✅ DONE — and the fix is measurably inert on these seeds, which is the desired finding: prior oracle measurements were **not** contaminated.
```
Oracle [all_to_all    ] seed 1000: events=  35782 ToD=86.0 t50=21.0
Oracle [per_neighbour ] seed 1000: events= 170126 ToD=1417.0 t50=202.0

Pre-RNG-leak-fix reference for oracle[all_to_all] seed 1000: events=35782, ToD=86.0
```
Identical to the pre-fix reference. The draws were rare because in oracle mode `active_auctions` is always empty and `broadcast_rate` stays 1.0, so neither RNG path in `prepare_broadcast` was reached. Correctness hygiene regardless.

### 0.6.2 — Oracle billing model: decision + sensitivity study

**Decision (author, 2026-08-16): `all_to_all` is primary.** Global awareness implies global bandwidth; billing the oracle per-neighbour to manufacture an upper bound would idealise away the exact cost this paper exists to study. `per_neighbour` is retained as a measured sensitivity variant, not as the headline.

Implemented as `SimConfig.oracle_comm_billing` ∈ `{"all_to_all", "per_neighbour"}` (`config.py`), consumed by `Phase1Simulation._oracle_recipient_count`. Unknown values raise rather than silently defaulting. New runner: `experiments/run_oracle_sensitivity.py`.

**Evidence:** ✅ DONE
```
Oracle communication-accounting sensitivity -- 10 seeds, N=100, max_time=2000

arm                            lambda_2     decay/tick        survival   cens           t50  alive_end
------------------------------------------------------------------------------------------------------
Unconstrained              0.068+/-0.058  0.2567+/-0.0442      427+/-88      0/10    102+/-4            0.0
Proposed                   0.079+/-0.059  0.0498+/-0.0001     2000+/-0      10/10    102+/-4            2.5
Oracle [all_to_all]        0.497+/-0.033  2.5761+/-0.4772       43+/-10      0/10     21+/-0            0.0
Oracle [per_neighbour]     0.985+/-0.133  0.0703+/-0.0065     1454+/-137     0/10    199+/-5            0.0
```

**Reading, reported as measured:**
- **Neither oracle configuration is a clean upper bound on total survival.** `all_to_all` is the *shortest-lived arm in the study* (43 ticks). `per_neighbour` survives 1454 versus Proposed's censored 2000.
- **`per_neighbour` IS a meaningful upper bound on the metric that actually discriminates.** Time to 50% attrition: **199 vs 102** for both decentralized arms — roughly double, and the only arm that moves this number at all. This is the cost decomposition the author anticipated: centralized coordination is worth ~2× swarm half-life, and under `all_to_all` accounting that entire benefit is consumed by the bandwidth needed to obtain it.
- Proposed's `2000 ± 0` is **10/10 censored** — a lower bound describing ~2.5 stationary survivors, not a measurement.
- ⚠️ **λ₂ figures in this table are pre-1.1** and computed over dead agents (F-03). Do not cite them. Re-run after 1.1.

**⚠️ OPEN — Table III is in an inconsistent intermediate state.** The oracle row is now correctly *labelled* but still carries the stale `static_bounded` numbers (`0.29 / 0.05 / 1999`), which are not the oracle's. A `PENDING REGENERATION` warning block has been placed above the table in the `.tex`. **Must be resolved by 1.2/1.3 before anything compiles for submission.**

### 0.6.3 — Algorithmic performance fixes

Three fixes, from the R0 profile (full results in `rust_conversion_plan.md`):

1. **Eliminated the duplicate O(N) scan.** `process_broadcasts` now returns `(delivered, in_range)`; `simulation.py:331-337`'s Python genexpr — 3.3M scalar `np.linalg.norm` calls — deleted. The counted condition is provably identical: both count `nbr != sender AND alive AND dist <= tx_radius`, the same predicate that gates `total_sent`.
2. **Live position matrix.** `self._positions`, synced in `_handle_kinematic_update` immediately after `apply_movement` (its sole mutator). Replaces rebuilding an N-element array of copies on every broadcast.
3. **BLAS pinned to 1 thread** in `run_monte_carlo_table.py`, set before the `numpy` import so spawned workers inherit it.

**Evidence — behaviour preservation (mandatory for a refactor):** ✅
```
REGRESSION CHECK -- decentralized arms must be UNCHANGED by the refactor
arm                events      ref    ok      ToD     ref    t50    ref  wall
Unconstrained      111727   111727  True    434.0   434.0    105    105   11.6s
Static-Epsilon     136935   136935  True     None    None    106    106   16.7s
Proposed           136935   136935  True     None    None    106    106   17.1s

ALL DECENTRALIZED ARMS BIT-MATCH REFERENCE: True
```
```
Unconstrained seed 1000 mean lambda_2 = 0.2923   (pre-change reference: 0.2923)
   -> matches reference: True
```
Event counts are a strong identity signal: they depend on every scheduling decision, every death, and every message delivery, so an exact match across three arms plus an exact λ₂ match rules out trajectory drift.

**Evidence — speed:** ✅
```
     N    wall_s     events   us/event      |  BEFORE wall_s
    50      6.26      44870     139.54      |    7.11
   100     13.97      89097     156.81      |   16.55
   200     35.24     194304     181.35      |   51.98
   400     72.32     384592     188.03      |  148.57

empirical exponent between N=200 and N=400:  wall ~ N^1.04     (BEFORE: N^1.52)
```

⚠️ **RETRACTED AS A FINDING (author-flagged, 2026-08-16).** I applied the truncation argument to the pre-fix N^1.52 reading and then failed to apply the identical argument to my own post-fix number. **N^1.04 is unsound for exactly the same reason**: same runs, same early swarm death (`alive_end` 4-5 regardless of N), same truncation of the quadratic term. "Scaling is now near-linear" is **not** a defensible claim.

The wall-clock *improvements* above (2.05× at N=400, 2.17× on the profiled run, 1.74× on the suite) remain valid — those are direct before/after measurements on identical workloads. Only the **exponent** is unsound. Tracked as `[MS-29]`.
```
AFTER all fixes: 16 tasks wall=39.1s
  BEFORE (unpinned BLAS) = 68.1s ; BEFORE (pinned BLAS) = 61.1s
  extrapolated 200-task suite = 8.2 min   (BEFORE: 14.2 / 12.7 min)
  end-to-end speedup on the suite = 1.74x
```
Profiled single run (N=200, max_time=300): **117.6 s → 54.1 s (2.17×)**. `_handle_msg_transmit` cumulative **64.6 s → 8.9 s (7.3×)**. `np.linalg.norm` calls **3,305,898 → 587,332**. Scaling **N^1.52 → N^1.04**; the N=400 case improved 2.05×.

**Not done, and why:** the KD-tree wiring (`MS-20`) is deferred — it replaces a 3.7 % cost and carries the fixed-vs-variable `tx_radius` hazard documented in R0. Sparse `eigsh` is deferred to 1.1, where the driver is correctness rather than the 2.2 % it would save.

---

## Phase 1 — Make the reported metrics measure what they claim

These are the fixes that change Table III's numbers. Do them **before** re-running the Monte Carlo study, so the study is run once against correct instrumentation.

> **Standing direction from the author (recorded 2026-08-16), binding on all of Phase 1:**
>
> 1. **A smaller true result is the goal.** If λ₂ collapses across all arms, **drop the column** — do not keep a metric that no longer discriminates. If the survival advantage does not survive honest censoring reporting, **report that it does not**. No claim survives Phase 1 that the corrected data does not support. Do not tune, re-seed, or re-scope to recover a favourable number.
>
> 2. **F-04 is a positive finding, not only a retraction.** A localized Voronoi coverage law that has an unintended *stationary fixed point* for isolated agents under fragmentation — where the commanded velocity is identically zero because an isolated agent's cell centroid is its own position — is a real, reportable property of the algorithm. It produces an emergent energy-conserving halt that was never designed in, and it is the actual reason the framework outlives its baseline. **§V-A should be framed around that discovery**, with the correction of the "velocity clamping" language as a consequence of the finding rather than its subject. The honest version is more interesting than the claim it replaces: the mechanism is a property of Algorithm 3, not of the Θ_safe projector the paper currently credits.

### 1.1 — Exclude dead agents from connectivity metrics `[F-03]`

**File:** `src/simulation.py`, `_handle_metrics_log`, lines 566-569 and the percolation branch, lines 586-588.

Replace `all_pos = self._get_all_positions()` with an alive-filtered array for **all** metric computation:

```python
alive_pos = self._get_all_positions()[self.alive_mask]
metrics = (compute_connectivity_metrics(alive_pos, self.config.comm_radius)
           if len(alive_pos) > 1 else
           {"largest_component": len(alive_pos), "component_count": len(alive_pos),
            "connectivity_ratio": 0.0, "spectral_gap": 0.0})
```

Do the same for the KD-Tree edge count at line 587.

Also fix `src/metrics/connectivity_metrics.py:105`: `np.linalg.eigvals` → `np.linalg.eigvalsh` (the Laplacian is symmetric; `eigvalsh` returns sorted reals). Drop the now-redundant `np.sort(np.real(...))`.

**Consider instead:** delete `compute_spectral_gap` entirely and call `src/analytics/spectral_analyzer.py:_compute_lambda2`, which is already correct. Two λ₂ implementations in one repo is how this diverged.

**Verify:** run
```python
# after a run where the swarm is fully dead
print(sim.alive_mask.sum(), sim.connectivity_log[-1]["spectral_gap"])
```
Currently prints `0 0.2919...`. **Must print `0 0.0`.**

**Evidence:** ✅ DONE

Changes made:
- `simulation.py` `_handle_metrics_log`: `alive_pos = self._get_all_positions()[self.alive_mask]`, with an explicit `n_alive <= 1` branch returning `spectral_gap = 0.0` (no edges can exist).
- Same alive-filter applied to the percolation branch's `active_network_edges` KD-tree count.
- `connectivity_metrics.py:compute_spectral_gap`: `np.linalg.eigvals` → `np.linalg.eigvalsh`; dropped the now-redundant `np.sort(np.real(...))`.

Correctness of the eigensolver change, against a closed-form value:
```
=== 1.1 VERIFY A: eigvalsh on a symmetric Laplacian ===
path graph P4 lambda_2 = 0.585786437627  exact 2-sqrt(2) = 0.585786437627  match=True
disconnected graph lambda_2 = 0.000000000000e+00  (must be ~0)
```

The F-03 case itself — a fully extinct swarm:
```
=== 1.1 VERIFY B: extinct swarm must log lambda_2 = 0.0 (was 0.2919) ===
alive at end            = 0/100
FINAL logged spectral_gap = 0.000000   (pre-fix: 0.2919)
FINAL logged LCC          = 0                (pre-fix: 100)
mean lambda_2 over run    = 0.0336   (pre-fix: 0.2923)
```

Trajectories must be untouched — λ₂ is logged, never fed back to agents:
```
1.1 REGRESSION -- lambda_2 is a LOGGED metric only; trajectories must be UNCHANGED
  Unconstrained    events=111727 (ref 111727)  ToD=434.0 (ref 434.0)  t50=105.0 (ref 105.0)  -> True
  Static-Epsilon   events=136935 (ref 136935)  ToD=None (ref None)  t50=106.0 (ref 106.0)  -> True
  Proposed         events=136935 (ref 136935)  ToD=None (ref None)  t50=106.0 (ref 106.0)  -> True
TRAJECTORIES UNCHANGED BY 1.1: True
```

**Result: mean λ₂ falls 8.7× (0.2923 → 0.0336) on this seed.** Table III's λ₂ column will change materially; see `[MS-02]` for whether it survives at all.

**Sparse `eigsh` deliberately NOT adopted.** It was carried into 1.1 on the stated grounds that "the driver is correctness rather than speed" — but that reasoning does not hold: dense `eigvalsh` is *more* accurate than sparse Lanczos, not less, and R0 measured the eigendecomposition at 2.2 % of runtime, so it is not a performance priority either. Sparse belongs in the N>10,000 scaling work (R7), not here.

**`[MS-02]` — RESOLVED BY MEASUREMENT: drop the λ₂ column from the Proposed-vs-Unconstrained comparison.**

Sensitivity study re-run after 1.1 (10 seeds, identical config, only the metric changed):

```
arm                            lambda_2  (pre-1.1)      decay/tick        survival   cens           t50
Unconstrained              0.009+/-0.002  (0.068)   0.2567+/-0.0442      427+/-88      0/10    102+/-4
Proposed                   0.010+/-0.002  (0.079)   0.0498+/-0.0001     2000+/-0      10/10    102+/-4
Oracle [all_to_all]        0.004+/-0.000  (0.497)   2.5761+/-0.4772       43+/-10      0/10     21+/-0
Oracle [per_neighbour]     0.049+/-0.002  (0.985)   0.0703+/-0.0065     1454+/-137     0/10    199+/-5
```

λ₂ collapsed by 7-20× across every arm once corpses were excluded. Two consequences, both to be stated:

1. **λ₂ no longer distinguishes Proposed from Unconstrained: 0.010 ± 0.002 vs 0.009 ± 0.002.** The confidence intervals overlap almost completely. Any claim that the framework maintains superior connectivity relative to the unconstrained baseline is **unsupported** and must be removed — including the abstract's "demonstrates sustained network connectivity" (`[MS-03]`) and §V-B's framing.
2. ~~λ₂ does still discriminate the oracle: 0.049 ± 0.002, ~5× the decentralized arms. Keep the column for the oracle comparison.~~ **RETRACTED — that claim was itself confounded. See below.**

Note the pre-1.1 numbers would have supported the opposite reading for the oracle (0.497 for `all_to_all`, which is in fact the *worst*-connected arm once corpses are removed — its swarm is dead by t=43, so nearly all of its logged λ₂ was corpse geometry).

#### Confound check: the surviving separation was survivorship, not connectivity

Having concluded λ₂ "still discriminates the oracle", I checked whether that separation was real before relying on it. It was not. After 1.1 an extinct swarm logs λ₂ = 0.0, so the **run-mean averages connectivity-while-alive together with a long tail of post-extinction zeros** — an arm that merely lives longer scores higher without maintaining connectivity any better. Added `n_alive` to `connectivity_log` and re-measured under three conditionings:

```
lambda_2 confound check -- 10 seeds

arm                           run-mean l2     l2 | n_alive>1      l2 | t<=100  %ticks alive  mean N alive
----------------------------------------------------------------------------------------------------------
Unconstrained              0.0092+/-0.0018    0.0590+/-0.0095   0.1836+/-0.0343         15.8%          38.2
Proposed                   0.0098+/-0.0016    0.0101+/-0.0018   0.1949+/-0.0321         97.6%           9.0
Oracle [all_to_all]        0.0040+/-0.0003    0.3225+/-0.0246   0.0806+/-0.0061          1.2%          82.5
Oracle [per_neighbour]     0.0493+/-0.0016    0.0719+/-0.0065   0.5316+/-0.0331         70.1%          24.7
```

The run-mean is dominated by how long each arm survives, not by connectivity:
- `Oracle [all_to_all]` run-mean **0.0040** vs alive-conditioned **0.3225** — an **80× swing**, because it is alive for only 1.2 % of logged ticks. On the run-mean it looks like the worst-connected arm; conditioned on being alive it is the best.
- `Unconstrained` swings 6.4× (0.0092 → 0.0590) for the same reason.
- `Proposed` barely moves (0.0098 → 0.0101) because it is alive for 97.6 % of ticks — which is exactly why it *appeared* comparable on the run-mean.

**λ₂ is not comparable across arms at any conditioning**, because every conditioning is confounded by a different variable:
- *Run-mean* → confounded by survival time.
- *Conditioned on `n_alive > 1`* → confounded by swarm size. Proposed averages 9.0 living agents against Unconstrained's 38.2; a sparser graph has lower λ₂ regardless of coordination quality. On this measure Proposed (0.0101) looks 5.8× **worse** than Unconstrained (0.0590), which is an artifact of population, not a finding.
- *Early window (t ≤ 100)* → the only fair Proposed-vs-Unconstrained comparison, since their attrition is identical (t50 = 102 for both). It shows **0.1949 ± 0.0321 vs 0.1836 ± 0.0343 — overlapping, no separation.** For the oracle arms even this window has divergent populations, so it does not license a clean claim either.

**`[MS-02]` FINAL: drop the λ₂ column from Table III entirely.** Per the standing directive, no column is kept alive on differences smaller than their confidence intervals, and no substitute connectivity metric will be sought to restore a separation. §V-B should state that under corrected measurement λ₂ does not distinguish the proposed framework from the unconstrained baseline, and that cross-arm λ₂ comparison is confounded by differential survival and swarm size, so it is not reported as a discriminating metric.

**`[MS-03]` FINAL — flagged now rather than deferred to the Phase 2 sweep, as directed.** The abstract's *"Stress testing with 100 agents over 15 million discrete events demonstrates **sustained network connectivity**"* **cannot survive**. There is no measurement supporting it: λ₂ does not separate Proposed from Unconstrained in the only fair window (0.195 vs 0.184, overlapping CIs), and the proposed framework's own alive-conditioned λ₂ is 0.0101 with a mean of 9 surviving agents. The clause must be deleted from the abstract, not softened. §V-B's "the system successfully stabilizes the network without centralized topology matrices" falls with it.
**`[MS-03]`** Abstract: "demonstrates sustained network connectivity" — re-verify against the corrected λ₂ and delete if unsupported.

### 1.2 — Separate the two ablation arms, or delete one `[F-02]`

`test_mode="static_bounded"` and `test_mode="none"` are bit-identical (verified across 3 seeds). Two options; pick one and state it in the paper.

**Option A (recommended, honest and cheap):** delete the `static_bounded` arm. Report three arms: Unconstrained / Proposed / **True Oracle** (`global_info_enabled=True`, which is a real global-information system: Hungarian assignment at `simulation.py:459-479`, true global Voronoi at `simulation.py:684-712`, exact global consensus at `simulation.py:406-411`). Table III currently has no oracle row at all; this gives it a real one.

**Option B:** make `static_bounded` actually differ by pinning `gossip_epsilon` to a fixed constant rather than merely skipping the dynamic bound. Only worth it if the paper needs a static-ε ablation.

**Verify (for A):** run `run_monte_carlo_table.py` and confirm the `True Oracle` row differs from `Proposed` on at least one metric.

**Evidence:** ✅ DONE — **Option A taken.** The `static_bounded` arm is removed from `run_monte_carlo_table.py`; the three remaining arms are Unconstrained / Proposed / True Oracle (billed `all_to_all` per the 0.6.2 decision). Full 50-seed output under 1.3 below. The oracle differs from Proposed on every metric (t50 21 vs 101, decay 2.801 vs 0.050, survival 39 vs >2000), so the arms are genuinely distinct — unlike the arm that was removed.

**`[MS-04]`** Table III row 2 is currently labelled "Centralized Oracle (Static Bound)". Under Option A, replace it with the real `True Oracle` results. Under Option B, rename it to "Static-ε Baseline" and **delete** the sentence "a Centralized Oracle (a theoretically perfect, global-information system serving as an upper bound)".
**`[MS-05]`** Delete "The oracle exhibits a significantly tighter confidence interval ($\pm 40$ ticks)". The table on the same page prints ±2. Whichever survives, they must agree.
**`[MS-06]`** Delete "the Proposed Framework perfectly matches the Centralized Oracle's maximum swarm survival" — under Option A this must be re-measured; under Option B it was an artifact.

### 1.3 — Report survival honestly under censoring `[F-09]`

The instrumentation exists (`run_monte_carlo_table.py:29-33, 152-155`); the reporting does not.

Change the reported statistic from a censored mean to:
- **median survival with censoring noted**, or
- **Kaplan–Meier** survival with the number at risk, or
- if `> 50 %` of runs are censored, report `"> max_time"` and refuse to print a point estimate.

Additionally promote `time_to_50pct_attrition` to a **primary** reported metric — it is uncensored and, per F-09, it is the metric that actually discriminates (and shows the two arms are equivalent: 105 vs 106).

**Verify:** the MC output must print, for each arm, `Censored Runs (Hit 2000s max_time cap): N/50` with `N > 0` for the Proposed arm, and must not print a bare mean alongside it without the qualifier.

**Evidence:** ✅ DONE — reporting rewritten: t50 promoted to PRIMARY, censored survival refuses to print a point estimate when >50 % of runs are censored, λ₂ and `coverage_completion_rate` demoted to explicitly-labelled diagnostics.

```
--- Final Results (50 Runs) ---
PRIMARY METRIC: Time to 50% Attrition. It is uncensored and is the
only survival measure that discriminates between arms.

[Unconstrained]
  Time to 50% Attrition (PRIMARY): 100 +/- 2
  Energy Decay Rate: 0.2277 +/- 0.0178
  Swarm Survival (Total Death): 476 +/- 38  (0/50 censored)
  [diagnostic, NOT a reportable result] run-mean lambda_2: 0.0095 +/- 0.0008
  [diagnostic, known-broken metric] coverage_completion_rate: 92.09%

[Proposed]
  Time to 50% Attrition (PRIMARY): 101 +/- 2
  Energy Decay Rate: 0.0501 +/- 0.0005
  Swarm Survival: RIGHT-CENSORED in 49/50 runs (swarm never fully died).
    -> lower bound only; median >= 2000, mean of censored data 1992 is NOT a measurement.
    -> 49/50 censored: report as '> 2000 ticks', do NOT quote a point estimate.
  [diagnostic, NOT a reportable result] run-mean lambda_2: 0.0098 +/- 0.0008
  [diagnostic, known-broken metric] coverage_completion_rate: 100.00%

[True Oracle]
  Time to 50% Attrition (PRIMARY): 21 +/- 0
  Energy Decay Rate: 2.8010 +/- 0.2067
  Swarm Survival (Total Death): 39 +/- 4  (0/50 censored)
  [diagnostic, NOT a reportable result] run-mean lambda_2: 0.0042 +/- 0.0002
  [diagnostic, known-broken metric] coverage_completion_rate: 100.00%

real	4m1.016s
```

**The 50-seed result confirms the 10-seed finding: the framework does not delay half-swarm loss.** t50 is **101 ± 2 (Proposed) vs 100 ± 2 (Unconstrained)** — a difference an order of magnitude inside the confidence intervals. Written into §V-A as an explicit refutation, not omitted.

**The framework's genuine, well-supported result is thermodynamic:** energy decay **0.0501 ± 0.0005 vs 0.2277 ± 0.0178**, a 4.5× reduction with cleanly separated intervals.

**Table III regenerated and the `PENDING REGENERATION` block removed:**
```
=== these must all be 0 ===
PENDING REGENERATION       0
1999                       0
perfectly matches          0
Spectral Connectivity      0
theoretically perfect      0
```
Columns are now Time to 50% Attrition (primary) / Energy Decay Rate / Total Swarm Survival, with censoring stated in-cell (`> 2000` (49/50 censored)) and the λ₂ column dropped per `[MS-02]`. `[MS-04]`, `[MS-05]`, `[MS-06]`, `[MS-07]`, `[MS-08]`, `[MS-09]` are discharged by this edit; `[MS-03]` (abstract) remains open.

**`[MS-07]`** Table III's "Mean Swarm Survival (ticks)" column: add a censoring footnote, or replace with median + censored count. `1999 ± 2` currently means "3 of 100 drones idled until the clock stopped".
**`[MS-08]`** Add a `Time to 50 % Attrition` column to Table III. Report that it is ~105 ticks for **both** arms, i.e. the framework does not delay the loss of half the swarm. This contradicts the current narrative; state it anyway.
**`[MS-09]`** §V-C: "We evaluated resilience against node attrition rates approaching 50\%" — the logs show 92 % (framework) and 100 % (baseline). Correct the number and Figure 4's caption ("even as node attrition reaches 50%").

### 1.4 — Plot the real λ̂₂, or stop calling it λ̂₂ `[F-19]`

**File:** `src/simulation.py:584` logs mean neighbour count under the name `avg_local_lambda_proxy`.

Fix the source, not the label: log the actual Eq. (2) quantity, which is already computed and thrown away:

```python
avg_local_lambda = (sum(a._lambda_2_proxy for a in alive_agents) / len(alive_agents)) if alive_agents else 0.0
```

Then remove the unexplained `/ 100.0` at `experiments/plot_results.py:68` and re-derive any scaling from the definition, or drop the scaling and use a secondary axis.

If the corrected λ̂₂ does **not** overestimate true λ₂ by ≈+0.04, the §V-B narrative is void and must be rewritten around whatever it does show.

**Verify:**
```python
print(df['avg_local_lambda_proxy'].describe())
```
Values must no longer be integer-valued neighbour counts (currently 0.0, 8.055, 19.43). Then print the measured mean of `(λ̂₂ − λ₂)` and compare to the claimed `+0.04`.
**Evidence:** _(paste output)_

**`[MS-10]`** §V-B: replace the `+0.04` figure with the measured value from the corrected proxy. If the sign or magnitude differs, rewrite the "Disconnected Subgraph Paradox" paragraph to match.
**`[MS-11]`** §V-B claims the bias is measured "across the 50 Monte Carlo runs". Figure 2 is **one** run at N=200. Either run the percolation experiment over 50 seeds, or change the text to "a representative single run (N=200)". Do not leave the Monte Carlo claim attached to a single run.

### 1.5 — Make the metrics logger side-effect free `[F-17]`

**File:** `src/simulation.py`.

1. Move the interference ramp (line 574) out of `_handle_metrics_log` into its own `ENV_UPDATE` event, or into `_handle_kinematic_update`. A metrics handler must not drive physics.
2. Remove the tick counters from `compute_velocity` (`agent_core.py:193-195`) and increment them in `_handle_kinematic_update` instead, where a tick actually occurs.
3. For the logged `avg_kinematic_velocity` (line 612) and `max_velocity_cmd` (line 629), either cache the velocity computed during the kinematic update on the agent (`self.last_velocity`) and read that, or add a pure `peek_velocity()` that does not touch `self._rng`.

Item 3 is the one that matters: `compute_velocity`'s random-walk branch draws from the agent RNG (`agent_core.py:239`), so logging currently perturbs the trajectory.

**Verify:** the determinism test that currently cannot exist —
```python
a = Phase1Simulation(SimConfig(seed=5, num_agents=20, max_time=100.0, test_mode=None)); a.run()
b = Phase1Simulation(SimConfig(seed=5, num_agents=20, max_time=100.0, test_mode="thermodynamics")); b.run(); b.close_loggers()
print(np.allclose([x.position for x in a.agents], [x.position for x in b.agents]))
```
Must print `True`. It will print `False` today.
**Evidence:** _(paste output)_

**`[MS-12]`** Contribution 4 ("Deterministic Reproducibility ... exact replay of stochastic failures") is only true once this passes. Do not restate it until the check above prints `True`.

---

## Phase 1.6 — Pre-Phase-2 hardening (author-directed, 2026-08-16)

### 1.6.1 — Close the scratch-log hole `[self-inflicted, caught during 1.5]`

My 1.5 verification runs used `test_mode='thermodynamics'` against the real `logs/` directory, clobbering the merged CSV. It was caught only because `plot_results.py` raised `KeyError`. **Had the corrupted file stayed parseable, it would have produced a plausible but wrong figure — the exact F-01 failure mode.** Two structural fixes, not a convention:

1. **Derived outputs renamed so a raw run cannot collide with them.** A bare simulation with `test_mode='thermodynamics'` writes `experiment_2_thermodynamics.csv`; the merge step now writes `experiment_2_thermodynamics_merged.csv` (same for stability). The collision is now impossible rather than merely discouraged.
2. **`plot_results.py` validates its inputs.** New `load_checked()` asserts required columns per figure and names the producing script in the error.

**Evidence:** ✅ the guard firing on a genuinely missing input:
```
FileNotFoundError: logs\experiment_3_stability_merged.csv not found.
Generate it with: python experiments/run_stability_test.py
```
and passing once regenerated:
```
  [experiment_3_stability_merged.csv] 40 rows, columns verified
  [experiment_1_percolation.csv] 100 rows, columns verified
  [experiment_2_thermodynamics_merged.csv] 60 rows, columns verified
```

### 1.6.2 — Repair Algorithm 1's tests before documenting it `[F-08, blocks MS-13]`

Documenting an untested component is backwards, especially this one: per F-11 the box clamp is the load-bearing step behind the entire surviving result. All 4 tests failed with `TypeError: missing 'theta_nominal'`.

**Evidence:** ✅ 4 repaired + 4 new tests asserting the two-stage contract:
```
tests/test_safety_projector.py::test_safety_projector_clips_values PASSED
tests/test_safety_projector.py::test_safety_projector_passes_valid_values PASSED
tests/test_safety_projector.py::test_safety_projector_handles_unknown_keys PASSED
tests/test_safety_projector.py::test_safety_projector_determinism PASSED
tests/test_safety_projector.py::test_box_clamp_raises_coverage_gain_from_zero_to_lower_bound PASSED
tests/test_safety_projector.py::test_velocity_scale_zero_is_NOT_clamped PASSED
tests/test_safety_projector.py::test_bisection_operates_on_the_box_clamped_value PASSED
tests/test_safety_projector.py::test_bisection_does_not_fire_when_value_is_already_safe PASSED

============================== 8 passed in 0.34s ==============================
```
The new tests pin exactly what `[MS-13]` must document: stage 1 raises `coverage_gain` 0.0 → 0.5 (and that this clears the `> 0.05` gate keeping the agent in the coverage law), stage 2 then bisects **the clamped value**, and `velocity_scale = 0.0` is *not* clamped despite a code comment claiming it is.

Suite: **15 failed / 95 passed**, down from 19/87. No new failures.

### 1.6.3 — Clamp ψ and mark the saturated regime `[F-39 / MS-30]`

**Evidence:** ✅ clamp added in `InterferenceField.evaluate`, and it is behaviour-neutral:
```
F-39 CLAMP VERIFY
  psi_max=2.55 -> evaluate() = 1.0   (must be 1.0)
  psi_max=-0.3 -> evaluate() = 0.0   (must be 0.0)
  psi_max=0.40 -> evaluate() = 0.4   (unchanged in range)

Percolation run AFTER clamp:
  final psi_max attribute = 2.550 (ramp still runs)
  evaluated psi           = 1.000 (clamped)
  agents alive at end     = 8   (pre-clamp A/B gave 8)
```
Identical outcome, because ψ > 1 already produced a negative `p_survive` and hence total drop. This is a correctness/clarity fix, not a behavioural one.

**Figure 2:** full range retained, with the post-blackout region shaded and annotated rather than truncated — the saturated regime is informative once labelled. §V-B no longer claims jamming "increases to 100%".

**Y-axis relabelled.** The regenerated figure made the F-19 scale gap visible (λ₂ flat below 0.151, proxy peaking above 7), but the axis still read "Fiedler Value" for both curves — re-introducing the exact conflation the `/100` created. Now "Magnitude (curves are not on a common scale)", with the caption stating it.

---

## Phase 2 — Align the algorithms in the paper with the algorithms in the code

### 2.1 — `[MS-13]` Add Algorithm 1 Step 1 (box clamp) `[F-11]`

**File:** `manuscript/final_manuscript.tex`, Algorithm 1 block (lines 177-199).

Insert, before the existing "// Step 2: Bounded Heuristic Clamping Search":

```
// Step 2a: Static Box Clamp
for each parameter k in θ_prop do
    θ_safe[k] ← min(max(θ_prop[k], θ_min[k]), θ_max[k])
end for
```

and state the actual `THETA_SAFE_BOUNDS` table from `src/adaptation/safety_projector.py:21-28`. Renumber the bisection to Step 2b.

Also document the unstated fallback at `safety_projector.py:82-83` (`if low >= bound: low = bound * 0.99`), since the printed precondition `θ_nominal ≤ θ_bounds ≤ θ_prop` is routinely violated in practice.

**Verify:** line-by-line diff of the pseudocode against `safety_projector.py:55-100`; every code branch must appear in the pseudocode and vice versa.
**Evidence:** _(paste the mapping)_

### 2.0 — `[MS-20]` KD-tree — DONE, inside timebox

**Implemented**, ~3 hours against the 1-day timebox. `RGGBuilder` gains `build_tree` / `query_radius`; the orchestrator owns the tree with a dirty flag set by `_handle_kinematic_update` (the sole position mutator), so it can never be served stale under any event interleaving. Correctness does not depend on ordering — ordering only sets how often we rebuild.

The tree is a **candidate filter only**: query radius widened by 1 part in 1e12, results sorted ascending, then the original exact `np.linalg.norm <= tx_radius` test applied. This preserves per-neighbour RNG consumption order exactly, which a raw tree query would not.

**Evidence — bit-identical, including RNG stream:**
```
  Unconstrained    events=111727 (ref 111727)  ToD=434.0  t50=105.0  ->True   tree_rebuilds=433
  Static-Epsilon   events=136935 (ref 136935)  ToD=None   t50=106.0  ->True   tree_rebuilds=2000
  Proposed         events=136935 (ref 136935)  ToD=None   t50=106.0  ->True   tree_rebuilds=2000
BIT-IDENTICAL TO PRE-KDTREE REFERENCE: True

  total_sent=100363 total_dropped=68780 total_delivered=31583
  reference (pre-KDTree): total_sent=100363 total_dropped=68780 total_delivered=31583
```
`tree_rebuilds = 2000` over a 2000-tick run confirms exactly one rebuild per tick, as the priority ordering predicts.

**Evidence — it buys nothing at tested scales:**
```
MS-20 -- linear scan vs K-D Tree, constant density, max_time=300
     N    scan_s    tree_s   speedup    events   alive
    50      5.19      5.11     1.02x     44708       1
   100     10.64     10.52     1.01x     89097       5
   200     21.88     21.25     1.03x    178524      10
   400     49.65     45.09     1.10x    357994      20
   800    113.36    118.35     0.96x    706592      32
```
At the suite operating point (N=100, max_time=2000), 3 alternating reps: scan median 21.7 s, tree median 21.7 s, **ratio 1.00×**. (An earlier single reading of 27.1 s was machine noise, not a regression.)

**Decision: keep it.** The justification was always claim alignment, never speed. §III-C's K-D Tree claim is now literally true, the cost is zero, and it is the right structure for the Rust port. §III-C rewritten to state the derived complexity *and* that wall-clock at N ≤ 800 cannot distinguish it, and that no scaling exponent is claimed (`[MS-29]`).

### 2.1 — `[MS-14]` The bisection is REDUNDANT, not inert — decision required before §IV is rewritten `[F-10 CORRECTED]`

**My original F-10 explanation was wrong and is corrected here.** F-10 claimed consensus was causally inert because `RegimeClassifier` tests `FRAGMENTED` first and the variance-dependent branches are never reached. Measured, that is false:

```
Regime classification branch taken, 3 seeds, 10,221 classifications

deciding branch                                       count    share
FRAGMENTED (density/staleness) [staleness]            7,175   70.20%
LATENCY_OSCILLATION (VARIANCE-dependent)              2,420   23.68%
MARGINAL (VARIANCE-dependent)                           217    2.12%
FRAGMENTED (density/staleness) [density]                183    1.79%
ENERGY_CASCADE (energy slope)                           139    1.36%
FRAGMENTED (density/staleness) [both]                    86    0.84%
MARGINAL (staleness)                                      1    0.01%
--------------------------------------------------------------------
decisions where consensus variance was the decider      2,637   25.80%
```

Consensus variance decides **25.8 %** of classifications. The subsystem is live. The real mechanism is a **duplicated bound**:

```
[Proposed (bisection ON)]
   agent gossip_epsilon  mean          = 0.0158935
   epsilon ACTUALLY USED mean          = 0.00389701
   internal clamp in gossip_consensus bound the value: 98.52% of calls
[Static-Eps (bisection OFF)]
   agent gossip_epsilon  mean          = 0.0429951
   epsilon ACTUALLY USED mean          = 0.00390103
   internal clamp in gossip_consensus bound the value: 98.62% of calls
```

The projector genuinely moves the parameter (0.0159 vs 0.0430 — a 2.7× difference). Then `gossip_consensus.py:69` independently recomputes `safe_bound = 0.99/(d_i(τ_max+1))` and clamps, on **98.5 % of calls**, so the epsilon actually applied is the same to three significant figures (0.003897 vs 0.003901). Algorithm 1's dynamic bound is overwritten downstream by a second implementation of itself.

Compounding this, the two implementations **disagree**: `gossip_consensus.py:65` uses `math.ceil` for τ_max while `agent_core.py:517` uses `int()` (floor) — F-24. Two bounds, different discretisations, and the undocumented one wins.

**Decision needed before §IV is rewritten. Recommendation: option (b).**

- **(a) Report as a negative result.** Cheap, honest, no re-runs. But it would document an architecture in which the paper's Contribution 3 is knowingly overridden by an undocumented clamp — reporting a defect as a finding.
- **(b) Make the projector authoritative** — remove the internal clamp from `compute_gossip_update`, so the bound is enforced once, in the component the paper describes. This is what §IV already claims happens. **Recommended.** Cost: a behaviour change requiring a full suite re-run, and it may move Table III. Also resolves F-24 by deleting one of the two discretisations.
- **(c) Make the gossip function authoritative** — drop the projector's dynamic bound and describe Algorithm 1 as a box clamp plus EMA only. Honest, but discards the paper's stated contribution.

I have **not** rewritten §IV pending this decision.

#### RESOLUTION — option (b) taken (author decision, 2026-08-16). ✅ DONE

Sequence executed as directed: clamp removed → sole-enforcement verified → F-24 confirmed dead → full suite re-run → §IV rewritten last.

**Code changes:**
- `gossip_consensus.py`: internal clamp deleted; `compute_gossip_update` applies the given epsilon verbatim. Module docstring now states the one-bound-one-owner policy and why a clamp must not be reintroduced.
- `agent_core.py:523`: surviving bound switched `int()` → `math.ceil()` — the conservative discretisation, and the one the previously-winning clamp used, so the bound's semantics carry over. **F-24 is dead by construction: one implementation remains.**
  ```
  === bound implementations remaining (must be agent_core only) ===
  src/agent/agent_core.py:524:                safe_bound = 0.99 / (d_i * (tau_max + 1))
  src/coordination/gossip_consensus.py:9:  [docstring text only]
  ```
- `audit_findings.md` F-10 corrected in place with the measured branch shares (variance decides 25.80 %), per the author's instruction not to leave the wrong explanation standing.

**Sole-enforcement verification (every call instrumented):**
```
[Proposed (bisection ON)]     applied != given on 0 calls   mean eps used = 0.0161845
[Static-Eps (bisection OFF)]  applied != given on 0 calls   mean eps used = 0.0427108
[Unconstrained (no projector)]applied != given on 0 calls   mean eps used = 0.0608024
```
Pre-fix, both bounded arms applied 0.0039 identically. The projector's bound now governs: Proposed's applied epsilon rose ~4×, exactly as predicted.

**Exposed consequence, reported not hidden:** with no bound anywhere, the Unconstrained arm's consensus **genuinely diverges** — `max|state| = 2.1e+39` at t=600. The duplicate clamp had been silently stabilising the baseline too. On one seed the divergence reached ~1e154+ and crashed `statistics.variance` (exact-fraction arithmetic) inside the regime monitor, killing the MC suite:
```
OverflowError: integer division result too large for a float
```
Fixed in `local_proxies.py`: the variance proxy now saturates at `VARIANCE_PROXY_CEILING = 1e12` (np.var, ddof=1) instead of crashing. This is overflow protection on a monitor, not tuning: the classifier only thresholds this value at ~1.5, so saturation changes no decision. Divergence remains fully observable — it reads as maximal variance, which is the semantically correct proxy signal. Unit-verified including the exact crash case (1e160-scale states → 1e12) and the spectral proxy under saturation (ratio 1 → λ̂ = 0, finite).

**Full 50-seed suite re-run — Table III does NOT shift materially. MS-04..MS-09 stay discharged:**
```
[Unconstrained]  t50 100 +/- 2   decay 0.2282 +/- 0.0175   survival 474 +/- 38  (was: 100, 0.2277, 476)
[Proposed]       t50 101 +/- 2   decay 0.0501 +/- 0.0005   49/50 censored       (was: identical)
[True Oracle]    t50  21 +/- 0   decay 2.8010 +/- 0.2067   survival  39 +/- 4   (was: identical)
```
Every delta is inside its CI. The 4× epsilon change and genuine baseline divergence move **nothing physical** — confirming the bound protects the consensus layer while contributing negligibly to the energy result. That decomposition is now stated in §V-D rather than left implied.

**Oracle sensitivity re-run:** only the Unconstrained row moved, within CI (decay 0.2567→0.2462, survival 427→442); manuscript table updated. Oracle rows identical (they bypass gossip).

**§IV rewritten against the new numbers (`[MS-13]` discharged in the same pass):**
- Algorithm 1 pseudocode now has **Step 2a (static box clamp, every parameter)** and **Step 2b (dynamic-bound bisection, `gossip_epsilon` only)**, the `low ≥ b_k → 0.99·b_k` fallback, and ceil discretisation — a line-for-line match to `safety_projector.py` + `agent_core.py`, each property pinned by a passing test from 1.6.2. The false precondition line is gone; three implementation properties (bisection operates on the clamped value; no nominal-safety precondition; EMA lag means transient exceedance of a just-tightened bound) are stated explicitly.
- The ε-bound sentence no longer claims it "prevents kinematic divergence" — consensus state does not drive kinematics. It protects the consensus iteration, with the measured counterfactual (states > 1e39 without it) cited.
- **Bonus correction:** §IV's "empirically observed to deviate from the true global average by approximately 5% to 8%" was unverified. Measured (10 seeds, t=100, pre-attrition): **mean 2.59 %, range 0.33–6.03 %**. Text now carries the measured numbers.
- **Second bonus:** my earlier evidence that "15 million" was gone was a **false negative** — the string survived in §V-A as `$15 \text{ million}$`, defeating the plain grep. Removed now, along with "Gaussian attenuation" and the false "all quantitative experiments were averaged over 50 runs" blanket, replaced by a per-experiment configuration table (`[MS-23]` discharged: N/runs/seeds/horizon per experiment, measured event counts).

All stale-claim greps now 0 with LaTeX-robust patterns. Figures regenerated post-(b): fig2 (percolation) changed as expected; fig1/fig3 hashes identical — those experiments are insensitive to the epsilon change at their seeds. All md5-match. Suite: same 15 failures, none new.

### 2.2 — `[MS-14]` (superseded by 2.1 above) `[F-10]`

Measured: the bisection fires on 2581/2588 checks and drives `gossip_epsilon` to 2.9e-05, yet removing it changes **nothing** in agent positions or energies (F-02), because `consensus_state` never reaches a deciding branch of `RegimeClassifier.classify`.

The paper currently presents this bisection as Contribution 3 and the subject of Algorithm 1. Two honest options:

- **(a)** Report the negative result: the ε-bound is enforced as designed but has no measurable effect on the studied metrics, and the observed benefit comes from the box clamp. A clean negative result is publishable; a mislabelled positive one is not.
- **(b)** Add an experiment where consensus *does* drive behaviour (e.g. make `RegimeClassifier` reachable via `mean_variance` by fixing F-12 so `FRAGMENTED` stops dominating), then re-measure.

Option (b) is real work and may still yield a null result. Choose before rewriting §IV.

**Verify:** re-run the Probe-6 instrumentation and report `bisection fired / checks` plus a paired comparison with the bisection disabled.
**Evidence:** _(paste output)_

### 2.3 — Reconcile Algorithm 2 with `auction.py` `[F-14]`

Pick one direction and make the other match. **Recommended: change the code**, because the paper's formulation (energy-aware bidding) is the one that supports the §V-C claim.

**Code changes** in `src/coordination/auction.py`:
- Line 32: add the energy term. `compute_bid` currently returns `task_reward - dist` with no energy dependence. Implement `bid = ω_d·‖p_i − p_τ‖ + ω_e·(1/E_i)` and flip `update_local_winner:48` to minimise, or keep the utility convention and document it as `reward − ω_d·d − ω_e/E_i`. Whichever you choose, the `.tex` must print the same formula.
- Fix the unit inconsistency at line 29 (`if dist > agent_energy`) — compare `dist * p_move` against `agent_energy`.
- `agent_core.py:430`: `active_task_id` is a single slot; winning a second task silently abandons the first. Either refuse to bid while holding a task, or make it a queue.

**Verify:** assert energy actually influences allocation —
```python
# two agents equidistant from a task, different energies
b_hi = compute_bid(pos, 100.0, task, 100.0); b_lo = compute_bid(pos, 5.0, task, 100.0)
print(b_hi, b_lo, b_hi != b_lo)
```
Must print `True`. Prints `False` today (both return `100.0 - dist`).

**Evidence:** ✅ code DONE; ⚠️ manuscript BLOCKED on author review (Table III moved).

Implemented: min-cost bid `dist + ω_e/E_i` (ω_e = E₀ = 100, a scale choice not a tuned one; reward dropped from the bid — it is constant per task across bidders and cannot change the winner, which also means the old `reward − dist` carried zero energy dependence); feasibility in consistent units (`p_move·dist ≥ E → inf`, and `inf` cannot win under min — the old `-inf` would have); no-bid-while-holding (kills the silent-abandonment overwrite); auction expiry in `LocalMap`; dead `update_local_winner` deleted (unused duplicate of `update_auction`'s rule — same pattern as F-10/F-24).

```
2.3 VERIFY -- energy must influence the bid
  equidistant, E=100 -> bid 51.000
  equidistant, E=5   -> bid inf          (feasibility: p_move*50 = 5.0 >= E)
  differ: True   (was False: both returned reward-dist)
feasibility in consistent units:
  E=4.9, cost-to-reach=5.0 -> bid = inf   |   E=5.1 -> bid = 69.608
```

**Regression I introduced and caught: gossip starvation.** My first replacement for the random gossip pick was "most recently seen". That starves every concurrent auction — gossiping the newest refreshes its timestamp on receivers, so it stays newest forever. Exposed by `test_coverage` (a NEW failure, investigated rather than waved through): two tasks spawned 0.015 ticks apart, task_0's bids never propagated, **all 100 agents resolved themselves its winner**, and the whole swarm converged on one point instead of dispersing (final variance 53 vs required >100). "Soonest-resolving" fails symmetrically (the older auction monopolises the younger's entire bidding window). Fixed with deterministic **round-robin** over live auctions — no RNG, cannot starve. `test_coverage` passes again; seed 1018's catastrophic collapse (below) also resolved by this fix (ToD=None after).

**Suite journey, all 50-seed runs:**

| run | Proposed decay | note |
|---|---|---|
| pre-auction reference | 0.0501 ± 0.0005 | |
| after bid/expiry/no-rebid, recency gossip | 0.0559 ± 0.0117 | bimodal: 49 runs at 0.050, **seed 1018 total death t=284, decay 0.352** — traced to sustained task pursuit draining the fragmented remnant (5–7 pursuers at v=2.0 with ~12 agents left); decomposition probe showed median unchanged at 0.0499 |
| after round-robin fix (**definitive**) | **0.0514 ± 0.0021** | 47/50 censored |

**⚠️ TABLE III MOVED — flagged per standing instruction, manuscript numbers NOT adopted:**
```
[Unconstrained]  t50  98 +/- 2   decay 0.2893 +/- 0.0216   survival 373 +/- 31  (0/50 censored)
[Proposed]       t50  98 +/- 2   decay 0.0514 +/- 0.0021   47/50 censored
[True Oracle]    identical to previous (bypasses the auction path)
```
- **Unconstrained decay and survival moved OUTSIDE their old CIs** (0.2282→0.2893, 474→373). Mechanism: the auction now works in the baseline arm too, so baseline agents pursue tasks and spend movement energy they previously never spent (the old broken auction was protecting the baseline the same way the duplicate ε-clamp was).
- Proposed decay 0.0501→0.0514, CI 4× wider; censoring 49/50→47/50 (three full-death runs — the pursuit-drain tail risk persists at reduced rate).
- t50 now 98 vs 98 — still indistinguishable, conclusion unchanged.
- **Headline ratio becomes 5.6× (was 4.5×) — a favourable shift, which is exactly when these numbers must not be self-adopted.** PENDING block updated above Table III; §V-C untouched; awaiting author review.

`[MS-15]` and `[MS-16]` — ✅ DONE (numbers adopted by author, 2026-08-16, with the framing directives applied):

- **Table III adopted**: t50 98±2 / 98±2 / 21±0; decay 0.289±0.022 / 0.051±0.002 / 2.801±0.207; survival 373±31 / >2000 (47/50 censored) / 39±4. Abstract and §V-A updated to match (5.6×). PENDING block removed.
- **`[MS-16]`** §V-C rewritten: (1) the 4.5×→5.6× shift stated explicitly as a *baseline correction, not an improvement* — the broken auction meant Unconstrained agents never paid task-pursuit movement costs, the same silent-protection pattern as the duplicate ε-clamp; (2) one line on the old bid's inert reward term (constant per task across bidders — the printed formula's reward-sensitivity never influenced a winner); (3) **the 3/50 (6%) total-loss tail reported as a finding, framed against F-04**: the Voronoi stationary fixed point halts isolated agents, task pursuit is the one mechanism that overrides the halt, the two are in direct opposition, and part of the headline energy advantage comes from *not* performing task work under fragmentation. Figure 4 caption fixed (96%/100% attrition, not 50%; single-run illustrative, stats from Table III).
- **`[MS-15]`** Algorithm 2 pseudocode rewritten against the implemented mechanism: event-driven per-task bidding (no task-set argmin), min-cost bid `‖p_i−p_τ‖ + ω_e/E_i` with `ω_e = E_0`, consistent-unit feasibility gate, single-commitment rule, expiry, round-robin gossip, and resolution as a read of the gossip-merged local belief — with the split-brain multiplicity stated as a measured property, not hidden. The gossip-selection failure history (random dilution; recency starvation) is documented in the algorithm's endnote.
- Renamed "Decentralized SSI Auction" → "Decentralized Energy-Aware Auction" throughout (the mechanism was never sequential-single-item); SSI survives only in the §II literature survey. **`[MS-21]` also discharged in passing**: §III-B's dwell-time sentence replaced with the honest polling-interval statement.
- `tab:oracle_sensitivity` re-measured under the adopted code (Unconstrained 0.259±0.039 / 418±83 / t50 96±5; Proposed 0.053±0.005 / 1926±137 / 9/10 censored; oracle rows unchanged — bypass the auction path). Oracle conclusions stable: per-neighbour still ~2× t50 (199 vs 97), all-to-all still shortest-lived.
- Figures regenerated from the adopted code and published (md5 MATCH all three).
- Retired-number sweep clean: `0.050±0.001`, `0.228` (except the intentional §V-C correction narrative), `4.5×`, `101±2`, `100±2`, `49/50`, `102±4`, `2000±0`, `10/10` — all 0 hits.

**`[MS-15]`** Update Algorithm 2's pseudocode to match: single-`τ_target` selection vs per-task bidding, the argmin/argmax convention, and the fact that resolution reads a cached winner rather than comparing against `min(B_nbrs)`.

### 2.4 — Restore auction activity, or scope the claim `[F-13]`

Measured: 26 of 129 tasks allocated; no win after t=183 of 2000.

**Fixes:**
- `agent_core.py:311-314`: gossip the **most recently started** or **soonest-resolving** auction, not a uniform random one. With 122 stale entries the random pick is a 4 % hit rate.
- Purge `active_auctions` entries whose `auction_timeout` has elapsed — currently only the winner's own task is ever removed (`handle_task_completion:446`).
- Reconsider `auction_participation = 0.0` in three of five strategies (`hybrid_supervisor.py:81, 86, 91`). If auctions are meant to be a "continuously-active core mechanism", they cannot be off in 4 of 6 regimes.

**Verify:**
```python
print(f"spawned={n_spawned} won={len(sim.auction_results)} unallocated={len(sim.active_tasks)}")
print(f"last win at t={max(t for _,_,t in sim.auction_results)} of {cfg.max_time}")
```
Target: allocation rate materially above the current 20 %, and a last-win time in the final quarter of the run. **Report the actual numbers achieved, whatever they are.**
**Evidence:** _(paste output)_

**`[MS-16]`** §V-C: "Because the localized SSI auction distributes tasks based on remaining energy margins ... the framework's average energy decay rate stabilized at approximately 0.05 units/tick" — this causal chain is false today (auction inactive, energy not in bid). Rewrite after 2.3 and 2.4, or delete the causal attribution and report the decay rate without it.

### 2.5 — Fix the equations `[F-20, F-21, F-24]`

**`[MS-17]` Eq. (1)** — remove `R̃_tx = R_tx − ω_env` (not implemented) and print the actual drop model from `packet_drop.py:60-63`:
`p_survive = (1 − p_drop)(1 − ψ)·max(0, 1 − (d/R_tx)²)`.
Also delete "decays quadratically with distance according to the inverse-square law" — `1 − (d/R)²` is not an inverse-square law and reaches exactly zero at `d = R_tx`.

**`[MS-18]` Eq. (2) guard** — §IV-B says the guard returns "a fully-fragmented proxy scalar"; `local_proxies.py:82-84` returns `1.0`, a *maximum* proxy, and §V-B's own paradox argument agrees with the code. Correct §IV-B. Also note the guard triggers on `current_variance` as well as `prev_variance`.

**`[MS-19]` Eq. (3)** — print `γ_c·(R_tx/R_base)²·N_msg` to match `agent_core.py:288`, or change the code to literal `γ_c·R_tx²`. As printed, Eq. (3) implies a 400× cost at `R_tx = 20` that the code does not charge.

**`[MS-20]` §III-C — DECIDED (author, 2026-08-16): implement properly here in Phase 2, or delete the claim. Do not wire a fixed-radius tree.**

The communication path is currently a linear scan (`comm_engine.py:107-109`). `RGGBuilder.build_neighbor_lists` is test-only and holds a **fixed** radius (`rgg_builder.py:33`), while senders transmit at `comm_radius * tx_power_scale`, `tx_power_scale ∈ [1.0, 2.0]`. Wiring it as written would silently disable dynamic transmission power — the same class of silent-flag-not-propagating defect this project has already hit twice (the `global_info_enabled` constructor omission, and `coverage_enabled` never being set).

Required implementation if attempted:
- **Per-sender `query_ball_point(sender_pos, sender_tx_radius)`**, not a single fixed-radius `query_pairs`.
- **Explicit staleness handling.** Broadcasts and kinematic updates interleave in the event queue, so a tree built once per tick is stale for any sender whose neighbours have moved since. Either rebuild on position change, or prove the ordering makes staleness impossible.
- Re-measure and confirm the decentralized arms remain bit-identical (same regression harness as 0.6.3).

**Timebox: ~1 day.** If it exceeds that, **delete the O(N log N) claim from §III-C** and state the measured complexity instead. Making the paper true by breaking a feature is not an acceptable outcome.

Note this is **not** a performance fix — it replaces a 3.7 % cost (R0). The only justification is claim alignment.

**`[MS-29]` §III-C / any scalability text — do not claim near-linear scaling.**

Both measured exponents (pre-fix N^1.52, post-fix N^1.04) are unsound: the swarm dies early, so `alive_end` is 4-5 regardless of N and the quadratic term never loads. Neither number may be carried into the paper. To make a real claim, re-run with survival-preserving parameters (raised `energy_initial`, or a short high-survival horizon) and measure the exponent there. Absent that run, §III-C's complexity statement rests on code inspection, not measurement, and should say so.

**`[MS-21]` §III-B** — either implement dwell-time enforcement (a last-transition timestamp checked in `HybridSupervisor.select_strategy`) or delete "strictly enforces a fixed Dwell Time (τ_d = 5.0 ticks) before permitting subsequent transitions" and the Liberzon/Zeno justification in `config.py:26-29`. Today `dwell_time` is only a polling interval.

**`[MS-22]` §V-A** — delete "stochastic Gaussian attenuation"; ψ is `FieldMode.CONSTANT`.

**`[MS-30]` §V-B / Figure 2 — NEW FINDING (F-39): the jamming ramp runs to 255 % and saturates at total blackout 38 % of the way through the run.**

Surfaced while isolating the ENV_UPDATE change. `run_percolation.py` starts at `psi_max = 0.05` and `_handle_env_update` adds `0.005 × 1.0 × 5 = 0.025` every 5 ticks for 500 ticks:

```
PERCOLATION JAMMING RAMP -- saturation analysis
  psi starts at 0.05, +0.025 every 5 ticks
  psi crosses 1.0 (total blackout) at t=191 of a 500-tick run
  final psi = 2.550  (i.e. 255% jamming)
  => 62% of the run is past total blackout

packet_drop.py: p_survive = (1-p_drop)*(1-psi)*path_loss
  at psi>1 the (1-psi) factor is NEGATIVE, so p_survive<0 and every
  packet drops. There is no clamp on psi_max, and InterferenceField
  documents psi in [0, psi_max] while psi_max itself is ramped past 1.
```

Consequences:
- §V-B says *"as environmental jamming increases to 100 %"*. It increases to **255 %**, reaching 100 % at t≈191.
- **The last 62 % of Figure 2's x-axis is not a jamming gradient**; it is a flat total-blackout regime in which no packet can ever be delivered. Any structure the reader infers from that region is post-extinction geometry, not a response to increasing interference.
- `InterferenceField` documents `ψ ∈ [0, ψ_max]` (`interference_field.py:63-66`) while `ψ_max` itself is ramped past 1.0 unclamped, and `PacketDropSampler` has no guard for `ψ > 1` — `p_survive` simply goes negative.

**Code fix:** clamp ψ to `[0, 1]` in `InterferenceField.evaluate`, or stop the ramp at 1.0. **Manuscript fix:** either truncate Figure 2 at t≈191 and state that the sweep covers 0-100 % jamming, or keep the full run and state explicitly that the network is fully jammed after t≈191.

**`[MS-28]` Introduction, line 60** — *"Without mathematical bounds, unconstrained adaptation often induces kinematic oscillations and network fracturing."* Not blocking (it is literature framing, not a measured claim about this system), but it motivates a failure mode we have now measured this system **not** to exhibit: all supervisor proposals are bounded constants in [0,2] and the unconstrained arm's trace is flat, not oscillatory (F-04, §V-D). Leaving it unqualified sets an expectation §V no longer meets. Either attribute it explicitly to prior work, or add a sentence noting that our unconstrained baseline degrades through a different mechanism.

**Verify (code-side, for the ones you implement):** unit test each corrected equation against a hand-computed value.
**Evidence:** _(paste output)_

### 2.6 — `[MS-23]` Correct the experimental-setup paragraph `[§11 of audit]`

§V-A currently asserts, for *all* quantitative experiments: N=100, 50 Monte Carlo runs, 95 % CI, >15 million discrete events. Measured reality:

| Experiment | Script | N | Runs | Seed |
|---|---|---|---|---|
| Table III | `run_monte_carlo_table.py` | 100 | 50 | 1000-1049 |
| Fig. 2 (spectral) | `run_percolation.py` | **200** | **1** | 42 |
| Fig. 4 (thermo) | `run_energy_cascade.py` | **50** | **1** | 42 |
| Fig. 5 (kinematic) | `run_stability_test.py` | **30** | **1** | 123 |

Replace the blanket paragraph with a per-experiment table. Replace "over 15 million discrete events" with the measured per-run count (**111,727–136,935** for N=100/2000 ticks) and, if a cumulative figure is wanted, state it as cumulative across all runs.

**Verify:** print `sim.run()`'s return value (events dispatched) for each experiment and paste the four numbers.
**Evidence:** _(paste output)_

### 2.7 — Enable coverage in the stability experiment `[F-#2 residual]`

**File:** `experiments/run_stability_test.py`, lines 17-26 and 33-42 — add `coverage_enabled=True` to both configs. Algorithm 3 is currently disabled for the experiment behind Figure 5.

Also change `src/core/config.py:108` `coverage_enabled: bool = False` → `True`, so no future runner silently disables Algorithm 3 again. Audit every remaining `SimConfig(...)` construction after the change.

**Verify:** `python -c "from src.core.config import SimConfig; print(SimConfig().coverage_enabled)"` → `True`, then confirm Figure 5's regenerated data differs from the current `logs/experiment_3_stability.csv`.
**Evidence:** _(paste output)_

**`[MS-24]`** If enabling coverage changes Figure 5, update §V-D's numbers.

### 2.8 — `[MS-25]` Fix Figure 5's caption, axis, and claims `[§11 of audit]`

- The plotted series is **max velocity magnitude**, not variance. Fix the y-axis label (`plot_results.py:50` currently reads "Maximum Kinematic Variance / Velocity ($v$)" — pick one) and the caption ("Kinematic variance ($\sigma^2$)").
- Remove the two invented annotations (`plot_results.py:42-45`): "Obstacle Avoidance Trigger" (no obstacle avoidance exists anywhere in `src/`) and "Severe Packet Loss Spike" (`run_stability_test.py` schedules no such spike).
- The `y=2.0` "Local Stability Boundary" is `v_max`, hard-clamped at `agent_core.py:274-275`. Relabel it "$v_{max}$ (kinematic saturation)" or remove it; neither trace can cross it, so it demonstrates nothing.
- "An unconstrained optimizer exhibits severe oscillations with parameter variations exceeding $\pm 0.3$" — unsupported; all supervisor proposals are bounded constants in [0,2]. Delete or substantiate.
- "strictly bounds parameter mutations to $\Delta\mu < 0.04$": measured **mean** Δμ = 0.0154 ✓ but **max** Δμ = 0.0761 ✗. Say "mean per-update shift" (as the abstract already correctly does) and drop "strictly bounds", or report the max separately.

**Verify:** `grep -c "Obstacle Avoidance" experiments/plot_results.py` → `0`; then print measured mean and max Δμ from `sim.adaptation_log[-1]`.
**Evidence:** _(paste output)_

---

## Phase 3 — Engine correctness (does not change Table III, but affects credibility of the artifact)

### 3.1 — Evict stale beliefs from `LocalMap` `[F-12]`

`remove_neighbor` (`local_map.py:74`) has **no caller**. Add age-based eviction in `process_inbox` or `handle_regime_update`:

```python
def evict_stale(self, current_time: float, max_age: float) -> int:
    stale = [k for k, v in self._beliefs.items() if current_time - v.timestamp > max_age]
    for k in stale: del self._beliefs[k]
    return len(stale)
```

This is a **behaviour-changing** fix: it will raise `FRAGMENTED` detection rates, shrink `τ_max`, and therefore raise `gossip_epsilon` off its current 2.9e-05 floor. It may change Table III. Re-run Phase 1 verification afterwards.

**Verify:** print `max(current_time - nb.timestamp for nb in agent._local_map.get_all_neighbors())` at end of run. Currently unbounded (implied τ_max ≈ 2800); must be ≤ `max_age`. Also print the resulting mean `gossip_epsilon`.

**Evidence:** ✅ code DONE; ⚠️ **manuscript BLOCKED — Table III moved again, and two prior findings change character. Flagged below, nothing edited.**

**max_age chosen from measurement, not a round number.** Empirical distributions (seed 1000, N=100, 2000 ticks):
```
inter-refresh gaps, LIVE-neighbour beliefs (n=28,620): p50=2  p90=7  p95=10  p99=31  p99.9=76  max=118
sampled belief ages, DEAD senders (n=23,175):          p50=263  p90=1513  p99=1839  max=1925
```
The populations are separable by two orders of magnitude. **`belief_max_age = 30.0` = p99 of live-link cadence** (~1 % spurious-eviction risk), and above the FRAGMENTED staleness trigger (3×3.0 = 9.0) so staleness-based detection still sees its evidence. Derivation recorded in `SimConfig`.

**Sensitivity (15 ≈ p95 / 30 / 60 ≈ p99.9, 10 seeds × both arms):**
```
arm             max_age              decay              t50  censored
Proposed             15     0.0499+/-0.0000      162+/-20        10/10
Proposed             30     0.0499+/-0.0001      128+/-15        10/10
Proposed             60     0.0499+/-0.0000       97+/-5         10/10
Unconstrained        15     0.2413+/-0.0279      158+/-20         0/10
Unconstrained        30     0.2489+/-0.0505      128+/-16         0/10
Unconstrained        60     0.2371+/-0.0381       97+/-5          0/10
```
**Decay is insensitive to max_age. t50's absolute value is NOT (97→162 across the range) — but it shifts identically in both arms at every value, so the Proposed-vs-Unconstrained contrast (no differential delay) is robust to the choice.** Consequence for the paper: t50 absolutes are partly a property of the monitoring configuration and should not be over-read; the contrast is the claim.

**The four F-12 predictions, measured individually (A/B, eviction OFF = max_age 1e18 vs ON = 30, seeds 1000-1002):**

| prediction | OFF | ON |
|---|---|---|
| 1. max belief age bounded | 1925–1961 | **30.00 exactly** (after moving eviction ahead of the broadcast-thinning check — it first measured 35–37 because `broadcast_rate=0.5` strategies skipped half the eviction opportunities; placement fixed and re-verified) |
| 2. applied ε | 0.0138–0.0169 | **0.0168–0.0183** (~+20 %; the 2.9e-05 floor was already gone via MS-14 — this is the phantom-τ recovery on top) |
| 3. FRAGMENTED onset | mean t≈54–57, 100/100 agents | **t≈79–112, 92–100/100** — later, and via the density trigger. Pre-eviction detection was *early for the wrong reason*: phantom beliefs inflated `mean_staleness` past the 3× trigger. Detection now tracks actual isolation |
| 4. phantoms in coverage inputs | 50–58 % of believed neighbours dead | **16–21 %** — bounded (corpses linger ≤ max_age), not zero |

**F-04 re-measurement (author-flagged as headline-sensitive):**
```
late-phase (t>500) survivors:      OFF: halted 0-47%, empty maps 0%
                                   ON:  halted 100%,  empty maps 100%
```
⚠️ **F-04's §V-A characterisation was mechanically inaccurate for the pre-eviction system it was written about.** Pre-eviction, no late-phase map was ever empty; survivors converged *asymptotically toward frozen phantom centroids* (v → 0 exponentially, sometimes never below 1e-9 in-run — seed 1002's survivor never halted). Post-eviction, the prose ("belief buffer holds no live neighbours … commanded velocity identically zero") is **literally true**: 100 % empty maps, 100 % exact halts. The fixed point triggers *more cleanly and universally*, not more often in a new regime. §V-A needs a flag-level edit acknowledging both mechanisms, not a quiet word-swap.

**Full 50-seed suite (max_age=30) — ⚠️ TABLE III MOVED:**
```
[Unconstrained]  t50 127 +/- 5   decay 0.2606 +/- 0.0173   survival 407 +/- 29  (0/50 censored)
[Proposed]       t50 128 +/- 5   decay 0.0499 +/- 0.0000   50/50 censored
[True Oracle]    identical (no gossip path)
```
vs adopted: t50 98→127/128 in **both** arms (+30 %, contrast unchanged — still no differential delay); Unconstrained decay 0.289→0.261 (~1.3 CI); Proposed decay 0.051→0.0499 with **variance collapsing to ±0.0000**; ratio would become 5.2× (was 5.6×, was 4.5×).

⚠️ **The §V-C pursuit–halt tail finding (3/50 = 6 % total loss) DOES NOT REPRODUCE under eviction: 50/50 censored, zero total-loss runs.** The tension is real but its fatal manifestation was a property of the phantom-map configuration (phantom-fed centroids kept remnants creeping while pursuers drained them). §V-C's finding paragraph, the abstract's numbers, `tab:oracle_sensitivity`, and §V-D's "mean applied step size 0.016" (now ~0.018) all need coordinated edits **after review** — none made.

**MS-14 revisit (author-flagged):** with τ_max no longer phantom-inflated, mean applied ε rises only 0.016→~0.018 (≈⅓ of the 0.05 cap) and decay is bit-flat at 0.0499. **The "real consensus-layer property, negligible energy contribution" conclusion stands under recovered ε.**

Tests: 15 failed / 95 passed — identical pre-existing set, no regressions.

### 3.2 — Fix `coverage_completion_rate` or delete it `[F-18]`

`agent_core.py:194` counts ticks where the controller was nominally enabled — it reports 92 % for a random-walk arm and 100 % unconditionally in oracle mode. Either implement a real coverage measure (fraction of the grid within `comm_radius` of some live agent, sampled on the metrics tick) or delete the metric. Do not report the current one.

**Verify:** with `coverage_enabled=False`, the metric must read `0.0`. It currently reads >90 %.
**Evidence:** _(paste output)_

### 3.3 — Stop the multiprocess log race `[F-26]`

`run_monte_carlo_table.py` passes `test_mode="none"`/`"static_bounded"`, so `KernelLogger` opens `logs/experiment_none.csv` in mode `'w'` from 200 concurrent processes and never closes them.

- `kernel_logger.py:23-43`: return early (no file) when `test_mode` is not one of the three known experiment modes.
- `run_monte_carlo_table.py:run_single_sim`: call `sim.close_loggers()` before returning.
- Delete `logs/experiment_none.csv` and `logs/experiment_static_bounded.csv`.

**Verify:** `ls logs/experiment_none.csv` after a full MC run → `No such file`.

**Evidence:** ✅ DONE (pulled forward into the R0 performance work). Added `LOGGED_TEST_MODES = frozenset({"percolation","thermodynamics","stability"})` in `kernel_logger.py`; the constructor now gates on membership rather than truthiness, so `"none"` / `"static_bounded"` open no file.
```
--- junk log files present? ---
ls: cannot access 'logs/experiment_none.csv': No such file or directory
ls: cannot access 'logs/experiment_static_bounded.csv': No such file or directory
```
**Performance result: none.** The I/O-contention hypothesis was wrong — identical wall clock before and after (68.1 s both, 16 tasks / 16 workers). This was a correctness fix only. `sim.close_loggers()` in `run_single_sim` is still outstanding.

### 3.4 — Harmonise the τ_max discretisation `[F-24]`

`gossip_consensus.py:65` uses `math.ceil`; `agent_core.py:517` uses `int()` (floor) for the same bound. Pick one (`ceil` is the conservative choice) and use it in both.

**Verify:** unit-test both call sites against the same `(d_i, max_delay)` input and assert equal bounds.
**Evidence:** _(paste output)_

### 3.5 — Remove dead code that misleads readers

- `src/core/config.py:88` `comm_radius_max` — unreferenced; delete or wire up.
- `src/core/event.py:26` `AUCTION_TIMEOUT` — no handler, never scheduled; delete.
- `src/communication/rgg_builder.py` — used only by tests; either wire into `CommunicationEngine` (see `[MS-20]`) or mark clearly as unused.
- `src/agent/agent_core.py:238-243` — annotate the random-walk branch as the *baseline fallback controller*, since Table III's result depends on it (F-04).

---

## Phase 4 — Repair the test suite and the broken subsystems

Not blocking for the paper, but the abstract's footnote advertises this repository as open-source evidence. A reviewer who clones it finds 19 failing tests and a replay system that raises on import of any run.

### 4.1 — Get the suite green `[F-08]`

Current: `19 failed, 87 passed`, plus 23 files uncollectable (`ModuleNotFoundError: No module named 'PySide6'`).

1. `pip install -r requirements.txt` so the GUI half can be collected at all.
2. `tests/test_safety_projector.py` — 4 × `TypeError: missing 1 required positional argument: 'theta_nominal'`. Update to the current signature. **Algorithm 1 has no passing coverage until this is done** — fix it first.
3. `tests/test_packet_drop.py` — 4 failures caused by the path-loss term added to `should_drop`. Update the expected contract (note `test_zero_drop_never_drops` is now false by design — decide whether that is intended).
4. `tests/test_no_global_access.py` — fails on `'lambda_2'` matching `self._lambda_2_proxy`. Tighten the pattern to `global_lambda_2` / `true_lambda_2`, and **add a check for the real leak**: `oracle_centroid` being set externally at `simulation.py:232`.
5. `tests/test_regime_detection.py` (5), `test_hybrid_supervisor.py` (2), `test_auction.py::test_dropout_robustness`, `test_scalability_n100.py::test_drop_rate_within_tolerance` (measured 0.642 vs expected ≤0.55 — this one is reporting a *real* discrepancy; fix the model or the tolerance, deliberately).

**Verify:** `python -m pytest tests/ -q` → `0 failed`, `0 errors`.
**Evidence:** _(paste the summary line)_

### 4.2 — Replace `diagnostics_audit.py`'s vacuous checks `[F-16]`

Each of these must assert on a measured quantity or be deleted:
- **line 110** `log_result("Task Resolution", True, ...)` → assert `len(sim.auction_results) > 0` and that every completed task was within `r_task`.
- **line 131** `log_result("Stability Constraints", True, ...)` → assert `total_projections > 0` **and** that every final parameter lies inside `THETA_SAFE_BOUNDS`.
- **line 146** `log_result("Communication Overhead", True, ...)` → assert no delivery occurred between agents further apart than `tx_radius`, by instrumenting `comm_engine`.
- **line 168** relabel the wall-clock check as a performance benchmark, not "Stable under load".
- **line 88** Fog-of-War: replace the `hasattr` check for never-existent attributes with a real one (e.g. assert `agent.oracle_centroid is None` whenever `global_info_enabled` is False).
- **line 41** delete the unused `events1`.

**Verify:** deliberately break one invariant (e.g. force a bound violation) and confirm the corresponding check reports `FAILURE`. A check that cannot be made to fail is not a check.
**Evidence:** _(paste both the passing and the deliberately-failed run)_

### 4.3 — Fix replay `[F-25, F-31]`

1. Add `agent_states` to all three loader paths (`replay_loader.py:117-133, 153-166, 184-197`). Persist it in the exporter (`exporter.py:239-254` currently omits it).
2. Persist and restore the real adjacency instead of `np.zeros((n,n))`, or — if size is the concern — persist the edge list. Do **not** ship a zero adjacency alongside `connected_components=[list(range(n))]`; they contradict each other and make every replay report total collapse.
3. `exporter.py:128` — a silent `deque(maxlen=10000)` truncation. Either raise/warn on overflow or spill to disk.
4. `exporter.py:78` — stop popping `regime` from `config_snapshot`; serialise `RegimeConfig` so runs are reconstructible.

**Verify:**
```python
frames = ReplayLoader('outputs/run_.../').load()
print(len(frames), frames[0].adjacency.sum(), SpectralAnalyzer().analyze_frame(frames[0]).lambda2)
```
Must load without error and produce a λ₂ matching `frames[0].spectral_gap`. Today it raises `TypeError`.
**Evidence:** _(paste output)_

### 4.4 — Fix scenarios `[F-35, F-36]`

`scenario_model.py:94` — `dataclasses.asdict` flattens `RegimeConfig` to a `dict`, so `Phase1Simulation` raises `AttributeError: 'dict' object has no attribute 'window_size'`. Use `dataclasses.replace(base_cfg, ...)` instead of `asdict` + `SimConfig(**d)`.

Also wire `TaskParams.count`/`.distribution` into `to_sim_config`, or remove them from the UI.

**Verify:** `Phase1Simulation(ScenarioConfig().to_sim_config(SimConfig())).run()` completes without exception.
**Evidence:** _(paste output)_

### 4.5 — Fix analytics defects `[F-27..F-30]`

- `spectral_analyzer.py:59-67` and `percolation_analyzer.py:64-68` — include `drone_failure_flags` in the cache key; deaths currently do not invalidate the cache.
- `percolation_analyzer.py:124-158` — stop mutating and returning the cached object by reference; return a copy.
- `percolation_analyzer.py:127` — initialise `self._last_time = -1.0` in `__init__` instead of the `hasattr` guard.
- `research_metrics.py:198-200` — aggregate `health.raw_score`, not the EMA-smoothed `health_score`.
- `research_metrics.py:146-152` — `time_to_stability` currently records the moment stability was *lost*. Rename or reimplement.
- `swarm_health.py:100` — resolve the sign confusion the comment admits to.
- `energy_cascade_analyzer.py:124-125` — take the normalisation baseline from `config.energy_initial`, not from the first frame observed.
- `main_window.py:394-400` — `reset_analyzers()` replaces analyzer objects, permanently orphaning every signal connection made in `_connect_signals`. Either reset in place (add `.reset()` methods) or re-run the connection wiring afterwards.
- `main_window.py:531` — display `raw_score`, not the smoothed value, in the status bar.
- `telemetry_frame.py:98` — `empty()` marks all drones dead; should be `np.zeros(n, dtype=bool)`.
- `telemetry_emitter.py:77-88` — apply the same alive-filtering fix as 1.1.

**Verify:** targeted unit test per item; specifically, a frame sequence where adjacency is constant but agents die must produce a **changing** λ₂ and connectivity ratio.
**Evidence:** _(paste output)_

### 4.6 — `[MS-27]` Fix the README overclaim

`README.md:13` — "mathematically **guaranteeing** they do not induce kinematic oscillations". No such proof exists, and the `.tex` explicitly disclaims mathematical novelty. Align the README to the paper's own hedging ("heuristic relaxations", "empirically bounded").

---

## Phase 5 — Final consistency gate

Do not submit until every box is ticked with pasted evidence.

- [ ] **`grep -c "PENDING REGENERATION" manuscript/final_manuscript.tex` → `0`.** Nothing compiles for submission while that block is present. It sits above Table III precisely so a stale table cannot look authoritative; it is removed only when 1.2/1.3 have regenerated the numbers and pasted the runner output here.
- [ ] `grep -c "No such issues were identified" manuscript/final_manuscript.tex` → `0`
- [ ] All three `manuscript/fig*.png` md5-match their regenerated `plots/` source
- [ ] Every number in Table III traces to a line of `run_monte_carlo_table.py` output pasted in this file
- [ ] Every number in §V-A/B/C/D traces to a named script, its N, and its seed count
- [ ] Algorithm 1, 2, 3 pseudocode diffed line-by-line against `safety_projector.py`, `auction.py`, `voronoi_coverage.py`
- [ ] Equations (1)-(7) each checked against the implementing function
- [ ] Every figure caption names the quantity actually plotted
- [ ] No figure annotation refers to a mechanism absent from `src/`
- [ ] `python -m pytest tests/ -q` → 0 failed, 0 errors
- [ ] `python diagnostics_audit.py` → no check that is incapable of failing
- [ ] `draft 11.pdf` recompiled from the corrected `.tex`, or deleted from the repo so the stale version cannot be mistaken for current
- [ ] ORCID `0000-0001-9876-5432` verified as real
- [ ] Author list reconciled between the PDF (3 authors) and the `.tex` (4 authors)

**Only when every box above is ticked may `rust_conversion_plan.md` begin.** Porting the current code to Rust would carry every defect above into a second implementation and make them harder to find.
