# Rust Conversion Plan

> **DO NOT START THIS DOCUMENT.**
> Every phase of `fixes_phases.md` must be complete **and** its Phase 5 checklist ticked with pasted evidence before sub-phase R0 begins.
>
> Rationale is concrete, not procedural. `audit_findings.md` documents ~38 defects, of which at least six change reported results (F-01 through F-06). Porting today means: re-implementing λ₂-over-corpses (F-03) in a second language; carrying the auction/Algorithm-2 divergence (F-14) into a new codebase; and — worst — making the cross-validation step of this very plan **useless**, because "Rust matches Python" would only prove the Rust port faithfully reproduces known-wrong behaviour. Cross-validation against a broken reference validates nothing.

**Motivation from the paper:** §VI Future Scope commits to "memory-safe, compiled environments (e.g., Rust or C++) to overcome Python's GIL and enable massive scaling (N > 10,000 agents)".

**Reality check on that claim before committing effort:** the current bottleneck is *not* the GIL. It is algorithmic — `CommunicationEngine.process_broadcasts` is an O(N) linear scan run once per agent per tick, i.e. **O(N²) per tick** (F-22), and `compute_spectral_gap` is a dense `O(N³)` eigendecomposition on the metrics tick. At N=10,000 the per-tick communication cost alone is 10⁸ distance evaluations. **Fixing the algorithm in Python (KD-Tree, sparse `eigsh`) will buy more than a language port.** Do that measurement first — it is R0 below — and be prepared for the honest answer that the Rust port is not the highest-value next step.

---

## Sub-phase R0 — Justify the port before writing Rust

**Goal:** establish, with numbers, that Rust is the right lever.

1. Profile the fixed Python (`cProfile` / `py-spy`) at N = 100, 500, 1000. Attribute wall-clock to: distance scans, KD-Tree, eigendecomposition, heap operations, per-agent Python object overhead.
2. Apply the two cheap algorithmic fixes in Python — wire `RGGBuilder` KD-Tree into `CommunicationEngine`, switch to sparse `scipy.sparse.linalg.eigsh` for λ₂ — and re-profile.
3. Extrapolate to N = 10,000 and record the measured scaling exponent.

**Deliverable:** a table of wall-clock vs N, before and after the algorithmic fixes, with the residual fraction of time spent in interpreter overhead. That residual is the *actual* ceiling a Rust port can lift.

**Decision gate:** if interpreter overhead is < 40 % of post-fix runtime, the port does not pay for itself and this document should be closed with that finding recorded. Say so if that is the answer.

**Crates:** none.
**Estimate:** **3-5 days.**

---

### R0 RESULTS — executed 2026-08-16, pre-fix baseline

**Hardware:** AMD Ryzen 7 5800HS — **8 physical cores, 16 logical**.

**Suite cost (the number that motivated the port):**
```
Unconstrained    wall=  24.94s  events=  111727
Static-Epsilon   wall=  39.28s  events=  136935
Proposed         wall=  39.55s  events=  136935
True Oracle      wall=  13.52s  events=   35782

one seed, all 4 arms  = 117.3s serial
50 seeds x 4 arms     = 5865s = 97.7 min serial
```
Measured under `ProcessPoolExecutor` (16 tasks, 16 workers):
```
wall=68.1s  speedup=6.89x  extrapolated 200-task suite = 14.2 min
```
With BLAS threads pinned (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`):
```
wall=61.1s  speedup=7.68x  extrapolated suite=12.7 min
```

**The suite takes ~13 minutes, not hours.** The "hours" figure predates the current runner: `mc_results.txt` records sequential `Run 1/50 … Run 50/50` output from a serial loop, whereas the current `run_monte_carlo_table.py` uses `ProcessPoolExecutor` and prints `Progress: N/200 tasks complete`. The parallelisation already landed.

**Parallelism is already saturated.** 7.68× against **8 physical cores** is ~96 % of the achievable ceiling; the earlier "43 % utilisation" figure used 16 *logical* cores as the denominator, which is the wrong baseline for a NumPy-bound workload. There is no meaningful parallelism win left — only the free ~10 % from pinning BLAS threads.

**Scaling (max_time=300, single process):**
```
     N    wall_s     events   us/event  alive_end
    50      7.11      44870     158.43          5
   100     16.55      89097     185.76          5
   200     51.98     194304     267.52          5
   400    148.57     384592     386.29          4

empirical exponent between N=200 and N=400:  wall ~ N^1.52
```
⚠️ **This exponent is measured but UNSOUND, and so is its post-fix counterpart (N^1.04).** The quadratic term is truncated because the swarm dies early — `alive_end` is 4-5 regardless of N, so the system is only briefly loaded with N agents and spends most of the run simulating a handful of survivors. **Neither N^1.52 nor N^1.04 is a defensible scaling result**, and "scaling is now near-linear" must not appear in the paper or in any Phase 2 scalability text (see `[MS-29]` in `fixes_phases.md`).

A real scaling measurement requires runs where the swarm survives long enough for N to actually load the system — raised `energy_initial`, or a short horizon chosen for high survival. Until that is run, the honest statement is that **scaling is unmeasured at survival-preserving parameters**, and the §III-C complexity claim rests on code inspection rather than measurement.

**cProfile, N=200, max_time=300 — top frames by cumulative:**
```
        1    0.831    0.831  117.578  117.578 src\core\kernel.py:90(run)
   194304    0.480    0.000  109.515    0.001 src\core\kernel.py:122(_dispatch)
    22362    0.656    0.000   64.619    0.003 src\simulation.py:277(_handle_msg_transmit)
    69417    0.557    0.000   41.811    0.001 {built-in method builtins.sum}
   310314   15.418    0.000   39.139    0.000 src\simulation.py:331(<genexpr>)
  3305898   17.795    0.000   35.741    0.000 numpy\linalg\_linalg.py:2598(norm)
    22557    0.418    0.000   30.172    0.001 src\simulation.py:224(_handle_kinematic_update)
    22362    0.731    0.000   27.575    0.001 src\agent\agent_core.py:183(compute_velocity)
    22054    8.318    0.000   26.058    0.001 src\coordination\voronoi_coverage.py:16(compute_local_centroid)
    17464    0.539    0.000   12.846    0.001 src\simulation.py:680(_get_all_positions)
    17464    2.563    0.000   10.235    0.001 src\simulation.py:682(<listcomp>)
    17404    4.274    0.000    8.940    0.001 src\communication\comm_engine.py:61(process_broadcasts)
  3505108    3.280    0.000    7.697    0.000 src\agent\agent_core.py:161(position)
  2316998    3.306    0.000    7.111    0.000 src\core\event.py:94(__lt__)
   194304    1.200    0.000    7.096    0.000 {built-in method _heapq.heappop}
  2786478    6.336    0.000    6.336    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

**Where the time actually goes — and it is not where we assumed:**

| Rank | Cost | Share | Assumed? |
|---|---|---|---|
| 1 | `simulation.py:331-337` — the **duplicate** neighbour count, a Python genexpr calling scalar `np.linalg.norm` **3,305,898** times: 15.4 s in the genexpr + 17.8 s in `norm` ≈ **33 s / 117 s** | **~28 %** | **No** — not on the fix list |
| 2 | `_get_all_positions()` — 17,464 calls, each rebuilding a list of N `.copy()`s → 3.5 M property calls | **~11 %** | **No** |
| 3 | `compute_local_centroid` (15×15 grid scan) | ~7 % | No |
| 4 | Event heap (`__lt__`, `_sort_key`, `heappop`) | ~7 % | No |
| 5 | `comm_engine.py:107` scan — **already vectorised** (`norm(..., axis=1)`), 4.27 s tottime | **~3.7 %** | Yes — but far smaller than assumed |
| 6 | Eigendecomposition for λ₂ | **~2.2 %** | Yes — **negligible; sparse `eigsh` is not a speed win** |

`_handle_msg_transmit` accounts for **64.6 s of 117.6 s cumulative (55 %)**, almost entirely from the duplicate scan at line 331.

```
process_broadcasts calls (each = one O(N) scan in comm_engine.py:107) = 17404
simulation.py:331-337 runs a SECOND O(N) scan per surviving call
=> ~34808 O(N) scans over the run at N=200
```

**Revised fix priority (supersedes the original step-2 list):**

1. **Eliminate the duplicate scan.** `process_broadcasts` already computes `distances` vectorised at `comm_engine.py:107`; have it return the in-radius neighbour count and delete `simulation.py:331-337`. Removes ~28 % of runtime and 3.3 M NumPy calls. **Highest value, lowest risk, no behavioural change.**
2. **Cache positions per tick.** `_get_all_positions()` rebuilds an N-element array of copies 17k times. ~11 %.
3. **Pin BLAS threads** in the MC runner. Free ~10 % on the suite.
4. **KD-tree into `CommunicationEngine`** — see the hazard note below. Worth doing for **claim alignment (`MS-20`), not for speed**: it replaces a 3.7 % cost.
5. **Sparse `eigsh`** — **do not do this for performance** (2.2 %). Defer to Phase 1.1, where the driver is correctness (dead-agent filtering, `eigvalsh`), and revisit only if N grows.

**KD-tree hazard (must be handled, blocks a naive swap):** `RGGBuilder` is constructed with a **fixed** radius (`rgg_builder.py:33`, `self._radius = comm_radius`), but each sender transmits at a **variable** radius — `tx_radius = comm_radius * tx_power_scale`, with `tx_power_scale ∈ [1.0, 2.0]` per `THETA_SAFE_BOUNDS`. Swapping in `build_neighbor_lists` as written would silently discard dynamic transmission power, a Phase-1 feature. The correct form is a per-tick tree with per-sender `query_ball_point(sender_pos, sender_tx_radius)`, plus care that the tree is rebuilt when positions change mid-tick (broadcasts and kinematic updates interleave in the event queue).

**Bearing on the Rust decision:** interpreter and NumPy-call overhead dominate ("everything else" ≈ 64 % of `tottime`, largely per-call NumPy dispatch on 2-element arrays), so a port would win a real constant factor. But the suite is already ~13 min and near core saturation, and fixes 1-3 above are expected to cut it substantially further in Python for a fraction of the effort. **This supports keeping the Rust port post-submission.** Re-run this profile after fixes 1-3 land and re-evaluate against the decision gate.

---

## Sub-phase R1 — Core kernel and event system

**Ports:** `src/core/{event,kernel,clock,config}.py`

| Python | Rust | Nature |
|---|---|---|
| `EventType(IntEnum)` | `#[repr(u8)] enum EventType` | mechanical |
| `Event` with `__lt__` on `(timestamp, type, seq)` | `struct Event` + `impl Ord`, wrapped in `Reverse` for `BinaryHeap` | mechanical, **but see f64 note** |
| `heapq` min-heap | `std::collections::BinaryHeap<Reverse<Event>>` | mechanical |
| `payload: Any` | **redesign** — `enum EventPayload { Message(Message), Auction{ task_id: TaskId, pos: Vec2, reward: f64 }, TaskSpawn{ idx: u32 }, None }` | **redesign** |
| `register_handler(type, callable)` | match on `EventType` in `dispatch`, or `HashMap<EventType, Box<dyn Fn>>` | prefer the match — faster and exhaustiveness-checked |
| `SimConfig` frozen dataclass | `#[derive(Clone, Serialize, Deserialize)] struct SimConfig` | mechanical |

**The genuine redesign is `payload: Any`.** Python passes arbitrary objects through `Event.payload`; the code relies on this in `_handle_task_spawn` (dict), `_handle_auction_resolve` (dict), and `_handle_msg_deliver` (a `Message` object). A Rust enum forces every payload shape to be declared. This is a net win but touches every handler.

**Ordering hazard — must be handled explicitly.** `Event.__lt__` sorts on an `f64` timestamp. Rust's `f64` is not `Ord`. Do **not** reach for `partial_cmp().unwrap()` — it will panic on a NaN and silently reorder on `-0.0`. Use `ordered_float::NotNan<f64>` for the timestamp, or store time as fixed-point integer ticks. Given the whole simulation runs at `dt = 1.0` (`config.py:92`) with only latency producing fractional times, **fixed-point integer time (e.g. microticks as `u64`) is the better choice** and removes a class of determinism bug outright.

**Crates:** `ordered-float` (if staying with f64), `serde` + `serde_json`, `thiserror`.

**Cross-validation (required before R2):**
Instrument both implementations to dump `(timestamp, event_type, agent_id, sequence_id)` for the first 100,000 dispatched events, on seeds 1000/1001/1002 at N=100. Diff must be **byte-identical**. A single reordered tie means the tie-break rule diverged, and every downstream comparison is void.

**Estimate:** **1 week.**

---

## Sub-phase R2 — RNG and determinism substrate

**Ports:** `SimConfig.spawn_rng_streams` (`config.py:138-166`)

This is the hardest correctness problem in the whole port, and it deserves its own sub-phase.

The Python implementation uses `np.random.SeedSequence(seed).spawn(4 + N)` feeding `np.random.default_rng` (PCG64). **There is no Rust crate that reproduces NumPy's `SeedSequence` spawn tree and PCG64 stream bit-for-bit out of the box.** Two options:

- **(a) Reimplement NumPy's `SeedSequence`** (the SHA-ish entropy mixer and pool-based `spawn`) plus PCG64 with NumPy's exact multiplier/increment and its exact `standard_normal` (ziggurat) and `exponential` implementations. This is the only path to bit-identical cross-validation on stochastic quantities. It is real work and easy to get subtly wrong.
- **(b) Accept a different RNG** (`rand_pcg::Pcg64` + `rand_distr`) and abandon bit-identical validation, falling back to **distributional** cross-validation: run 50+ seeds in each implementation and compare metric distributions with a two-sample Kolmogorov–Smirnov test.

**Recommendation: (a) for the uniform stream, (b) tolerated for the derived distributions.** Uniform draws (`packet_drop`, `random`) are cheap to match exactly and cover most of the stochasticity. `standard_normal` and `exponential` are where NumPy-specific algorithms bite; matching those is where the effort explodes.

Whichever is chosen, **write it down in the paper**. If the Rust results come from a different RNG, they are not "exact replay" and Contribution 4 does not extend to them.

**Crates:** `rand`, `rand_pcg`, `rand_distr`; possibly a hand-written `numpy_compat_rng` module.

**Cross-validation:** dump the first 10,000 draws from each named stream (`positions`, `packet_drop`, `latency`, `task_spawner`, `agent_0..agent_9`) in both implementations. Under (a): identical to the last bit. Under (b): KS test p > 0.01 per stream, plus documented non-identity.

**Estimate:** **1.5-3 weeks** for (a); **3 days** for (b). This variance is the single largest scheduling risk in the plan — resolve the choice at the start of R2, not during it.

---

## Sub-phase R3 — Environment and communication layer

**Ports:** `src/environment/{spatial_grid,interference_field}.py`, `src/communication/{rgg_builder,packet_drop,latency_model,comm_engine,message}.py`

| Python | Rust | Nature |
|---|---|---|
| `SpatialGrid.clamp_position` | `Vec2::clamp` | mechanical |
| `InterferenceField` (3 modes) | `enum FieldMode` + `fn evaluate` | mechanical |
| `PacketDropSampler.should_drop` | direct port of `p_survive = (1-p)(1-ψ)max(0, 1-(d/R)²)` | mechanical |
| `LatencyModel` `Exp(mean) + tau_min` | `rand_distr::Exp` | mechanical, RNG-sensitive (R2) |
| `scipy.spatial.KDTree.query_pairs` | `kiddo` or `rstar` | **behavioural risk — see below** |
| `CommunicationEngine.process_broadcasts` | port | mechanical, but **fix the O(N²) here** |

**KD-Tree tie-breaking is a real hazard.** `query_pairs(r)` returns pairs at distance **exactly** `r` inclusively; different libraries differ at the boundary and in iteration order. Since edge presence feeds `should_drop` and thus RNG consumption order, a single boundary disagreement desynchronises every downstream draw. Mitigate by: (i) using `<=` explicitly and validating against a brute-force O(N²) reference on random point sets including deliberately co-radial points; (ii) sorting the returned pair list canonically before use.

Note that `comm_engine.py` currently does **not** use the KD-Tree at all (F-22). The Rust port should use it — but then Python and Rust take different code paths and cross-validation will diverge. **Fix Python first (`fixes_phases.md` MS-20) so both sides use the KD-Tree**, then port.

**Crates:** `kiddo` (fastest) or `rstar`, `glam` or `nalgebra` for `Vec2`, `rayon` (deferred to R7).

**Cross-validation:** for 1000 random configurations at N ∈ {50, 100, 500}, assert Rust and Python produce identical sorted neighbour-pair lists. Then assert identical drop/keep decisions given the same RNG stream (requires R2(a)).

**Estimate:** **1.5 weeks.**

---

## Sub-phase R4 — Agent logic

**Ports:** `src/agent/{agent_core,energy_model,local_map}.py`, `src/coordination/{gossip_consensus,voronoi_coverage,auction}.py`

The largest sub-phase and the one with the most redesign.

| Python | Rust | Nature |
|---|---|---|
| `EnergyModel` | `struct EnergyModel` | mechanical; make `consume` return `Result` instead of `raise RuntimeError` |
| `LocalMap._beliefs: dict[int, NeighborBelief]` | `HashMap<AgentId, NeighborBelief>` — **use `BTreeMap`** | **redesign, see below** |
| `compute_local_centroid` (15×15 grid scan) | direct port | mechanical; vectorise later |
| `compute_gossip_update` | direct port | mechanical |
| `compute_bid` / `resolve_local_winner` | direct port | mechanical — **only after `fixes_phases.md` 2.3 lands** |
| `theta` as `dict[str, float]` | `struct ThetaParams { coverage_gain: f64, gossip_epsilon: f64, ... }` | **redesign** |
| `AgentCore` (25+ mutable fields) | `struct Agent` | mechanical, but borrow-checker work at the call sites |

**Redesign 1 — `HashMap` iteration order.** `LocalMap.get_all_neighbors()` returns `list(self._beliefs.values())`, and Python dicts preserve **insertion order**. That order determines the summation order in `compute_gossip_update` (`sum((x_j - own_state) for x_j in neighbor_states)`), and floating-point addition is not associative. Rust's `HashMap` iteration order is **randomised per process**. Using `HashMap` will make the Rust build non-deterministic across runs. **Use `BTreeMap<AgentId, NeighborBelief>` or an insertion-ordered `IndexMap`.** If bit-identical validation against Python is required, `IndexMap` (insertion order) is the faithful choice; `BTreeMap` (id order) is more defensible but will produce different low-order bits and needs Python changed to match.

This is exactly the kind of silent divergence the cross-validation gates exist to catch.

**Redesign 2 — dict-based parameters.** `theta_proposed`/`theta_safe` are `dict[str, float]` threaded through `HybridSupervisor.propose_parameters` → `project_to_theta_safe` → `smooth_update`. A `ThetaParams` struct with named fields is the right Rust shape and eliminates the `theta_safe.get(key, default)` fallbacks at `agent_core.py:540-545` that silently preserve stale values on a missing key. Keep `THETA_SAFE_BOUNDS` as a `const [(f64, f64); 6]` indexed by a `ParamId` enum so the box clamp is exhaustive.

**Do not port `agent_core.py:238-243` (the random-walk branch) without a decision.** Per F-04 it is the mechanism behind Table III's headline number and is labelled a placeholder. Resolve its status in Python first.

**Crates:** `indexmap` or `std::collections::BTreeMap`, `glam`.

**Cross-validation:** deterministic replay at N=100 for 2000 ticks, seeds 1000-1004. Compare per-agent `position` (both components), `energy`, `consensus_state`, and all six `theta` parameters at every metrics tick. Tolerance: **exact equality** if R2(a) succeeded; otherwise `|Δ| < 1e-9` relative, with the divergence point recorded.

**Estimate:** **3 weeks.**

---

## Sub-phase R5 — Regime detection and adaptation

**Ports:** `src/regime/{classifier,local_proxies,telemetry_buffer}.py`, `src/adaptation/{hybrid_supervisor,safety_projector,stability_tuner}.py`

| Python | Rust | Nature |
|---|---|---|
| `Regime` / `Strategy` enums | `enum Regime` / `enum Strategy` | mechanical |
| `RegimeClassifier.classify` (ordered if-chain) | direct port | mechanical — preserve branch order exactly; it is load-bearing (F-10) |
| `TelemetryBuffer` (`deque(maxlen=w)`) | `VecDeque` with manual cap, or `ringbuf` | mechanical |
| `statistics.mean` / `statistics.variance` | **redesign** | see below |
| `project_to_theta_safe` | direct port over `ThetaParams` | mechanical |
| `smooth_update` (EMA) | direct port | mechanical |

**Redesign — variance algorithm.** Python's `statistics.variance` uses exact-fraction arithmetic internally and is the **sample** variance (`n-1` denominator). A naive Rust `sum((x-mean)²)/(n-1)` differs in low-order bits, and `n` vs `n-1` differs materially at small neighbourhoods (`compute_local_consensus_variance` frequently runs on 2-5 elements). Use Welford's algorithm and **assert the `n-1` convention explicitly** — an off-by-one here shifts `mean_variance`, which shifts regime classification, which shifts everything.

Port `dwell_time` enforcement only if `fixes_phases.md` MS-21 chose to implement it; do not port the current non-enforcement and then describe it as enforced.

**Crates:** none beyond std.

**Cross-validation:** feed 10,000 recorded `TelemetrySnapshot` sequences (dumped from the Python run) into both classifiers and assert an identical `Regime` sequence. Then assert identical `ThetaParams` out of `project_to_theta_safe` for 10,000 recorded proposals. This is a pure-function comparison and needs no RNG match — **run it early**, it is the cheapest high-value gate in the plan.

**Estimate:** **1 week.**

---

## Sub-phase R6 — Orchestrator, metrics, and the oracle path

**Ports:** `src/simulation.py`, `src/metrics/*`

| Python | Rust | Nature |
|---|---|---|
| `Phase1Simulation` handler registration | `impl Simulation { fn dispatch(&mut self, e: Event) }` with a `match` | mechanical |
| `alive_mask: np.ndarray[bool]` | `Vec<bool>` or `bitvec` | mechanical |
| `compute_connectivity_metrics` (BFS + eigendecomposition) | `petgraph` + `nalgebra`/`faer` | **see below** |
| `KernelLogger` CSV | `csv` crate | mechanical |
| `_compute_oracle_centroid` (`scipy.spatial.Voronoi`) | **problem — see below** |
| `linear_sum_assignment` (Hungarian) | `pathfinding::kuhn_munkres` or `lapjv` | mechanical |

**Borrow-checker note.** `Phase1Simulation` handlers mutate `self.agents[id]` while reading `self.alive_mask` and pushing to `self.kernel`. In Rust this needs either split borrows (`let Simulation { agents, kernel, .. } = self;`) or an index-passing design. Plan for a restructure of the handler signatures — this is the most common place a naive port stalls.

**Voronoi is the hard dependency.** `scipy.spatial.Voronoi` (Qhull) has no mature pure-Rust equivalent. Options:
- **(a)** `voronator` / `spade` (Delaunay → dual). Different degenerate-case handling than Qhull; the boundary-point trick at `simulation.py:692` and the `-1 in region` check at line 705 will need re-derivation.
- **(b)** FFI to Qhull directly — preserves behaviour, adds a C dependency and undermines "memory-safe" as a motivation.
- **(c)** Do not port the oracle path. It exists only as an experimental upper bound and is not part of the deployable agent logic.

**Recommend (c) for the first port,** with (a) as follow-up. Note the consequence: the Rust build cannot reproduce the `True Oracle` arm, so that row of Table III stays Python-only. State this rather than quietly omitting the arm.

**λ₂ at scale:** dense `eigvalsh` is O(N³) — unusable at N=10,000. Use `faer` or `nalgebra-lapack` with a sparse/shift-invert Lanczos for the two smallest eigenvalues, mirroring `spectral_analyzer.py:142`. Validate against the dense result at N ≤ 500 before trusting it above.

**Crates:** `petgraph`, `faer` or `nalgebra`, `csv`, `bitvec`, `pathfinding`.

**Cross-validation:** full-run comparison at N=100, `max_time=2000`, seeds 1000-1009. Compare the complete `connectivity_log` and `adaptation_log` series element-wise, plus final `summary()`. λ₂ tolerance `1e-10` absolute (different eigensolvers); everything else exact under R2(a).

**Estimate:** **2.5 weeks.**

---

## Sub-phase R7 — Parallelism and scaling to N > 10,000

Only after R1-R6 cross-validate.

- Parallelise the per-agent kinematic and regime updates with `rayon`. **This is where determinism dies if you are careless:** parallel reduction reorders floating-point summation. Keep all reductions sequential (or use a deterministic tree reduction with fixed arity), and parallelise only genuinely independent per-agent work.
- Consider ECS layout (`hecs`, or hand-rolled struct-of-arrays) — `Vec<f64>` per field rather than `Vec<Agent>` — for cache efficiency at N=10,000.
- The event queue is the serial bottleneck. A single `BinaryHeap` at N=10,000 with ~4 events/agent/tick is 40,000 pushes per tick. Investigate a calendar queue or bucketed time-wheel before assuming the heap scales.
- Re-run R0's profile at N = 1,000 / 5,000 / 10,000 and report the measured scaling exponent against the O(N log N) claim in §III-C.

**Crates:** `rayon`, `hecs`, `criterion` for benchmarks.

**Cross-validation:** the parallel build must reproduce the serial Rust build **exactly** at N=100 across 10 seeds. If it does not, the parallelisation is wrong — do not accept "close enough" here, because the paper's Contribution 4 is exact reproducibility.

**Estimate:** **2-3 weeks.**

---

## Sub-phase R8 — Cross-validation harness (build this in R1, use it throughout)

Not a final step. Stand this up during R1 and extend it each sub-phase.

**Design:**
1. A shared JSON fixture format: `{seed, config, expected: {events[], per_tick_state[], summary{}}}`.
2. A Python dumper (`experiments/dump_reference.py`) that emits fixtures for a fixed seed list.
3. A Rust integration test that replays each fixture and diffs.
4. CI runs both on every commit.

**Non-negotiable gates:**

| Gate | Requirement |
|---|---|
| Event ordering | byte-identical dispatch sequence, first 100k events |
| RNG streams | bit-identical (R2a) or KS p>0.01 with non-identity documented (R2b) |
| Pure functions (classifier, projector, bid, centroid) | exact equality on 10k recorded inputs |
| Full run, N=100 | per-agent position/energy exact; λ₂ within 1e-10 |
| Scaling | measured exponent reported, not assumed |

**No Rust number goes into any paper, README, or presentation until its gate passes and the evidence is pasted into this document.** A Rust result that has not cleared its gate is an unvalidated result, regardless of how plausible it looks.

**Estimate:** **1 week** initial, ~2 days maintenance per sub-phase.

---

## Schedule summary

| Sub-phase | Scope | Estimate |
|---|---|---|
| R0 | Profile and justify the port | 3-5 days |
| R1 | Kernel and event system | 1 week |
| R2 | RNG and determinism | **3 days - 3 weeks** (decision-dependent) |
| R3 | Environment and communication | 1.5 weeks |
| R4 | Agent logic and coordination | 3 weeks |
| R5 | Regime detection and adaptation | 1 week |
| R6 | Orchestrator, metrics, oracle | 2.5 weeks |
| R7 | Parallelism and N>10,000 | 2-3 weeks |
| R8 | Cross-validation harness | 1 week + 2 days/phase |

**Total: 12-17 weeks** of focused work for one engineer, dominated by the R2 RNG decision and R4.

This is substantially more than "translate the Python". The honest framing for the paper's Future Scope section is that the port is a multi-month engineering effort whose main risks are RNG reproducibility and Voronoi/KD-Tree behavioural equivalence — not the GIL.

---

## Things that must **not** be carried into Rust

Cross-reference `audit_findings.md`. If any of these are still present in Python when the port starts, stop and finish `fixes_phases.md`.

- **F-03** λ₂ computed over dead agents — do not port `_get_all_positions()` into the metrics path unfiltered.
- **F-12** `LocalMap` with no eviction — unbounded growth is far more damaging in Rust at N=10,000 (memory, not just correctness).
- **F-17** `compute_velocity` mutating counters and consuming RNG from a logging path — this will make the Rust build non-reproducible and will be blamed on the port.
- **F-19** neighbour count logged as `avg_local_lambda_proxy` — port the metric under its true name or not at all.
- **F-02** `test_mode == "static_bounded"` as a distinguishing flag — it distinguishes nothing; do not encode it as a variant in a Rust enum.
- **F-06** `run_mc_fast.py`'s fabricated survival — do not port this script at all.
- **F-24** two different `τ_max` discretisations (`ceil` vs `int`) — Rust will not let you leave that ambiguous; resolve it in Python first.
