# Status Snapshot — end of session, 2026-08-16

**For the next session.** Companion documents: `audit_findings.md` (findings F-01..F-39),
`fixes_phases.md` (plan + pasted evidence per item), `rust_conversion_plan.md` (R0 results; port
is post-submission). Repo head at time of writing: `c7aee7a` on `main`, pushed.

---

## Where the paper stands

All numbers below are the **adopted, current** Table III (50 seeds, post-auction-correction):

| Arm | t50 (ticks) | Decay (units/tick) | Survival (ticks) |
|---|---|---|---|
| Unconstrained RGG | 98 ± 2 | 0.289 ± 0.022 | 373 ± 31 |
| Oracle (all-to-all cost) | 21 ± 0 | 2.801 ± 0.207 | 39 ± 4 |
| **Proposed** | **98 ± 2** | **0.051 ± 0.002** | **> 2000 (47/50 censored)** |

**Surviving headline:** 5.6× lower energy decay, cleanly separated CIs. §V-C states explicitly
that the growth of this ratio across revisions (4.5× → 5.6×) is a *baseline correction, not an
improvement* — the broken auction had been silently undercharging the baseline.

**Findings (new, positive):**
- Oracle cost decomposition: centralized coordination ≈ 2× swarm half-life (t50 199 vs 97),
  entirely consumed by all-to-all bandwidth cost under this energy model (`tab:oracle_sensitivity`).
- F-04 Voronoi stationary fixed point under isolation — the real mechanism behind the energy result.
- **Pursuit–halt tension (§V-C):** task pursuit overrides the fixed-point halt; in 3/50 runs (6%)
  that opposition produces total swarm loss. Reported as a finding qualifying the headline.
- ε-bound decomposition (§V-D): real consensus-layer stability property (unbounded divergence
  without it — 1e39 in 600 ticks, 1e154 on one seed), negligible energy contribution.

**Refuted, stated in-text:** no attrition delay (t50 98 vs 98); oracle-matching; sustained
network connectivity (λ₂ column dropped — confounded at every conditioning); the +0.04 proxy
bias (real figure: +1.81, scale mismatch, `/100` concealment — withdrawn in §V-B).

## What is done

- **Phase 0** (fabrications), **0.6** (oracle billing + perf: suite 98 min → ~4 min), **Phase 1**
  (1.1–1.5: metrics measure what they claim) — all with pasted evidence.
- **Phase 2 largely done:** MS-13 (Algorithm 1 two-stage pseudocode, 8/8 tests green),
  MS-14 option (b) (one bound, one owner; internal clamp removed; ceil discretisation — F-24 dead),
  MS-15 (Algorithm 2 pseudocode = code), MS-16 (§V-C rewrite), MS-20 (KD-tree per-sender
  `query_ball_point`, bit-identical, §III-C updated), MS-21 (dwell time → polling interval),
  MS-22, MS-23 (per-experiment methodology table), MS-25, MS-30 (ψ clamped; blackout marked),
  MS-02..MS-09 (Table III rebuild), MS-01/03 (abstract).
- Auction (2.3/2.4): energy-aware min-cost bid, single-commitment, expiry, **round-robin gossip**
  (recency starves concurrent auctions — measured; "soonest-resolving" fails symmetrically).
- Tests: 15 failed / 95 passed, all pre-existing failures (GUI-blocked files aside).

## Open items, in intended order

1. **Phase 3.1 — LocalMap eviction (F-12). NEXT, fresh session.** Behaviour change that can
   reopen Phase 1 numbers (staleness → τ_max → ε; FRAGMENTED onset; phantom-neighbour Voronoi
   targets). Sequence like MS-14/Algorithm-2: code → verify → full suite → flag if Table III
   moves → only then manuscript (`[MS-26]`).
2. **MS-17 Eq. (1)** — `R̃_tx = R_tx − ω_env` still printed (2 hits); replace with the actual
   three-factor survival model. **MS-18 Eq. (2) guard** — "fully-fragmented proxy scalar" still
   printed (1 hit); code returns a *maximal* proxy. **MS-19 Eq. (3)** — `γ_c·R_tx²` still printed;
   code charges `γ_c·(R_tx/R_base)²`. All three are §IV text-only edits, no re-runs needed.
3. **MS-24** — §V-D "±0.3 oscillations" claim unsubstantiated; Fig. 5 annotations were removed,
   §V-D prose partially rewritten; re-check the remaining sentence against regenerated Fig. 5.
4. **MS-27** — README overclaim ("mathematically guaranteeing"). **MS-28** — Intro line 60
   qualifier. Quick edits.
5. **Phase 3.2/3.5** — `coverage_completion_rate` still a labelled-broken diagnostic; dead config
   (`comm_radius_max`, `AUCTION_TIMEOUT` enum) still present.
6. **Phase 4** — remaining test repairs (packet-drop contract, regime-detection, replay F-25/F-31,
   scenario F-35, analytics F-27..F-30, GUI F-37; `diagnostics_audit.py` vacuous checks 4.2).
7. **Phase 5 gate**, then recompile the PDF (draft 11.pdf is stale by many revisions) and
   reconcile the author list / ORCID (`0000-0001-9876-5432` looks placeholder).
8. **Rust port: post-submission** (author decision; R0 measured the suite at ~4 min and near
   core-saturation).

## Standing rules this project runs on (author-set, keep following)

- Evidence rule: no item is done until its verify step **ran** and the literal output is pasted
  into `fixes_phases.md`.
- Behaviour changes: code → verify → full 50-seed suite → **flag before rewriting the paper if
  Table III moves** — especially when the shift is favourable.
- A smaller true result beats a larger unsupported one; negative results are reported with the
  same weight as positive ones; metrics that stop discriminating are dropped, not rescued.
- Manuscript figures only via `experiments/publish_figures.py` (md5-verified); figure CSVs are
  `*_merged.csv` (collision-proof names); `plot_results.py` refuses wrong-but-parseable inputs.
- **Verification artifacts must never share a namespace with production artifacts.** This shape
  recurred three times before it was named: the metrics handler mutating simulation state (F-17),
  ad-hoc verification runs clobbering the thermodynamics CSV that feeds Figure 4, and finally the
  F-17 regression pin itself writing `experiment_3_stability.csv` into the real `logs/` — inside
  the test that pins this very class. The rule generalises all three fixes: observers are
  read-only, verification runs write to scratch/tmp namespaces (`tmp_path`, the session
  scratchpad), and anything that generates a production input does so through its named producer
  script. If a check needs to write, the first question is *where*, not *what*.
