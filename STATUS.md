# Status Snapshot — submission-ready, 2026-08-18

**Companion documents:** `audit_findings.md` (findings F-01..F-39), `fixes_phases.md` (plan +
pasted evidence per item, including the executed Phase 5 gate), `rust_conversion_plan.md`
(post-submission). Repo head at time of writing: `fd22601` on `main`, pushed.

---

## Where things stand: SUBMISSION-READY

The Phase 5 gate has been executed with evidence, and the two former author blockers are
RESOLVED (author decision, 2026-08-18): the 4th author was removed — which also dissolved the
placeholder ORCID, hers — and the paper of record carries 3 authors, verified rendering in the
compiled PDF with the removed name and ORCID verified ABSENT. Tables IV/V were restructured
(fixed tabular, scriptsize) after a rendering-overflow report and visually confirmed clean.

The last template placeholder — the IEEEtran funding-agency thanks — was deleted (author
decision, 2026-08-18, no funding to declare); the verifier now asserts its absence. Nothing
remains open.

## The paper of record

`manuscript/final_manuscript.pdf` — compiled 2026-08-18 with tectonic 0.15.0 from the corrected
`.tex` (12 pages, 5 figures). The stale `draft 11.pdf` is deleted. The compiled output was
**verified, not assumed**: `experiments/verify_manuscript_pdf.py` extracts the text layer and
asserts corrected numbers present, retired set absent outside the two withdrawal narratives,
exactly 3 authors rendered with the removed name and ORCID asserted absent. CLEAN, exit 0.

Current Table III (50 seeds, all corrections applied):

| Arm | t50 (ticks) | Decay (units/tick) | Survival |
|---|---|---|---|
| Unconstrained RGG | 127 ± 5 | 0.261 ± 0.017 | 407 ± 29 |
| Oracle (all-to-all cost) | 21 ± 0 | 2.801 ± 0.207 | 39 ± 4 |
| **Proposed** | **128 ± 5** | **0.0499 ± 0.0000** (genuine, see caption) | **> 2000 (50/50 censored)** |

Claims: 5.2× energy reduction (headline); no differential attrition delay (stated as a negative
result); oracle buys 1.55× half-life, consumed entirely by all-to-all bandwidth cost; the
pursuit–halt tension with its 6% tail observed → traced → fixed → re-measured to 0/50; the
two-regime F-04 story (phantom-centroid convergence vs exact fixed point, separated by belief
eviction); §V-E provenance section naming all three silent-protection corrections.

## Verification infrastructure (all gate items, re-run at submission)

- `python -m pytest tests/` → **199 passed, 0 failed** (full suite incl. GUI; PySide6 installed)
- `python diagnostics_audit.py` → 9/9, every check falsifiable; `--self-test` proves it by
  sabotaging the projector and catching it
- `python experiments/publish_figures.py --check` → 3× md5 MATCH
- `python experiments/sweep_retired_claims.py` → exit 0 (encoding-agnostic byte sweep; limits
  stated in its docstring)
- `python experiments/verify_manuscript_pdf.py` → exit 0 (text-layer check of the compiled PDF;
  authoritative for the one artifact whose payload is compressed)
- `tests/test_regression_pins.py` — the audit's history as permanent tests (hidden ε clamp,
  1e154 overflow, ψ clamp, logging purity, corpse-λ₂, belief-age bound, oracle channel)

## History rewrite note (2026-08-18)

The git history was rewritten (`git filter-repo`) to remove private material: `manuscript/`,
`Review/`, `Documents/`, `Architecture/`, `refrences/`, and the old compiled drafts. Consequence:
**commit hashes cited anywhere in the audit documents predate the rewrite** and no longer resolve
on the remote; they resolve only in the pre-scrub backup bundle
(`../swarm_backup_pre_scrub.bundle`, kept locally, never pushed). The audit *content* is
unchanged — every finding, measurement, and evidence block stands as written.

## Standing rules (author-set; a fresh session inherits these, not rediscovers them)

- Evidence rule: no item is done until its verify step **ran** and the literal output is pasted
  into `fixes_phases.md`. Use TRUE exit codes — never a pipeline's tail status.
- Behaviour changes: code → verify → full 50-seed suite → **flag before rewriting the paper if
  Table III moves** — especially when the shift is favourable.
- A smaller true result beats a larger unsupported one; negative results carry the same weight;
  metrics that stop discriminating are dropped, not rescued.
- Manuscript figures only via `publish_figures.py`; the compiled PDF only via
  `verify_manuscript_pdf.py`; figure CSVs are `*_merged.csv`; `plot_results.py` refuses
  wrong-but-parseable inputs.
- **Verification artifacts must never share a namespace with production artifacts.** Named after
  three instances of one shape: the metrics handler mutating simulation state (F-17), ad-hoc
  verification runs clobbering the thermodynamics CSV, and the F-17 regression pin writing into
  the real `logs/` — inside the test that pins this very class. Observers are read-only;
  verification runs write to scratch/tmp namespaces; production inputs come only from their
  named producer scripts. If a check needs to write, the first question is *where*, not *what*.
- Specification defects are distinct from code defects: correct code under an undefined printed
  algorithm corrupts *reproducibility*, not results (Algorithm 3's empty-neighbourhood case).
  The gate's algorithm diffs are line-by-line for this reason.

## After submission

- Rust port per `rust_conversion_plan.md` (R0 already measured; the suite runs in ~4 min and the
  port is a constant-factor play — post-submission by decision).
- Remaining non-blocking niceties: GUI panels beyond test coverage; scenario TaskParams UI
  surfacing; percolation Figure 2 could move to a multi-seed sweep if a reviewer asks.
