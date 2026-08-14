# RegDet V1.1 — Sherm Quanty regime detector

**New session / new chat? Read `PROJECT_STATE.md` first.** It is the authoritative
handoff document: shipped config, every decision made and why, what is settled, what
is still open, and the verification discipline that has repeatedly caught real bugs.
Everything below is just a map of the folders.

---

## What this is

A 2-hour Nifty regime detector. Every bar gets one of five labels:
`H_BULL / L_BULL / SIDEWAYS / L_BEAR / H_BEAR`.

It is built from **two independent axes** that get combined:

- **Direction** (bull / sideways / bear) — a Gaussian Hidden Markov Model over 9 causal
  features, optionally blended with a per-bar momentum score.
- **Intensity** (High vs Low conviction) — pure rolling price statistics
  (`trend_z` × `trend_efficiency`). **Contains no HMM at all.** This is load-bearing:
  it is why changing the HMM's weight barely moves occupancy.

## Folder map

| path | what |
|---|---|
| `PROJECT_STATE.md` | **START HERE.** Full state, decisions, open items, constraints. |
| `protocols/` | Frozen experiment protocols, each written *before* its numbers existed. |
| `generators/` | **Source of truth.** Notebooks are always regenerated from these, never hand-edited. |
| `notebooks/` | Generated deliverables (ship unexecuted, 0 stored outputs). |
| `notes/regdet_notes.pdf` | 14-page standalone explainer — architecture, HMM/EM math, every chart, all bugs. |
| `harnesses/` | Verification scripts (cell-by-cell runner, off-path equivalence, truncation probes). |
| `reference/` | v6 and v7 real-data runs, kept as the anchors the ablation compares against. |

## The generator → notebook rule

`build_*.py` files are authoritative. The `.ipynb` files are build products.
**Never hand-edit a notebook** — change the generator and regenerate. Several bugs in
this project's history came from notebooks drifting from their generators.

```bash
cd generators && python3 build_master_notebook_v2.py   # regenerates regdet_v11_master.ipynb
```

## Shipped configuration (summary — see PROJECT_STATE.md for the reasoning)

v6 base + 6 audited bug fixes + a 12-day context window.

```
BAR_DIR_WEIGHT=0.5      ENSEMBLE_K=4        CONF_L=0.50       CONFIRM_BARS=2
INTENSITY_MODE='frozen_z'                   ESCALATION_DURING_HOLD='allow'
DIRECTION_MODE='rank'   (verified bit-for-bit no-op)
Z_HI=0.5  EFF_HI=0.35   Z_HI_EXIT=0.35      EFF_HI_EXIT=0.25   EFF_WIN=9
Momentum ladder: 1/3/5 days (kept fast)     Context window: 12 days (36 bars)
ADOPTED_CONFIG_NAME='A: lean-cov'  (N=5, diag, 9 features, 114 params)
```

## Hard-won facts a new session should not have to rediscover

- **`BAR_DIR_WEIGHT` is settled.** Its effect between 0.0–0.75 is smaller than the noise
  from merely reseeding the same value. Reported UNMEASURABLE under a frozen protocol.
- **Config ranking is provably undecidable** at 4 folds — the notebook prints
  `NOT DECIDABLE` rather than a false winner. `A: lean-cov` is pinned on parameter
  efficiency, not on a Sharpe contest.
- **EM is bistable** — refits land in one of two "basins" (~98% agreement within,
  ~63% across). Suspected cause: collinear nested momentum features. Unresolved.
- **Every "obvious" bug hypothesis in this project has been wrong at least once when
  actually measured** (`CONF_L`, `DIRECTION_MODE='rank'`, capacity→stability). Measure first.
- **yfinance is firewalled in the sandbox.** Only `raw.githubusercontent.com` is reachable
  for data. Real intraday FX was sourced from a community GitHub repo and validated
  against Brexit (−10.84%) and COVID before use.

## Verification discipline (this catches real bugs — keep doing it)

1. Run notebooks **cell-by-cell**, compiling each cell separately against a shared
   namespace. Cell 0 is always `%pip install` — a Jupyter magic, a `SyntaxError` in plain
   Python, **not** a failure.
2. **Truncation probe** for causality: cut the series at `T`, assert no label at `t <= T`
   changes. Stronger than checking only the final bar.
3. Pin `OMP_NUM_THREADS=1` for any two-run equivalence check — multithreaded BLAS makes
   identical configs differ by reduction-order noise.
4. **Look at the rendered PNGs.** Real rendering defects (weekend-gap shading,
   semantically inverted heatmap colours, overlapping legends) have been caught this way
   at least six times, never by reading code.
5. **Pre-register criteria in a frozen protocol file** before running anything, then grade
   CONFIRMED/REFUTED in plain words afterward. This produced the project's most trustworthy
   findings — including results that went *against* the hypothesis being tested.
