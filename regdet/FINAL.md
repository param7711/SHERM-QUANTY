# RegDet — FINAL (Phase 4 closed)

Status: **FINAL. No further tuning.** Declared final by the user after the
7-instrument FX run. This document is the durable close-out; `PROJECT_STATE.md`
keeps the full experiment-by-experiment history.

## What ships

| artefact | what it is |
|---|---|
| `notebooks/regdet_v11_master.ipynb` | the detector on **Nifty 2h** (the user's `final_regdet_v9.ipynb`) |
| `notebooks/fx_v9.ipynb` | the **same** detector on **FX/commodity**, 7 instruments, one-line switch |
| `generators/build_master_notebook_v2.py` | source of truth for the master |
| `generators/build_fx_v9.py` | source of truth for the FX build |

`fx_v9.ipynb` differs from the master in exactly **one functional cell** (the
data loader) plus **five display strings** in chart cells 27/40/42/60/66,
enforced by an allowlist assert. **Not one constant is re-tuned.** FX at 8h is
3 bars/day, matching Nifty 2h's 3 bars/session, so `BARS_PER_DAY = 3` stays
literally correct and `assert MOM_3D_BARS == 9` passes untouched.

## Shipped configuration (unchanged across every market)

```
N_STATES=5  COVARIANCE='diag'  9 features  114 params
BAR_DIR_WEIGHT=0.5   ENSEMBLE_K=4 (seeds 42-45)   CONF_L=0.50   CONFIRM_BARS=2
INTENSITY_MODE='frozen_z'   ESCALATION_DURING_HOLD='allow'   DIRECTION_MODE='rank'
Z_HI=0.5  EFF_HI=0.35  Z_HI_EXIT=0.35  EFF_HI_EXIT=0.25  EFF_WIN=9
momentum ladder 1/3/5 days   CONTEXT_DAYS=12   TRAIN_FRACTION=0.70  N_FOLDS=4
ADOPTED_CONFIG='A: lean-cov'
```

## Final measured result — 7 FX/commodity instruments, nothing re-tuned

8h bars, 2,822 bars each, 2018-07 → 2022-03. 0 cell failures on every run.

```
instrument   ordering    ANTI  PASS  cfg   med run len  SIDE%  Sharpe   B&H   maxDD  B&H DD
EUR/USD     3/3 BROKEN    12    10   NO     5/3/6/3/6   34.5   -0.07  -0.21   -7.3   -11.5
GBP/USD     3/3 BROKEN    13     6   NO     4/2/8/2/6   41.9   -0.26   0.11  -11.5   -14.9
USD/JPY     3/3 BROKEN    12     7   NO     4/2/5/3/4   36.1   -0.18   0.10   -6.1   -10.7
XAU/USD     3/3 BROKEN    12    10   NO     4/3/8/3/5   30.6    0.30   0.94  -18.3   -18.6
AUD/USD     3/3 BROKEN     7     8   NO     5/2/7/2/5   39.3    0.14   0.06   -7.0   -24.7
USD/CAD     3/3 BROKEN    12     7   NO     3/4/6/3/5   32.7   -0.95  -0.13  -17.9   -17.7
USD/CHF     3/3 BROKEN    10     8   NO     5/2/6/3/4   34.5   -0.23  -0.35   -6.7   -14.0
NIFTY v9    3/3 BROKEN     2     9   ok     5/2/5/3/5   28.8    0.41   0.60   -7.8   -16.6
```

## What this detector IS, stated honestly, so the next phase plans around it

**It is a DESCRIPTIVE regime segmenter.** At matched bar density it produces
coherent, visually sensible regime blocks on every instrument tested — AUD/USD
reads the COVID crash as one sustained bear block, gold holds long green through
2019-2020.

**It is NOT a forward-return predictor, and this is measured, not suspected:**

* Forward-return ordering (`H_BULL > L_BULL > SIDEWAYS > L_BEAR > H_BEAR`) is
  **BROKEN at all 3 horizons on all 8 runs — Nifty included.** This is a shared
  failure of the architecture, not an FX-specific one.
* On FX, `FAIL(ANTI)` — scorecard rows where the statistic is materially
  *backwards* — runs **7-13 (mean 11.1) vs Nifty's 2, worse on 7 of 7**. On FX
  the detector is frequently **anti**-informative, not merely uninformative.
* The config head-to-head **disagrees** with the adopted `A: lean-cov` on 7 of 7
  FX runs (it agrees on Nifty).
* `fx_meanrev.ipynb`: intraday FX variance ratios 0.80-1.06, **no series
  reaches |z*| ≥ 2** — near-random-walk. There is little directional structure
  for any method to find at these horizons.

**Consequence for downstream phases:** consume RegDet as a *state/context
label*, not as a directional signal. Anything that treats `H_BULL` as "go long"
is relying on a relationship this project measured and could not find.

## Things tried and rejected (do not re-litigate)

| attempt | outcome |
|---|---|
| shorter lookback (`CONTEXT_DAYS` 12→3) | discovery effect +0.484 **did not replicate**; held-out −0.908 → overfitting |
| disjoint-block feature set (7 feat, 94 params) | grind whipsaw −15%, conditioning 7/7 better, **but** seed stability *worse* (91.6→83.5) |
| volatility-adaptive context window | all 6 predictions "CONFIRMED" yet **failed**: 17% slower vs shipped, ~0 coherence gain. Root cause: `P(LOW-VOL \| grind) = 45.7%` vs base 48.7% — volatility carries **no** information about the grind |
| XGBoost / deep learning | argued against: no valid target (forward returns carry no signal; hindsight labels are look-ahead), and ~1,300 effective observations vs 114 params is already marginal |

## Methodological findings worth carrying forward

1. **A plotting artefact drove months of conclusions.** Earlier FX charts drew
   57,311 bars into one panel vs Nifty's 2,858 — 20× density — which merges
   shading into stripes regardless of label quality. The "FX barcodes" reading
   was substantially rendering, not detection. **Always match bar density before
   comparing regime charts across markets.**
2. **Pre-registration is necessary but not sufficient.** The adaptive-window test
   passed all six pre-registered predictions and was still a failure, because one
   prediction was framed against the wrong control (the always-slow arm instead
   of the shipped one). Pre-register the comparison *against what ships*.
3. **Endpoint comparisons masquerade as monotonicity.** Several "confirmed"
   results were Q1-vs-Q5 checks on curves that were never monotone in between.

## Explicitly NOT done

* RegDet is **not wired into production**. `regime_engine_tactical.py` remains
  Nifty-only; nothing in `regdet/` is imported by it.
* 5 of 8 older generators cannot rebuild their notebooks (they assert
  `BAR_DIR_WEIGHT = 0.0` while the shipped master is 0.5). Pre-existing; fixing
  it risks changing shipped notebooks and needs a deliberate decision.
* `CONFIRM_BARS` 2→1 (prediction S4) remains **NOT EVALUATED**.
* FX data ends **2022-03-04**. Nothing here speaks to 2022-2026.
* FX window (2018-2022) and Nifty window (2023-2026) differ, so market **era** is
  confounded with market in every cross-market comparison above.
