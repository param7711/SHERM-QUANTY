# Disjoint-block feature set protocol — frozen BEFORE the notebook was built or run

Grade in plain words afterward, CONFIRMED / REFUTED. Do not reinterpret a
prediction after seeing its number. Do not soften a refutation.

## The two complaints this answers

1. **User, reading the regime-on-price charts**: "great when moves are very
   rapid and volatile, but very bad when there's a consistent trend going on
   for a long period with slight ups and downs." Confirmed by direct
   inspection (`_zoom.py`, `_eff_diag.py` E3): the shipped detector produces
   coherent coloured blocks in violent windows and a "barcode" of rapid
   colour flips in sustained grinds — the exact regime it should read best.
2. **User, on the feature set**: "isn't the parameters too many? Like, it
   could be cluttering and the noise could be too much." Measured on real
   AUD/USD data: the shipped 9 features carry only **4.20 effective
   dimensions** (participation ratio). `mom_3d`, `mom_5d`, `drawdown`,
   `dist_ma` correlate at 0.71–0.88 pairwise — one dimension measured four
   ways.

These are the SAME finding. Nested cumulative momentum (`mom_1d/3d/5d`, all
measured from bar 0) is collinear by construction, and its longest reach is
5 days — inside any 5-day slice of a 3-month grind there genuinely are
mixed-direction wiggles, so the HMM's direction axis never sees the trend
whole. That is a timescale-and-collinearity problem, not a threshold problem
— the same species of bug as the 2025 "decline-lag" fix (`PROJECT_STATE.md`),
just on the feature axis instead of the window-width axis.

## The fix — replace, do not add

An earlier proposal in this conversation ("extend the ladder to 1/5/20/60d")
was WRONG and is superseded here: adding nested rungs makes conditioning
*worse* (measured: 11 nested features → only 2.74 effective dims, condition
number 6.1). The fix actually measured to work is **disjoint, non-overlapping
return blocks**:

```
d0_1   = cumsum(0 -> 1 day)
d1_5   = cumsum(0 -> 5 day)  - cumsum(0 -> 1 day)
d5_20  = cumsum(0 -> 20 day) - cumsum(0 -> 5 day)
d20_60 = cumsum(0 -> 60 day) - cumsum(0 -> 20 day)
```

Measured on the same real data: max |off-diagonal correlation| falls from
0.588 (nested, same 4 horizons) to **0.036**, condition number 6.1 → **1.1**,
effective dimensions 2.74 → **4.00 of 4**.

### The 7-feature replacement set

| kept unchanged | replaced | dropped |
|---|---|---|
| `vol_2h`, `vix_chg`, `dist_ma` | `mom_1d/3d/5d` → `d0_1/d1_5/d5_20/d20_60` | `drawdown` (−0.86 corr with `dist_ma`), `ret_2h` (subsumed by `d0_1`), `vol_expansion` |

`FEATURE_COLS_DISJOINT = [d0_1, d1_5, d5_20, d20_60, vol_2h, vix_chg, dist_ma]`
— 7 features, N=5 diag → **94 params** (vs shipped 9 → 114). Same
`FEATURE_SIGN`/`FEATURE_MAG(0.4)` convention as the shipped momentum terms;
`DIRECTION_EXCLUDE = ('vol_2h',)` for the same FIX-1 reason (magnitude, not
signed).

**The INTENSITY (H vs L) axis is left completely untouched.** `TREND_FEATURE
= 'mom_3d'` is still computed, at its original 3-day/9-bar window, for the
gate only — it is deliberately NOT one of the 7 HMM inputs. That axis is
already fast and already correct on violent moves (E1 in `_eff_diag.py`
CONFIRMED: W falls monotonically as `eff_fast` rises). Only the DIRECTION /
HMM state-identification axis is being changed. One mechanism at a time.

## Not a parameter search — stated so this cannot be misread later

`fx_sensitivity.ipynb` searched a knob's VALUE (3 lookback arms) against
these same instruments and the winning arm was picked by looking at their
scores — that is why it needed a held-out replication, and why the ctx3
effect turned out to be overfitting (`fx_replication.ipynb`).

This is different in kind: the block edges (1/5/20/60 days) and the dropped
feature were fixed by measured collinearity structure on AUD/USD, decided
**before this notebook ran on any instrument**, and applied identically
everywhere. There is no scoreboard of candidate edges to have picked the best
from. Regime charts for AUD/USD and USD/CAD were inspected earlier in this
conversation to diagnose the *problem* (barcode in grinds) — but never to see
what disjoint blocks would output, because that run had not happened yet. All
7 instruments below are therefore graded together, not split into
discovery/held-out; a split would be theatre without an actual search to
guard against.

## Grid

| axis | values |
|---|---|
| feature design | **nested** (shipped, 9 feat, baseline) vs **disjoint** (7 feat, the fix) |
| instrument | EUR/USD, GBP/USD, USD/JPY, XAU/USD, AUD/USD, USD/CAD, USD/CHF (the repo's full non-redundant 7; crosses and 2h excluded for the same measured pseudo-replication reasons as `fx_replication.ipynb`: corr(1h,2h delta)=+0.986, crosses corr 0.989–0.9985 vs triangular synthetics) |
| timeframe | 1h only |
| `CONTEXT_DAYS` | 12 (shipped baseline) for both designs — the ONLY thing that changes is feature construction |

2 designs × 7 instruments = **14 runs**. Everything else (N=5, diag cov,
`BAR_DIR_WEIGHT=0.5`, `ENSEMBLE_K=4`, `R=4`, `CONF_L`, `CONFIRM_BARS`,
`Z_HI`/`EFF_HI` bands, `TRAIN_FRACTION=0.70`) stays at the shipped value for
both arms — re-armed as a guarded invariant exactly as `sweep_knobs` does in
`fx_sensitivity.ipynb`, with `FEATURE_COLS`/`DIRECTION_EXCLUDE` as the two
knobs deliberately let out of the cage.

## Pre-registered predictions

```
F1 (PRIMARY -- the user's visual complaint): in the "steady trend with
   wiggles" cell (eff_fast BELOW its OOS-span median AND eff_slow AT/ABOVE
   its OOS-span median -- the exact E3 cell from _eff_diag.py, same
   eff_fast=9-bar / eff_slow=20-day windows), the whipsaw rate W is LOWER
   for disjoint than nested, on the majority (>= 4 of 7) instruments, and on
   the pooled mean across all 7.
   CONFIRMED if both the majority count and the pooled mean move that
   direction. A drop on the pooled mean alone, with the majority failing,
   is NOT a confirmation -- report the split.

F2 (do-no-harm on the axis that already works): disjoint's W still falls
   monotonically from the lowest to the highest eff_fast quintile, pooled
   across the 7 instruments -- the same shape E1 already confirmed for
   nested. CONFIRMED if the pooled Q1 -> Q5 mean W is monotone decreasing
   for disjoint too.

F3 (conditioning, measured live, not assumed from the earlier AUD/USD-only
   probe): disjoint's feature-correlation condition number is lower, and its
   participation-ratio effective-dimension SHARE (dims / feature count) is
   higher, than nested's, on the majority (>= 4 of 7) instruments.

F4 (parameter economy -- arithmetic, not a measurement): n_params(N=5,
   diag, F=7) < n_params(N=5, diag, F=9). Included for completeness; will
   trivially CONFIRM.

F5 (do-no-harm, structural): guards G1-G4 pass on disjoint on AT LEAST as
   many of the 7 runs as they do on nested. A regression here is reported
   even if F1 confirms -- a coherence win that breaks a guard is not a win.

F6 (bistability -- the mechanism PROJECT_STATE suspects but has not
   resolved): mean seed-stability S is HIGHER for disjoint than nested,
   pooled across the 7 runs.

F7 (honesty check -- NOT the goal, expect null): disjoint's OOS
   (BULL - BEAR) HAC contrast is not systematically WORSE than nested's
   (mean difference >= -0.20). This notebook is not expected to manufacture
   forward-return skill where fx_meanrev.ipynb already found none -- F7
   exists only to catch the fix accidentally making things worse, not to
   claim it makes them better. A neutral or even negative F7 result does
   NOT invalidate F1-F6; a large negative one is flagged regardless.
```

## Guards and reporting

* Guards G1–G4, S/L/W, full 5-label occupancy, and IS/OOS contrasts reported
  **side by side per instrument**, never pooled into a single number without
  also showing the per-instrument spread.
* A run that fails guards is REPORTED, not dropped.
* No threshold is moved. No instrument is dropped. If the fix voids a guard
  somewhere, that is the finding, stated plainly.
* Bit-for-bit check: the nested arm of THIS notebook must reproduce
  `intraday_fx.ipynb`'s EUR/USD@1h / GBP/USD@1h / USD/JPY@1h labels exactly
  (0 bars differ) — proves the harvested engine was not accidentally altered
  while wiring in the design switch.

## Runtime

14 runs × R4 × K4 = 224 fits, vs the main grid's 96 fits in ~399s (~4.2s /
fit averaged). Projected ~940s (~16 min). Measured via the runtime probe
before launch, same as every sibling notebook; the only sanctioned reduction
is the training-window cap, and if armed it uses the same declared constant
(`TRAIN_CAP_BARS = 12000`) already fixed for reproducibility in
`fx_sensitivity.ipynb` — never a wall-clock-derived value. `R = 4` is never
cut and no instrument is ever dropped.

## What this cannot establish

Not a return, not a Sharpe, not a tradeability claim. Data ends 2022-03-04.
The volatility input remains a causal realised-vol PROXY. This notebook
tests whether the feature redesign fixes the DESCRIPTIVE failure the user
identified by eye (F1/F2) without breaking anything (F3-F6), and checks
honestly whether it also buys forward-return skill (F7, not expected to).
It is not a claim that the detector becomes tradeable.
