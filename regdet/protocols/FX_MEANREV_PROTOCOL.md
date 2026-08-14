# FX mean-reversion protocol — frozen BEFORE any number existed

Written before `fx_meanrev.ipynb` was built or run. Grade in plain words
afterward, CONFIRMED / REFUTED. Do not reinterpret a prediction after seeing
its number. Do not soften a refutation.

## Why this exists

`intraday_fx.ipynb` measured, on 6 real intraday FX runs with every constant
frozen at its Nifty value:

* **all 6** out-of-sample `(mean fwd | BULL) − (mean fwd | BEAR)` contrasts
  NEGATIVE (OOS HAC *t* from −0.18 to −1.62),
* full 5-label ordering holding on **0 of 6** IS spans and **0 of 6** OOS spans,
* 3 of 6 runs inverting sign IS→OOS (P3 REFUTED).

A hypothesis was offered for that, and it was explicitly labelled a hypothesis:

> RegDet is backward-looking momentum — it labels BULL *after* an up-move. FX
> majors are much closer to driftless with short-horizon mean reversion than an
> equity index is. If the series mean-reverts at the detector's horizons,
> forward returns after a BULL label are negative.

This notebook tests that hypothesis directly. It fits **no HMM** and emits **no
regime label** — it measures the price series and the detector's own trend
input, nothing else.

## The claim being tested, stated so it can fail

The hypothesis needs BOTH links to hold. Either one failing refutes it:

1. the FX series mean-reverts at the horizons the detector actually uses, and
2. the detector's own trend feature (`mom_3d`, the shipped `TREND_FEATURE`)
   **anti-predicts** forward return — high past momentum → low forward return.

If (2) fails, the negative BULL−BEAR contrast is NOT explained by mean
reversion at the detector's input, and the cause lies somewhere else (the HMM
state assignment, the vol proxy standing in for VIX, or something unidentified).
Saying so is the useful outcome, not a failure of the notebook.

## Scope

* Same 6 series as the main grid: EUR/USD, GBP/USD, USD/JPY × {1h, 2h}.
* Same data source, same divisor detection, same 14 verification checks, same
  provenance labelling: COMMUNITY GITHUB DATA, verified against Brexit/COVID,
  NOT an official feed, **ends 2022-03-04**.
* 2h is a DOWNWARD resample of 1h. No finer bar is fabricated from a coarser one.
* Horizons are the detector's own: 1/3/5 days = 24/72/120 bars at 1h,
  12/36/60 at 2h.
* **No Nifty comparison.** The user scoped this to FX only. Every verdict is
  therefore stated against the random-walk null (VR = 1), which is an absolute
  benchmark, never against an equity baseline.

## Definitions, fixed here

* **Variance ratio** `VR(q) = Var(q-bar log return) / (q · Var(1-bar log return))`,
  overlapping windows, unbiased correction as in Lo–MacKinlay (1988).
  `VR < 1` = mean reversion, `= 1` = random walk, `> 1` = trending.
  Significance via the heteroskedasticity-robust `z*(q)`; `|z*| >= 2` is the bar.
* **Autocorrelation** of 1-bar log returns at lags 1…120.
* **The decisive test (link 2)**: bucket every bar into quintiles of `mom_3d`
  computed EXACTLY as the shipped engine computes it, then take mean forward
  return per quintile per horizon, with a Newey–West (Bartlett, lag = h−1) HAC
  *t* on the top-minus-bottom quintile spread. Causal throughout: `mom_3d` at
  bar *t* reads bars ≤ *t*; the forward return is retrospective grading only.

## Pre-registered predictions

```
M1: VR(q) < 1 at the detector's own horizons on a MAJORITY (>= 4 of 6) of the
    series. CONFIRMED only if the majority ALSO reach |z*| >= 2 at some tested
    q -- a VR below 1 that is not distinguishable from 1 is not evidence.

M2: The mom_3d quintile -> mean-forward-return relation is DECREASING (top
    quintile below bottom quintile) on a MAJORITY (>= 4 of 6) of the series,
    h-averaged. This is the link that actually explains the BULL-BEAR sign.

M3: The top-minus-bottom quintile spread is NEGATIVE with HAC |t| >= 2 on a
    MAJORITY (>= 4 of 6) of the series. This is M2 with significance attached;
    M2 can pass on sign alone while M3 fails, and that combination would mean
    the effect is real in direction but too weak to carry the explanation.
```

## Verdict rule for the hypothesis as a whole

* **M2 and M3 both CONFIRMED** → mean reversion at the detector's own input is
  a sufficient explanation for the negative BULL−BEAR contrast.
* **M2 CONFIRMED, M3 REFUTED** → direction consistent, magnitude too weak to
  carry it alone. Report as partial; do not claim the mechanism is settled.
* **M2 REFUTED** → the hypothesis is REFUTED regardless of M1. The negative
  contrast is not explained by the detector's trend input, and the honest
  statement is that the cause is still unknown.

M1 alone settles nothing: a series can mean-revert without that being what
drives this particular detector's labels.

## What this notebook does NOT do

* It does not re-tune anything, and it does not test a fix.
* It does not claim tradeability, return, or Sharpe.
* It says nothing about 2022–2026; the data ends 2022-03-04.
