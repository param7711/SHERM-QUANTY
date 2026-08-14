# FX sensitivity sweep protocol — frozen BEFORE the sweep was built or run

Grade in plain words afterward, CONFIRMED / REFUTED. Do not reinterpret a
prediction after seeing its number. Do not soften a refutation.

## What this is, and what it is NOT

This **deliberately re-tunes** two constants the frozen generalization test
holds fixed. It is therefore **not** an architecture-generalization test and
makes no claim to be one. `intraday_fx.ipynb` stays untouched and keeps its
"nothing was re-tuned" property; this is a separate experiment that asks a
different question:

> Does making RegDet **more sensitive** recover forward-return skill on
> intraday FX, or does it only trade lag for whipsaw?

Requested directly by the user: shorten the max lookback, and drop
`CONFIRM_BARS` from 2 to 1.

## Why the answer is in doubt

`fx_meanrev.ipynb` measured, on the same data:

* variance ratios of 0.80–1.06 with **no series reaching |z\*| ≥ 2** at any
  tested horizon — none distinguishable from a random walk,
* `mom_3d` quintile curves **monotone on 0 of 6 series**, top-minus-bottom
  spreads at |HAC *t*| ≤ 0.81.

If there is little directional structure to find, sensitivity changes *what the
detector reacts to*, not *whether the reaction is informative*. That is the
reasoning behind S3 below — and S3 is the prediction that matters.

## Grid

| axis | values |
|---|---|
| instrument | EUR/USD, GBP/USD, USD/JPY, **XAU/USD** |
| timeframe | 1h, 2h (2h resampled DOWN from 1h) |
| `CONTEXT_DAYS` | **12** (shipped baseline), 6, 3 |
| `CONFIRM_BARS` | **2** (shipped baseline), 1 |

4 instruments × 2 timeframes = 8 runs; 3 lookback arms × 2 confirm settings
= 6 configurations. The shipped `(12, 2)` corner is included as an **internal
baseline** so every comparison is within this notebook and does not lean on the
earlier run.

`CONFIRM_BARS` is a LABELING knob, not a fitting knob: it is applied after
direction is decided. Each lookback arm is therefore **fitted once** and
**labelled twice**. Fits = 3 × 8 × R4 × K4 = 384. This is asserted in the
notebook, not assumed.

Everything else stays at its shipped Nifty value.

## XAU/USD — stated plainly

* It is a **commodity quoted in USD, not an FX pair**, and is labelled that way
  everywhere. It is included because it was requested as an instrument expected
  to "work in long regimes like Nifty".
* Its integer scale is a **third distinct divisor** (÷1e2), detected not
  hardcoded, asserted into a 1000–2100 band.
* Its span **starts 2012-05-17**, ~6 months before the majors' 2012-11-16. Same
  end date. Spans therefore differ across instruments by design; not truncated,
  because discarding real data to equalise a cosmetic property is worse. Flagged
  wherever instruments are compared.
* Verified live against known gold events before use: Aug-2020 peak high
  ≈ 2074.87 vs the known all-time high ≈ 2075; Apr-2013 crash ≈ −14.7%;
  Dec-2015 low ≈ 1050.
* **Its variance ratios were already measured and it is NOT measurably
  trending**: VR 0.92–1.02 at 1–20 days, 0.88–0.92 at 40–120 days, `|z*| ≤ 0.6`
  everywhere. S5 below is written knowing that, and is a genuine prediction
  about the DETECTOR, which that statistic does not settle.

## Pre-registered predictions

```
S1: Shortening CONTEXT_DAYS 12 -> 6 -> 3 RAISES the whipsaw rate W.
    CONFIRMED if mean W is monotone increasing as lookback shortens, at
    CONFIRM_BARS = 2.

S2: Shortening CONTEXT_DAYS 12 -> 6 -> 3 LOWERS the transition lag L.
    CONFIRMED if mean L is monotone decreasing as lookback shortens, at
    CONFIRM_BARS = 2.

S3: SENSITIVITY DOES NOT BUY SKILL.  No configuration produces a MAJORITY
    (>= 5 of 8) of runs with a POSITIVE out-of-sample (BULL - BEAR) contrast.
    CONFIRMED if every one of the 6 configurations fails to reach that
    majority. REFUTED the moment any single configuration reaches it --
    which would mean sensitivity DOES recover skill, and would be the most
    important result in this project.

S4: CONFIRM_BARS 2 -> 1 raises W and lowers L at every lookback, and does
    NOT improve OOS ordering. CONFIRMED if mean W rises and mean L falls at
    all 3 lookbacks AND the count of runs with positive OOS contrast does
    not increase at any lookback.

S5: XAU/USD does not separate from the FX majors on OOS ordering. CONFIRMED
    if XAU/USD's 2 runs land inside the range spanned by the 6 major runs at
    the shipped (12, 2) baseline. A separation would support the "long
    regimes" intuition that motivated including it.
```

## Guards and reporting

* Guards G1–G4, S / L / W, full 5-label occupancy, and IS/OOS contrasts
  reported **side by side, never pooled**, exactly as in the sibling notebooks.
* A configuration that fails guards is REPORTED, not dropped.
* No threshold is moved. No run is dropped. If a shortened lookback voids the
  guards, that is the finding.

## Runtime

384 fits is ~4× the main grid (96 fits, 399 s). Projected ~27 min, over the
15-minute budget. The **only sanctioned reduction is a training-window cap**,
stated in the output with its reason. `R = 4` is never cut and no instrument or
timeframe is ever dropped.

## What this cannot establish

Not a return, not a Sharpe, not a tradeability claim. Data ends 2022-03-04 and
says nothing about 2022–2026. The volatility input remains a causal realised-vol
**proxy** on every run — FX and gold have no natural VIX.
