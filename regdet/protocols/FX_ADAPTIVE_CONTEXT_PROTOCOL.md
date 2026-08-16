# Volatility-adaptive context window — frozen BEFORE the notebook was built or run

Grade in plain words afterward, CONFIRMED / REFUTED. Do not reinterpret a
prediction after seeing its number. Do not soften a refutation.

## SUPERSEDES the static sweep

`FX_CONTEXT_WIDEN_PROTOCOL.md` (frozen minutes earlier, static
`CONTEXT_DAYS` 12/18/24) is **superseded as the primary question** and its
arms are demoted to **controls** inside this test. It is not deleted and its
predictions are not rewritten — the static generator was built but never run,
so no result is being buried.

Reason, in the user's words: *"I don't want to increase the context length at
all times... at times of high volatility, when there is a high bear or high
bull, [the wide window] won't be good. The context window should be increased
only for times of relatively lower volatility."*

That is a real objection to the static test. A window wide enough to make a
quiet grind coherent is, by construction, slow to turn on a violent move —
the sensitivity sweep already measured that trade (`fx_sensitivity.ipynb`:
narrowing cut lag L 44.7→23.2 bars while raising whipsaw). Static widening
buys grind coherence and pays for it exactly where the detector currently
works best. Adaptive widening is an attempt to stop paying that price.

## The mechanism

```
selector : vol_expansion computed at a FIXED reference window (the shipped
           12 days), NEVER at the adaptive window.
state    : LOW-VOL  when selector < enter-threshold
           HIGH-VOL when selector > exit-threshold
           hysteresis band between them; state HOLDS inside the band.
window   : LOW-VOL  -> CONTEXT_LONG  (24 days)
           HIGH-VOL -> CONTEXT_SHORT (12 days, the shipped value)
adapts   : drawdown, dist_ma, vol_expansion -- exactly the three features
           CONTEXT_DAYS controls. Nothing else moves.
```

Three design decisions, each with its failure mode named:

1. **The selector is computed at a FIXED window.** If the selector used the
   adaptive window it would depend on its own output — circular, and the
   state could oscillate with no external cause. Fixed reference removes
   this by construction.
2. **Hysteresis on the state, not a bare threshold.** A bare threshold makes
   the *window itself* flicker bar-to-bar near the boundary, re-introducing
   the exact whipsaw this is meant to remove. Same device as the shipped
   `Z_HI`/`Z_HI_EXIT` gate bands (FIX 3).
3. **Thresholds are fit-window quantiles** (40th / 60th percentile of the
   selector over the leading `n_fit` bars), not hardcoded numbers. A
   hardcoded ratio would mean something different on each instrument. This
   is the same re-derivation the `StandardScaler` and `trend_z` baseline
   already do — permitted, and NOT a per-market re-tune, because the RULE is
   identical everywhere.

## Causality — the new risk surface, and how it is proved

The adaptive window is the first thing in this project where **a feature's
own window length is data-dependent**. That is a genuine new opportunity for
look-ahead, so it gets a dedicated probe rather than an assurance:

* every rolling window is trailing (bars ≤ t);
* the selector thresholds come from the leading `n_fit` bars ONLY, computed
  once, never per-bar;
* the hysteresis state machine is a strict forward scan — state at bar t
  reads state at t−1 and selector at t, nothing later.
* **TRUNCATION PROBE (required to pass):** cut the series at several T,
  re-run the whole adaptive pipeline on the prefix, and assert that every
  emitted label at t ≤ T is bit-identical. This is the real test; the three
  bullets above are only the reasons it is expected to pass.

## Grid

| axis | values |
|---|---|
| arm | **ctx12 static** (shipped baseline), **ctx24 static** (the "always wide" control), **ADAPTIVE 12↔24** (the proposal) |
| instrument | all 7: EUR/USD, GBP/USD, USD/JPY, XAU/USD, AUD/USD, USD/CAD, USD/CHF |
| timeframe | 1h only |

3 arms × 7 instruments = 21 runs. Feature SET is the plain shipped 9-feature
nested set throughout — this does not combine with `fx_featureset.ipynb`'s
disjoint set. One mechanism at a time.

## Pre-registered predictions

The whole point of the adaptive arm is that it must beat BOTH controls, on
DIFFERENT axes. A1 and A2 together are the claim; either alone is not.

```
A1 (GRIND -- must beat the SHIPPED baseline): adaptive's grind-cell whipsaw
   (same low-eff_fast / high-eff_slow cell as F1 in fx_featureset.ipynb) is
   LOWER than static ctx12, on >= 4 of 7 instruments AND on the pooled mean.

A2 (VIOLENT -- must beat the ALWAYS-WIDE control, and this is the whole
   point of adapting): on the most violent bars (top eff_fast quintile),
   adaptive's transition lag L is LOWER than static ctx24, on >= 4 of 7
   instruments AND on the pooled mean. If adaptive is as slow as ctx24 on
   violent moves, it has bought nothing that static widening did not, and
   the added complexity is unjustified.

A3 (SATURATION GUARD): no single regime exceeds 90% of OOS bars and all 5
   labels stay reachable, on the adaptive arm AND on ctx24. A REFUTED A3 on
   an arm VOIDS that arm's other results -- a detector with no
   discrimination is not an improvement regardless of its whipsaw number.

A4 (THE SWITCH ACTUALLY SWITCHES): the adaptive arm spends between 15% and
   85% of OOS bars in each state. Outside that band the arm has silently
   collapsed to one of the static controls and A1/A2 are measuring nothing.
   This is a validity check on the experiment, not a claim about markets.

A5 (do-no-harm, guards): G1-G4 pass on adaptive on >= as many instruments
   as static ctx12.

A6 (HONESTY CHECK, not the goal): adaptive's OOS (BULL-BEAR) HAC t is not
   systematically worse than static ctx12 (mean difference >= -0.20). Not
   expected to manufacture skill fx_meanrev.ipynb already found absent, and
   does NOT gate A1-A5.
```

## Guards and reporting

* All three arms reported **side by side per instrument**, never pooled
  without also showing the spread.
* A run that saturates (A3) is reported as such, not excluded from the table.
* No threshold moved after the fact. No instrument dropped.
* Bit-for-bit: the adaptive path forced permanently into HIGH-VOL state must
  reproduce static ctx12 labels exactly, and forced permanently into LOW-VOL
  must reproduce static ctx24 exactly. An adaptive switch whose OFF values
  are not its own controls is not a controlled experiment.

## What this cannot establish

Not a return, not a Sharpe, not a tradeability claim. Data ends 2022-03-04.
Tested on FX only — yfinance is firewalled in this sandbox, so the Nifty
series this idea came from cannot be swept here. A positive result is
evidence the MECHANISM works on these instruments; reproducing it on Nifty
is a separate, still-unrun test.
