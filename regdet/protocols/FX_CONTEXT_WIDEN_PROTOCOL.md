# Context-widening protocol — frozen BEFORE the sweep was built or run

Grade in plain words afterward, CONFIRMED / REFUTED. Do not reinterpret a
prediction after seeing its number. Do not soften a refutation.

## What this is

User proposal, made while looking at the regime-on-price chart in their own
uploaded `final_regdet_v9.ipynb` (real yfinance Nifty 2h, 2023-08→2026-07,
shipped config): "if we just increase the context window by a little bit, it
could be phenomenal." This tests that lever directly, on FX (real data
reachable here; yfinance is firewalled in this sandbox so the exact Nifty
series cannot be re-fetched to sweep on Nifty itself).

**One variable at a time.** `fx_featureset.ipynb` changed the feature
construction; this changes NOTHING about features — the plain shipped
9-feature NESTED set, unmodified, exactly as `intraday_fx.ipynb` and
`fx_sensitivity.ipynb` used it. Only `CONTEXT_DAYS` (which sets `SWING_WIN`
and `VOL_SLOW` — the `drawdown`/`dist_ma` window) moves.

## Why the answer is genuinely in doubt, both directions

Two things are already measured, pointing opposite ways:

* **Narrowing** `CONTEXT_DAYS` 12→6→3 (`fx_sensitivity.ipynb`, real FX):
  RAISES whipsaw W (S1 CONFIRMED), LOWERS lag (S2 CONFIRMED), buys no skill
  (S3 CONFIRMED). Shortening trades lag for noise.
* **Widening** `CONTEXT_DAYS` on real DAILY Nifty (the original decline-lag
  fix, `PROJECT_STATE.md`): 7d bear 90.7%/bull 7.4% → 12d bear 94.4%/bull
  0.0% → **20d bear 100.0%/bull 0.0%/side 0.0% — full saturation**, the
  detector loses ALL discrimination. There is a real ceiling; "a little bit"
  past 12 is untested territory between a real fix (7→12) and a known
  collapse (→20).

Both directions have failure modes already measured. This sweep is not
assumed to help — it could recreate the sensitivity sweep's "buys lag
reduction, not skill" story, or start sliding toward saturation, or genuinely
help. All three are reported as found.

## Grid

| axis | values |
|---|---|
| feature design | NESTED only (shipped, unmodified) — no combination with the disjoint set |
| instrument | all 7: EUR/USD, GBP/USD, USD/JPY, XAU/USD, AUD/USD, USD/CAD, USD/CHF |
| timeframe | 1h only (2h dropped: corr(1h,2h delta)=+0.986, established pseudo-replication) |
| `CONTEXT_DAYS` | **12** (shipped baseline), 18, 24 |

3 arms × 7 instruments = 21 runs. Everything else stays at the shipped value,
guarded exactly as `sweep_knobs`/`ctx_arm` do in the sibling notebooks.

## Pre-registered predictions

```
C1 (PRIMARY -- the user's hypothesis): grind-cell whipsaw (same "steady
   trend, low eff_fast / high eff_slow" cell as F1 in fx_featureset.ipynb)
   is LOWER at ctx18 and/or ctx24 than ctx12, on a majority (>= 4 of 7)
   instruments AND on the pooled mean, for AT LEAST one of the two wider
   arms.
   CONFIRMED if true for ctx18 or ctx24 (report both; a win on either counts,
   since "a little bit" and "more" are different claims).

C2 (SATURATION GUARD -- do not ship a collapse and call it an improvement):
   occupancy stays multi-label at every arm -- no single regime exceeds 90%
   of OOS bars, and every one of the 5 labels remains reachable (>0 bars).
   CONFIRMED if this holds at ctx18 AND ctx24. A REFUTED C2 at either arm
   means that arm is saturating, exactly like the 20-day daily-Nifty case,
   and its C1 result (if positive) is void -- a detector with no
   discrimination cannot be "phenomenal" regardless of its whipsaw number.

C3 (do-no-harm, guards): G1-G4 pass on the widened arms on AT LEAST as many
   instruments as at ctx12.

C4 (do-no-harm, violent moves): the eff_fast quintile W curve does not
   INVERT shape at the widened arms (Q5 mean W stays below Q1 mean W,
   pooled) -- widening context should not make the detector worse on the
   moves it already reads well.

C5 (honesty check, not the goal): OOS (BULL-BEAR) HAC t at the widened arms
   is not systematically worse than at ctx12 (mean difference >= -0.20).
   Not expected to manufacture skill fx_meanrev.ipynb already found absent.
```

## Guards and reporting

* Guards G1–G4, S/L/W, full 5-label occupancy, and IS/OOS contrasts reported
  **side by side per instrument**, never pooled without also showing spread.
* A run that saturates (fails C2) is REPORTED as a saturation failure, not
  quietly excluded from C1's arithmetic.
* No threshold moved. No instrument dropped.

## What this cannot establish

Not a return, not a Sharpe, not a tradeability claim. Data ends 2022-03-04.
This does not test the hypothesis on Nifty itself — only on FX, using the
same lever. A positive result here is evidence the LEVER can help on some
markets; it is not proof it would reproduce on Nifty without an actual
Nifty-side test (blocked here by the yfinance firewall).
