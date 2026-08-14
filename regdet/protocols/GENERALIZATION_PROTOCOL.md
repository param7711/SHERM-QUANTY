# Cross-market / cross-timeframe generalization — pre-declared protocol

Written BEFORE any numbers exist. Frozen at commit time.

## The question
RegDet V1.1 was developed entirely on Nifty 2h over ~2.9 years. Does the
ARCHITECTURE work elsewhere, or is it tuned to that one series?

This is an ARCHITECTURE-level overfitting test, not a parameter search.
NOTHING here is allowed to re-tune RegDet. If a market fails, that is the result.

## Grid A — Kaggle / yfinance (the trading-condition test)
5 markets x 3 timeframes = 15 independent runs.

  markets    : ^GSPC (S&P 500), ^IXIC (NASDAQ), EURUSD=X, GBPUSD=X, JPY=X
  timeframes : 1h, 2h, 4h   (yfinance 60m base, resampled: 2h/4h by
               OHLC aggregation high=max, low=min, close=last)
  span       : whatever yfinance serves at 60m (~730d cap). PRINT the actual
               bar count and span per run -- do not assume.

## Grid B — GitHub daily FX (the long-horizon test)
3 pairs, daily, verified locally against real events:
  JPY/USD  13,926 bars  1971-01-04 -> 2026-07-24
  GBP/USD  13,932 bars  1971-01-04 -> 2026-07-24
  EUR/USD   7,190 bars  1999-01-04 -> 2026-07-24
Source: raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv
(US Federal Reserve H.10 release, redistributed by the `datasets` org)
Verified: Brexit 2016-06-24 GBP +8.17%; Bretton Woods 1973-02-13 JPY -9.50%;
GFC Oct-2008 spikes in both. Monotone dates, no duplicates, no non-positive prices.
Label as: REAL FED DATA via GitHub mirror — decision-grade for FX,
NOT a substitute for the user's own Kaggle equity run.

Grid B additionally resamples daily -> weekly to test cadence robustness
at the long-horizon end. Daily->intraday is IMPOSSIBLE and must not be faked.

## WHAT IS FIXED (never re-tuned per market)
    N_STATES=5, covariance='diag', the 9-feature set,
    CONF_L=0.50, CONFIRM_BARS=2, BAR_DIR_WEIGHT=0.5, ENSEMBLE_K=4,
    Z_HI=0.5, EFF_HI=0.35, Z_HI_EXIT=0.35, EFF_HI_EXIT=0.25, EFF_WIN=9,
    momentum ladder 1/3/5 "days", CONTEXT_DAYS=12,
    INTENSITY_MODE='frozen_z', ESCALATION_DURING_HOLD='allow',
    DIRECTION_MODE='rank', TRAIN_FRACTION=0.70, N_FOLDS=4.

"Days" are expressed in BARS PER SESSION for that timeframe. State the
bars-per-day assumption per timeframe explicitly and print it:
    1h -> BARS_PER_DAY=6 (FX ~24h, equities ~6.5h -- STATE the choice per market)
    2h -> BARS_PER_DAY=3
    4h -> BARS_PER_DAY=2 (approx; print the realised median bars/session)
If a market's real bars-per-session differs from the assumption, PRINT the
discrepancy. Do not silently rescale.

## WHAT MAY RE-DERIVE PER MARKET (data-dependent by design)
    - the StandardScaler (fit on that market's own leading train window)
    - trend_z's mu/sd baseline (already frozen on the fit window by design)
    - any fit-window quantile threshold the engine already computes internally
Nothing else. If the architecture only works when knobs are re-tuned per
market, THAT IS THE OVERFITTING FINDING and must be reported as such.

## METRICS — identical to the existing notebooks so results are comparable
Reuse verbatim where they exist:
    occupancy per label (all 5)      -- from the master
    S  = seed-set label agreement, R=4 disjoint seed sets  (build_stability_lag.py)
    L  = ZigZag transition lag in bars; ZigZag must NEVER reach a label (assert)
    W  = switches per 100 bars
    guards G1-G4 (occupancy 3-50%, no collapse, W<=25, SIDEWAYS<=50%)
Plus, for the overfitting question specifically:
    IS_vs_OOS: the forward-return direction ordering and HAC t computed
    SEPARATELY on the in-sample (train) span and the out-of-sample span.
    A detector that works IS but collapses OOS is the classic overfit signature.
    Report both spans side by side for every run. Do NOT report a pooled number.

## PRE-REGISTERED PREDICTIONS (falsifiable; print BEFORE the numbers)
  P1. Guards G1-G4 pass on a MAJORITY of the 15 Grid-A runs. If they fail
      broadly, the architecture is Nifty-specific.
  P2. SIDEWAYS occupancy lands in a similar band (roughly 20-45%) across
      markets. Wild swings mean the thresholds are scale-dependent.
  P3. OOS direction ordering degrades vs IS but does not INVERT. Inversion
      across many runs = overfitting.
  P4. FX and equities behave DIFFERENTLY (FX has no earnings cycle, different
      vol clustering). A difference here is expected and is NOT a failure.
The notebook must grade itself against P1-P4 in plain words and state
CONFIRMED / REFUTED. Do not reinterpret a prediction after seeing its number.

## GUARDS AGAINST SELF-DECEPTION
  - No composite score across markets. Report the full table.
  - No market may be dropped from the report for looking bad. If a run fails
    or errors, it appears in the table with the reason.
  - No threshold may be moved to make a market pass.
  - yfinance is FIREWALLED in the sandbox: Grid A falls back to synthetic with
    a loud banner and is machinery-proof ONLY. Grid B runs for real locally.
  - Truncation probe on at least one run per grid: cutting the series at T must
    leave every label at t <= T bit-identical.

## SCOPE LIMITS
  - Do NOT re-tune RegDet. This measures; it does not optimise.
  - Do NOT add features (no Vortex, no intrabar) -- that is separate work.
  - Daily -> intraday resampling is impossible; only downward resampling allowed.
