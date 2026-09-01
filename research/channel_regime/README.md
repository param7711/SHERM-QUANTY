# Regression-channel regime switch — hypothesis test

**Verdict: no edge found. The rule is long beta with a decoration on top.
Nothing here survives a multiple-testing correction, and the trend-following
half of the hypothesis never actually fires.**

## The hypothesis as tested

Use a rolling linear-regression channel as a trend/range filter, then switch
logic on it:

* **inside the channel → mean-revert** — buy near the lower band, short near
  the upper band, take profit back at the centre line;
* **20-SMA breaks out of the channel → follow the trend** — long when it
  breaks above the upper band, short when it breaks below the lower band.

Formalised in `backtest.py` as a state machine over
`z = (close - centre) / sigma` and `smaz = (sma - centre) / sigma`, where
`centre` and `sigma` come from an OLS fit of close on bar index over the
trailing `n` bars, read at the fit's endpoint (`channel.py`).

Defaults: `n=100`, `k=2.0`, `sma_len=20`, mean-reversion entry at
`|z| >= 0.9k`, exit at the centre line, 5 bps per side, signal formed at
bar `t` and filled at `t+1`, close-to-close P&L, ±1 unit positions.

## Data

The sandbox's egress policy blocks every market-data host (Yahoo, Stooq, NSE
archives, AlphaVantage, Tiingo — all 403 at the proxy), and
`data/download_data.py` silently falls back to synthetic GBM, on which any
timing rule is guaranteed to lose. So the test runs on genuinely observed
history that ships inside PyPI packages (`data.py`):

| Instrument | What | Span | Bars |
|---|---|---|---|
| SP500 | S&P 500 daily OHLCV | 1999-01-04 → 2018-12-31 | 5,031 |
| NASDAQ | Nasdaq Composite daily OHLCV | 1999-01-04 → 2018-12-31 | 5,031 |
| WTI | WTI crude daily close | 1986-01-02 → 2019-01-03 | 8,321 |
| GOOG | Google daily OHLCV | 2004-08-19 → 2013-03-01 | 2,148 |
| EURUSD | EUR/USD hourly OHLCV | 2017-04-19 → 2018-02-07 | 5,000 |

**This is not NIFTY/BANKNIFTY.** The conclusions are about whether this class
of rule works on real price series; the read has to be re-run on Indian index
data before it decides anything about capital.

## What the tests found

### 1. The regime switch does not switch

A 20-bar SMA of price cannot deviate from a 100-bar regression line as far as
price does — smoothing shrinks it by roughly half. Measured share of bars
outside the band:

| | SP500 | NASDAQ | WTI | GOOG | EURUSD |
|---|---|---|---|---|---|
| `\|z\| > 2` (price) | 11.8% | 12.8% | 14.1% | 16.4% | 14.4% |
| `\|smaz\| > 2` (SMA) | 0.73% | 0.28% | 0.07% | 0.39% | 0.25% |
| max `\|smaz\|` ever | 2.43 | 2.15 | 2.07 | 2.17 | 2.23 |

The trend leg fires **2–3 times in 20 years** and can never fire at all for
`k >= 2.5`. The full hypothesis and the mean-reversion leg alone are the same
strategy: Sharpe 0.347 vs 0.362 on SP500, 0.104 vs 0.110 on NASDAQ.

### 2. The mean-reversion leg is long beta, not alpha

Newey-West regression of net strategy returns on the underlying:

| Instrument | Sharpe | Beta | Ann. alpha | alpha t-stat | Long P&L | Short P&L |
|---|---|---|---|---|---|---|
| SP500 | 0.35 | 0.26 | +3.5% | 1.26 | +1.04 | −0.02 |
| NASDAQ | 0.10 | 0.20 | +0.2% | 0.05 | +0.82 | −0.37 |
| WTI | 0.16 | 0.04 | +4.3% | 0.89 | +1.08 | +0.55 |
| GOOG | −0.49 | 0.06 | −13.4% | −1.63 | +0.05 | −1.03 |
| EURUSD | −3.43 | −0.05 | −17.1% | −3.19 | −0.01 | −0.10 |

Every long-side unit makes money in instruments that went up; the short side
makes nothing anywhere except WTI. No alpha t-stat clears 1.4.

Against buy & hold the strategy wins on one of five: SP500 (0.35 vs 0.28
Sharpe, 4.0% vs 3.6% CAGR, at 48% exposure). It loses on NASDAQ (0.10 vs
0.34), WTI (0.16 vs 0.25), GOOG (−0.49 vs 0.88) and EURUSD (−3.43 vs 2.32).
One win out of five, on the instrument with the mildest drawdowns, is what a
zero-skill long-biased filter looks like.

### 3. It is indistinguishable from chance

500-rep **sign-flip null** (randomly flips the sign of each return, which
keeps the volatility clustering the channel reacts to and destroys only the
direction):

| Instrument | Observed | Null mean | Null 95th pct | p-value |
|---|---|---|---|---|
| SP500 | 0.347 | −0.043 | **0.360** | 0.066 |
| NASDAQ | 0.104 | −0.021 | 0.334 | 0.304 |
| WTI | 0.161 | −0.004 | 0.273 | 0.148 |
| GOOG | −0.491 | −0.005 | 0.529 | 0.926 |
| EURUSD | −3.431 | −2.188 | −0.668 | 0.892 |

The best instrument's Sharpe sits *below* the 95th percentile of its own null.
An i.i.d. permutation null (`09_nulls.csv`) gives the same picture.

### 4. Tuning it would not rescue it

648-point grid (`n × k × sma_len × entry_frac × exit_z`):

| Instrument | Best | Median | % positive | Deflated-Sharpe prob. | PBO | Walk-forward OOS |
|---|---|---|---|---|---|---|
| SP500 | 0.73 | 0.23 | 86% | 0.74 | 0.30 | +0.24 |
| NASDAQ | 0.35 | 0.07 | 70% | 0.30 | 0.79 | −0.05 |
| WTI | 0.44 | 0.09 | 75% | 0.55 | 0.55 | −0.12 |
| GOOG | 0.44 | −0.36 | 8% | 0.23 | 0.48 | −0.19 |
| EURUSD | 0.87 | −2.99 | 2% | 0.00 | 0.05 | −1.01 |

* **Deflated Sharpe** — probability the best config beats the best of 648
  random tries. Nothing reaches 0.95; SP500's 0.73 is the high-water mark.
* **PBO** (CSCV, 12 blocks, 924 splits) — probability the in-sample winner is
  below median out of sample. 0.79 on NASDAQ, 0.55 on WTI. OOS-minus-IS Sharpe
  is negative on every instrument (−0.38 to −2.18). EURUSD's low PBO is an
  artefact of every config being consistently bad: low PBO is not good news.
* **Anchored walk-forward** (6 folds, pick the best config on everything so
  far, trade the next fold) — stitched OOS Sharpe is ≈0 or negative on 4 of 5,
  and the selected parameters jump around between folds.

### 5. No nearby repair works

Ten variants (`10_variants.csv`), mean net Sharpe across the five instruments:

| Variant | Mean Sharpe | Positive on |
|---|---|---|
| A as specified | −0.66 | 3/5 |
| B mean-reversion leg only | −0.70 | 3/5 |
| C trend leg only (SMA trigger) | +0.26 | 3/5 *(2–3 trades — noise)* |
| D trend trigger = price, same band | −1.96 | 0/5 |
| E trend trigger = SMA, band 1.0σ | −0.56 | 1/5 |
| F mean-reversion only when the channel is flat | −0.46 | 2/5 |
| G flat-gated MR + steep-gated trend | −0.60 | 2/5 |
| H trend only, price breakout + steep channel | −0.66 | 1/5 |
| I MR with a 2σ stop | −0.70 | 3/5 |
| J MR with a 20-bar time stop | −0.72 | 3/5 |

Making the trend leg actually fire (D, E) makes things clearly worse — the
breakout leg is a net loser when it is live, which is why the version that
never fires looks best.

### 6. Costs are not the binding constraint

Turnover is low (95–171 trades over 20–33 years), so the strategy only dies at
20–30 bps per side. The problem is upstream: the signal is weak, not expensive.

## Reading this honestly

It is not *overfit* — it has not been fitted to anything yet. It is worse than
that: **there is no edge here to overfit.** The one number that looks like a
result (SP500 Sharpe 0.35) is long-dip-buying in a bull market, it loses to
buy & hold, its alpha t-stat is 1.26, and it does not clear its own null.
Meanwhile the PBO and deflated-Sharpe numbers say that if you *do* start
tuning this on one instrument, the tuned version will not survive contact with
out-of-sample data.

The two structural facts worth carrying forward:

1. **An SMA-vs-band breakout is the wrong regime variable.** It is mechanically
   incapable of triggering. If the channel is going to say "trending vs
   ranging", the variable has to be the channel's own drift-to-noise —
   `slope × n / sigma`, equivalently the fit's R² — not where a smoothed price
   sits relative to a band built from unsmoothed residuals. `slope_max` /
   `slope_min_trend` in `Params` implement that; variants F–H are a first cut
   and are still negative, but they at least test the intended idea.
2. **Fading a band is directionally biased by construction.** In a series with
   drift, the long side gets the drift and the short side fights it. Any future
   version has to be scored on beta-adjusted alpha, not raw Sharpe, or it will
   keep rediscovering the equity risk premium.

## Reproducing

```
pip install -r requirements.txt arch backtesting
python research/channel_regime/channel.py      # self-check: fit, sigma, causality
python research/channel_regime/backtest.py     # self-check: no look-ahead, costs
python research/channel_regime/data.py         # materialise data/research/*.parquet
python research/channel_regime/experiments.py --stage all --reps 500
```

Roughly 12 minutes end to end. Outputs land in `results/channel_regime/`
(`01_baseline` … `12_signflip`).

## Guards in the code

* The channel is fitted on trailing windows only and read at the endpoint;
  `channel.py` asserts that truncating the future leaves past values unchanged.
* `backtest.py` asserts the realised position is exactly the signal lagged one
  bar, that costs strictly reduce P&L, and that positions stay in {−1, 0, +1}.
* Sharpe, PBO and deflated Sharpe are all computed on net-of-cost returns.

---

# v2 — the specified indicators

v1 tested the idea as described in prose. This section tests it against the
two indicators actually named: **Linear Regression ++ [Dev Lucem]** for the
channel, and a **20-SMA with inner bands at 0.3 stdev** for the mean.

## DevLucem's exact settings, verified against the published source

TradingView is blocked by this sandbox's egress policy (403 at the proxy for
`www.tradingview.com`, and for every search engine), so the Pine v5 source was
read from a public mirror on GitHub instead. Defaults, from the script itself:

| Input | Default |
|---|---|
| `source` | `close` |
| `length` | **100** |
| `dev` (deviation multiplier) | **2.0** |
| `offset` | 0 |
| `smoothing` | 1 (no smoothing of the regression output) |
| Resolution | chart timeframe |

The band half-width is the part that matters:

```pine
deviationSum += math.pow(source[i] - (slope * (x - i) + intercept), 2)
deviation = math.sqrt(deviationSum / length)
```

Divided by `length`, **not** `length - 2` — the population standard deviation
of the residuals. v1 used the textbook `n-2` correction, which makes the bands
`sqrt(100/98) = 1.0102x` too wide. `dev_ddof=0` is now the default in `Params`;
the v1 CSVs reproduce with `dev_ddof=2`.

Two further details taken from the source rather than assumed:

* `linreg = ta.linreg(source, length, offset)` is the fitted value at the
  window's **endpoint**, and `slope = linreg - linreg_p` is the fitted line's
  per-bar slope. Both match the v1 channel exactly.
* The indicator's own alerts are `ta.crossunder(close, lower)` and
  `ta.crossover(close, upper)` — it is a **fade**, and it fires on the band
  itself, not near it. So `entry_frac = 1.0`, not v1's 0.9.

`channel.py` now carries `devlucem_reference()`, a deliberately unvectorised
transcription of the Pine in Pine's own variable names, and `_pine_parity_check()`
asserts the fast path matches it. Max absolute difference over 500 bars:
centre 9.1e-13, slope 2.3e-13, sigma 1.8e-14, z 5.2e-14.

## The 0.3-sigma inner band

Implemented as `inner_sd=0.3` on `SMA(close, 20)` with `stdev(close, 20)`
population (Pine's `ta.stdev` convention). Measured share of bars with price
outside that band:

| | SP500 | NASDAQ | WTI | GOOG | EURUSD |
|---|---|---|---|---|---|
| beyond 0.3 sd | 88.9% | 88.4% | 87.3% | 89.1% | 88.1% |
| beyond 1.5 sd | 30.4% | 32.4% | 30.7% | 34.1% | 29.2% |
| beyond 2.0 sd | 10.2% | 10.8% | 11.8% | 13.2% | 13.8% |

This is the exact mirror of v1's problem. There the trigger fired on 0.3% of
bars; here it fires on 89%. Used as a **trigger** the 0.3 band is unusable —
spec S2 puts the system in the market 87% of the time over 610 trades for a
Sharpe of −0.42 on the S&P. Used as an **exit** it cuts winners short: S1
scores below S0 on three of five instruments. Its useful role is the third
one: a **no-fade zone**, which is what the cap turns out to be.

## The cap: 1.5 sigma, and it is a plateau

Sweeping the distance from the 20-SMA past which no new fade is opened
(`14_cap_sweep.csv`), on the DevLucem fade with the exit at the regression
centre:

| cap | SP500 | NASDAQ | WTI | GOOG | EURUSD |
|---|---|---|---|---|---|
| 1.00 | +0.264 | +0.271 | −0.094 | −0.197 | −2.340 |
| 1.25 | +0.592 | +0.264 | −0.234 | −0.353 | −3.346 |
| **1.50** | **+0.693** | +0.264 | −0.166 | −0.535 | −2.791 |
| 1.75 | +0.678 | +0.194 | −0.105 | −0.660 | −3.136 |
| 2.00 | +0.629 | +0.144 | +0.011 | −0.570 | −3.269 |
| 2.50 | +0.395 | +0.139 | −0.036 | −0.510 | −3.202 |
| none | +0.375 | +0.125 | +0.024 | −0.489 | −3.337 |

On the S&P 500 this is the first result in the study that behaves like a real
effect rather than a fitted one:

* **A plateau, not a spike.** 1.25 → 2.00 all score 0.59–0.74 (peak 0.74 at
  1.60). v1's grid had no such shape anywhere.
* **Positive in every sub-period** — 1999-04 +1.24, 2004-09 +0.27,
  2009-14 +0.87, 2014-19 +0.46.
* **Survives costs.** 49 trades in 20 years; Sharpe 0.72 gross, 0.69 at 5 bps,
  0.55 at 30 bps.
* **Clears its null.** 400 sign-flip runs: observed 0.693 against a null mean
  of −0.003, sd 0.235, 95th percentile 0.387 — p = 0.003. Corrected for the
  16 cap/exit cells searched, family-wise p ≈ 0.047.

And the reasons not to bank it yet:

* **It is one instrument.** NASDAQ is flat-to-mildly-positive; WTI, GOOG and
  EUR/USD are unhelped or worse at every cap. Count instrument selection as
  part of the search (16 cells × 5 instruments) and the corrected p is ≈ 0.24.
* **The economics are still long-biased.** This is the v1 dip-buyer with a
  falling-knife filter; it has not been re-tested for beta-adjusted alpha.
* **It has not been walk-forwarded or PBO'd** under the v2 spec, and it has
  never seen NIFTY or BANKNIFTY.

Recommended setting: **cap at 1.5 sd** — mid-plateau, not the peak, which is
the setting least likely to be an artefact of where the peak happened to land.

## Still unverified

The **RegDet BB IL** indicator itself could not be read. It is not on any
branch of this repo (`regdet/` on `claude/regdet-intraday-forex-3votj4`
contains the regime detector and its Pine port, no Bollinger indicator), it
returns nothing on GitHub code search, and TradingView is blocked here. What
is implemented is what was specified in words: mean = SMA(close, 20), inner
bands at 0.3 population stdev. If the real indicator has an outer band, a
different stdev basis, or its own cap, those numbers change.
