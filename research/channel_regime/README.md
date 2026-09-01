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
