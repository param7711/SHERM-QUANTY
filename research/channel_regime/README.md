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

---

# v3 — with the actual indicator file

`regdet_v6_bb_IL.pine` settled both open questions and revealed that v1 and
v2 had been filtering on the wrong variable.

## What the file says

**The Bollinger block is display only.** Lines 209-210: *"This is PURELY
VISUAL. It feeds nothing: not the direction features, not the intensity
gate, not the regime label."* Its five lines are one basis and one
standard deviation read at two distances:

| Input | Default | On this chart |
|---|---|---|
| `bbLength` | 20 | 20 |
| `bbMaType` | SMA | SMA |
| `bbSrc` | close | close |
| `bbMult` (outer) | **2.0** | 2.0 |
| `bbInnerMult` (inner) | 1.0 | **0.3** |
| `bbOffset` | 0 | 0 |

`bbSd = ta.stdev(bbSrc, bbLength)` is biased=true, i.e. population — which
is what `regdet.bollinger()` implements. **The outer band is the cap.** The
v2 sweep found a plateau from 1.25 to 2.0 and recommended 1.5; the
indicator's own outer default of 2.0 sits inside that plateau.

**And the chart carries a real regime detector.** RegDet emits five labels
(H_BULL / L_BULL / SIDEWAYS / L_BEAR / H_BEAR) from a direction axis
(softmax over five signed momentum features, with a `CONF_L` confidence
override and a `CONFIRM_BARS` delay) and an intensity axis (`trend_z`,
Kaufman efficiency, and this file's volatility cap). `regdet.py` ports it
line for line at w=1.0, where the HMM term drops out exactly. On S&P 500
daily the occupancy is SIDEWAYS 49.0%, L_BULL 20.4%, L_BEAR 16.9%,
H_BULL 8.1%, H_BEAR 5.7%.

## Result 1 — the detector's own warning is correct

The file says: *"NOT A SIGNAL... the 5-label forward-return ordering is
BROKEN at every horizon. Do not treat H_BULL as 'go long'."* Tested
directly (`15_regdet_specs.csv`), net Sharpe:

| Spec | SP500 | NASDAQ | WTI | GOOG | EURUSD |
|---|---|---|---|---|---|
| R6 trend leg only, long H_BULL / short H_BEAR | −0.90 | −0.92 | −0.28 | −0.11 | −13.04 |
| R3 fade + follow the trend in H_* | −0.33 | −0.73 | −0.37 | −0.65 | −12.53 |

Negative on every instrument. Every spec containing a trend leg is negative
on every instrument. This is now the third independent way the
trend-following half of the hypothesis has failed: v1's SMA breakout could
not fire, v2's price breakout lost money when it did, and the actual regime
detector's high-intensity labels lose money in both directions.

## Result 2 — as a *gate on fading*, it works

Fade the DevLucem band only while RegDet says SIDEWAYS, with the 2.0-sigma
outer band capping new entries (spec R5):

| | SP500 | NASDAQ | WTI | GOOG | EURUSD |
|---|---|---|---|---|---|
| R0 fade, no gate | +0.375 | +0.125 | +0.024 | −0.489 | −3.337 |
| **R5 fade in SIDEWAYS + 2.0s cap** | **+0.733** | **+0.506** | −0.040 | −0.391 | −2.595 |

On the two equity indices this is a different animal from anything in v1:

| | SP500 | NASDAQ |
|---|---|---|
| Net Sharpe | +0.733 | +0.506 |
| CAGR | +4.43% | +4.02% |
| Max drawdown | **−9.5%** | −21.1% |
| Trades / exposure | 39 / 17.1% | 41 / 19.8% |
| Hit rate | 82% | 73% |
| **Beta** | **+0.048** | **+0.002** |
| **Ann. alpha (Newey-West)** | **+4.26%, t = +3.30** | **+4.28%, t = +2.29** |
| Long / short P&L | +0.64 / +0.28 | +0.79 / +0.09 |
| Walk-forward OOS (80 configs) | +0.580 | +0.490 |
| Deflated Sharpe | 0.924 | **0.973** |
| Surface: % of 80 configs positive | 95% | 95% |

Near-zero beta, both directions profitable, positive in all four S&P
sub-periods (+1.04 / +0.33 / +0.93 / +0.59), still +0.45 at 50 bps per
side, and a broad plateau rather than a spike. v1's walk-forward was ≈0 or
negative on four of five; here it is +0.58 and +0.49.

## Result 3 — but most of the S&P gain is not the detector

The sharpest control: rotate the label series by a random offset. Occupancy
and run-length structure survive intact; only the alignment to price dies.
500 rotations:

| | observed | rotated-label null | p | sign-flip null | p |
|---|---|---|---|---|---|
| SP500 | +0.733 | mean **+0.615**, sd 0.101 | **0.144** | mean +0.007, sd 0.205 | <0.002 |
| NASDAQ | +0.506 | mean +0.174, sd 0.081 | **<0.001** | mean +0.004, sd 0.220 | 0.013 |

Read carefully:

* Against sign-flipped prices both indices clear easily — **the DevLucem
  fade itself has real edge.** That is the finding v1 missed by testing the
  wrong entry rule.
* On the S&P, a *randomly rotated* regime series scores +0.615 against the
  real one's +0.733. The detector's timing is not what earns the return
  there; **any blocky entry filter of roughly this shape would do**, because
  what the gate really does is stop the fade re-entering repeatedly inside
  one persistent move. Attribute the S&P number to "fade plus a sparse
  entry filter", not to RegDet.
* On Nasdaq the real labels beat the rotated ones decisively (+0.506 vs
  +0.174). There the detector's timing is doing genuine work.

## Result 4 — the strategy uses one RegDet knob, not the whole detector

Varying the detector's own settings under spec R5:

| RegDet setting | SP500 | NASDAQ | SIDEWAYS share |
|---|---|---|---|
| shipped (vol cap on) | +0.733 | +0.506 | 48.5% |
| **vol cap OFF** | **+0.733** | **+0.506** | 48.5% |
| `Z_HI` 0.40 / 0.60 | +0.733 | +0.506 | 48.5% |
| `EFF_HI` 0.25 / 0.45 | +0.733 | +0.506 | 48.5% |
| `CONF_L` 0.45 | +0.615 | +0.546 | 32.1% |
| `CONF_L` 0.55 | +0.682 | +0.712 | 63.1% |
| rolling baseline | +0.582 | +0.265 | 45.6% |

SIDEWAYS is decided on the **direction** axis alone, so gating on it uses
`CONF_L`, the softmax and the confirmation delay — and *nothing* from the
intensity axis. `trend_z`, Kaufman efficiency and this file's volatility cap
(Deviation 2) have **exactly zero effect** on this strategy. If the fade
gate is the use case, the vol cap is not earning its complexity here.
`CONF_L` is the one live knob, and 0.45-0.55 is a plateau.

## Standing caveats

* **Still not NIFTY/BANKNIFTY.** Everything above is S&P 500, Nasdaq, WTI,
  GOOG daily and EUR/USD hourly.
* **Two instruments out of five.** WTI, GOOG and EUR/USD are flat or
  negative under every spec tried.
* **PBO is ~0.5** on the R5 surface. With 95% of configs positive and
  clustered, ranking among them is noise — which argues for taking the
  middle of the plateau rather than the in-sample winner, not for
  abandoning the surface. Walk-forward, which does select naively, still
  earns +0.58 / +0.49.
* This is now the third pass over the same five price series. The
  cross-instrument split (indices work, commodity/stock/FX do not) is the
  thing most likely to be a sample artefact, and the thing NIFTY data would
  settle fastest.

---

# v4 — NIFTY, BANK NIFTY and 15 constituents

**It does not transfer. The Indian indices are not mean-reverting, and the
strategy is a mean-reversion strategy.**

## Data

NSE end-of-day bars from `BennyThadikaran/eod2_data` (public GitHub, pinned
at `8761629d`), reached through the anonymous git lane — NSE's own archives
are 403 at this sandbox's egress proxy like every other market-data host.

`data_in.py` screens every series for unadjusted corporate actions before
use: a split the vendor never adjusted shows up as a one-day return near
−50%/−67%/−90% and would hand a band-break rule a free fictional signal.
Six were caught — **ITC −92.7% on 2005-09-21** (1:10 split), INFY −74.5%
on 2004-07-01, RELIANCE, BHARTIARTL, KOTAKBANK, ASIANPAINT — and each
series starts the bar after its last event rather than being discarded.

| | Series | Span | Bars |
|---|---|---|---|
| Indices | NIFTY 50, NIFTY BANK, NIFTY 500 | 2012-02 → 2026-08 | 3,583 each |
| Stocks | 15 majors (RELIANCE, HDFCBANK, TCS, INFY, ITC, SBIN …) | 1995 → 2026, post-trim | 3,944 – 7,828 |

## The result

Spec R5 — the DevLucem 2σ fade, gated to RegDet SIDEWAYS, capped at the
2.0σ outer band; the exact configuration that scored +0.73 / +0.51 on the
S&P and Nasdaq:

| | R5 Sharpe | Buy & hold | Beta | Alpha t (NW) |
|---|---|---|---|---|
| **NIFTY 50** | **−0.281** | +0.719 | 0.237 | **−2.07** |
| **BANK NIFTY** | **−0.314** | +0.632 | 0.176 | −1.78 |
| NIFTY 500 | +0.013 | +0.813 | 0.048 | −0.29 |
| 15 stocks (mean) | +0.055 | +0.652 | 0.014 | +0.14 |

Positive on 9 of 18. One stock clears t > 1.96 (TCS, 2.32) — which is
exactly what 15 independent tests deliver by chance. NIFTY 50's alpha is
significantly **negative**.

## Why — measured, not asserted

`mechanism.py` computes the same four diagnostics on all 23 series. The one
that matters is the **variance ratio**: VR(k) = Var(k-bar return) /
(k × Var(1-bar return)). Below 1 the series reverts; above 1 it trends. A
band-fade is a direct bet on VR < 1.

| Class | AC(1) | VR(5) | VR(20) | Lower-band break: 5-day excess | t |
|---|---|---|---|---|---|
| **US indices** | **−0.052** | **0.857** | **0.753** | **+0.40%** | **+2.43** |
| **Indian indices** | **+0.020** | **1.026** | **1.071** | **−0.21%** | −0.91 |
| Indian stocks | +0.016 | 0.985 | 0.946 | +0.08% | +0.35 |
| US stock (GOOG) | +0.006 | 1.007 | 1.075 | −0.59% | −1.62 |
| **FX (EUR/USD)** | **−0.012** | **0.986** | **0.957** | **−0.01%** | −0.43 |
| WTI | −0.017 | 0.920 | 0.799 | −0.13% | −0.69 |

Read the second and last columns together and the whole cross-instrument
pattern falls out:

* **US indices are the outlier, not the norm.** VR(20) = 0.75 means a
  20-day move is a quarter smaller than a random walk implies. That is a
  large, real reversion effect, and the event study confirms it pays: after
  price closes below the DevLucem lower band, the next five days beat the
  unconditional drift by 0.40%, t = 2.43. The strategy is harvesting that
  and nothing else.
* **FX is a random walk.** VR(5) = 0.986, VR(20) = 0.957, AC(1) = −0.012,
  band-break excess −0.01% with t = −0.43. There is no reversion to fade
  and no trend to follow at this horizon — so the fade collects costs and
  whatever drift it happens to lean against. EUR/USD's −2.6 Sharpe is not
  the strategy being wrong about FX; it is the strategy trading noise in a
  sample that happened to trend.
* **Indian indices trend.** VR(20) = 1.07 and AC(1) is *positive*. A NIFTY
  band break is followed by continuation, not reversion — in 2019-2026 the
  lower-band excess is −0.62% at t = −2.09, i.e. significantly the wrong
  way. Fading it is backwards.
* **Single stocks reprice.** GOOG VR(20) = 1.075. A single name's move
  through a band is usually news being priced, which does not come back.
  Indian stocks sit in between (VR(20) = 0.95) and are mildly fadeable on
  the upper band only (mean t = +1.76), which is a short-rallies effect,
  not a dip-buying one.

### The explanation that does *not* survive

The standard story for index mean-reversion is the leverage effect —
panics spike volatility, the fade sells the variance premium. Measured,
that story fails: leverage correlation is −0.152 for US indices and
−0.144 for Indian ones, essentially identical. India has the same
vol-return asymmetry and none of the reversion. What differs is the
**bounce**: the 5-day return after a 1-sd down day minus that after a 1-sd
up day is **+0.48%** for US indices and **−0.17%** for Indian ones. US
down-days bounce; Indian down-days keep going.

### And it is the market, not the era

The obvious objection — the US sample is 1999-2018, the Indian one
2012-2026 — is testable on the overlapping window:

| 2012–2018 only | VR(20) | AC(1) | Lower-break t | R5 Sharpe |
|---|---|---|---|---|
| S&P 500 | 0.676 | −0.009 | +2.63 | **+0.557** |
| Nasdaq | 0.709 | −0.002 | +3.24 | **+0.814** |
| NIFTY 50 | 0.952 | +0.071 | +1.41 | **−0.308** |
| BANK NIFTY | 1.045 | +0.076 | +1.36 | **−0.493** |

Same seven years, opposite behaviour. The split is between markets, not
between periods.

## What this means for the hypothesis

The v3 read was "it works on equity indices". The correct read is narrower:
**it works on US equity indices, because those specific series mean-revert
at the 5-20 day horizon and most things do not.** NIFTY and BANK NIFTY —
the instruments this system is actually built to trade — are on the wrong
side of that line, and the strategy's alpha there is negative.

Two consequences worth acting on:

1. **Screen before you build.** VR(20) and the band-break event study cost
   nothing and are computable on any candidate instrument before a single
   line of strategy code. Anything with VR(20) ≥ 1 should never see a fade.
   Both numbers are in `mechanism.py`.
2. **For NIFTY, the sign is the finding.** Positive AC(1), VR(20) > 1 and a
   significantly negative lower-band excess in 2019-2026 all say the same
   thing: Indian indices continue after a band break. That is a
   *trend-following* setup on the same indicator, and it is the version
   worth testing next — the exact opposite of the fade.

---

## v5 — the band-to-band fade, on the timeframe it was drawn on

The spec changed, and so did the data. This pass tests the strategy exactly
as specified from the settings panels, on **NSE hourly bars**, because the
chart is a 1-hour chart and a 20-bar band means a different thing on every
timeframe.

### The spec

| | |
|---|---|
| Bollinger (RegDet+VC) | length 20, SMA basis, source close, **outer 2.1 sd**, **inner 0.3 sd**, offset 0 |
| Regression channel | **LRC_SH, length 400** |
| Short | when price touches the **outer upper** band |
| Cover | when price touches the **inner upper** band |
| Long | when price touches the **outer lower** band |
| Sell | when price touches the **inner lower** band |
| Filter | trade only while the **blue line** is inside the regression channel |
| Stop | none |

**Which line is blue** is settled from the indicator source, not the screenshot.
Of the five Bollinger plots exactly one is blue:

```
BB Upper        #F23645  red          BB Inner Lower  #089981  green, dim
BB Inner Upper  #F23645  red, dim     BB Lower        #089981  green
BB Basis        #2962FF  BLUE   <-- the 20-period SMA
```

So the gate is: *the 20-SMA must lie between the two regression-channel lines.*

**The one number not observed.** LRC_SH exposes only `Length`; its deviation
multiplier is hard-coded in a script not in evidence. The channel's width *is*
the gate, so the multiplier is swept from 0.5 to 3.0 everywhere below and
nothing is fitted to it. At 2.0 the gate is open 87–90% of the time.

### The data, and what was wrong with it

Every market-data host is blocked at the egress proxy, so hourly bars were
built from 1-minute prints in a public archive
(`aeron7/nifty-banknifty-intraday-data`), resampled to the NSE session
(09:15–15:30, seven bars a day, verified at 6.99–7.00 bars/session).

Four separate defects had to be fixed before any number was worth printing:

| Defect | Example | Fix |
|---|---|---|
| Foreign prices in the file | BANKNIFTY 2015-06-24 09:15 opens at 18403 with a low of **1439** | drop the whole session |
| Interleaved adjusted/unadjusted prints | ICICIBANK Jun-2017 alternating between two levels 10% apart | drop sessions with a move undone by the next bar |
| **A two-year hole** | stock files run 2013-06→2014-06, then nothing until 2016-09 | cut at every calendar gap, keep the longest run |
| Unadjusted corporate actions | RELIANCE 1:1 Sep-17, TCS 1:1 May-18, INFY 1:1 Sep-18, HDFCBANK 1:2 Sep-19, LT 1:2 Jul-17 | back-adjust to the snapped canonical ratio |

The hole was the dangerous one: a 400-bar regression fitted across it is
fitted across a two-year move that never appeared on any chart, and splits
that happened *inside* the hole never show up as a single-day ratio at all.
Before the fix, BANKNIFTY showed a `long_pnl` of +22.2 — one fake −92%/+1199%
pair.

**Independent validation.** Cleaned hourly bars, collapsed to daily, against
the unrelated `eod2` daily feed: price ratios constant to **0.2%** (fixed
historical scale factors only), daily return correlation **0.983–0.991**,
median daily difference 14–22 bps — which is the closing-auction gap between
a 15:29 bar close and the official close. `results/.../29_v5_data_validation.csv`

Final universe: NIFTY (22,777 bars, 2009–2023), BANKNIFTY (22,444, 2010–2023),
15 large caps (≈10,600 each, 2017–2023).

### Execution

Signal-at-t, fill-at-t+1 throws away the snap-back the strategy exists to
capture. The reported engine uses **resting limit orders at the bands**: the
levels for bar *t* are fixed by bars ≤ *t−1*, so an order is already sitting
there when the bar opens. Fills are taken *at* the level even when the bar
gapped clean through it, which is the conservative side in all four cases.
Verified: truncating the series leaves earlier P&L bit-identical, and every
trade reconciles by hand to (exit level − entry level)/entry level − costs.

### The result

**Nothing works, anywhere, under any setting.**

| | mean Sharpe |
|---|---|
| NSE 1h, 17 instruments, 2 bps/side | **−1.34** (0 of 17 positive) |
| … at zero cost | −1.12 |
| Parameter surface, 120 configs × 17 instruments | best config **−0.23**; **0 of 120** positive |
| Stop-loss / time-stop grid, 102 combinations | **0 of 102** positive; stops make it *worse* (−2.36 at 1 sd) |
| By calendar year | **0 of 14** years positive |
| Alpha after market beta (Newey-West, 20 lags) | −0.35 to −9.97 t; every instrument negative |
| Buy-and-hold on the same bars | +0.42 to +1.01 |

The blue-line gate is not the problem — it helps, slightly and consistently
(+0.03 to +0.10 Sharpe at dev ≥ 1.5, +0.25 at dev 0.5), purely by trading less.

### Where the money goes

The trade shape is the whole story:

```
win rate        64.7%          <- you are right about two times in three
average win     +0.67%
average loss    -1.97%         <- and wrong three times as expensively
payoff ratio     0.34
expectancy      -0.26% per trade
profit factor    0.41   Kelly -0.34   max drawdown -98%
```

At a 64.7% win rate, break-even needs a payoff ratio of **0.55**. It is 0.34.
Equivalently: at the payoff ratio you have, you would need to win **74.7%** of
trades, not 64.7%.

Three Monte Carlo controls locate the cause exactly:

| Control | Result | Reading |
|---|---|---|
| **Trade bootstrap** (20k resamples) | p(positive) < 0.01 on 13/17; trade t −0.6 to −9.1 | not luck |
| **Rotated entry** (200 rotations; same trade count, side mix and exits, only the timing randomised) | real −3.51 vs rotated −3.98 | the bands *do* pick better moments than random, by +0.47 — nowhere near enough |
| **Synthetic market** (150 block bootstraps; same vol, same tails, serial structure destroyed) | real **−1.34** vs synthetic **−1.41** | the market contributes nothing |
| **Synthetic, drift removed** | **−1.37** | and it is not "shorting a bull market" either |

The strategy scores the same on real NSE data as on data with every trace of
structure removed, drift included. **The loss is the payoff geometry, not the
market.**

That is confirmed independently, without reference to any exit rule, by an
event study on the entry itself: excess return in the fade direction after a
band touch, over ~26,000 events per cell, at 5/10/20-bar horizons, with and
without the gate:

```
excess  -0.03%  to  +0.08%          t  -0.58  to  +0.15
7 of 17 instruments positive
```

There is no predictive content in a band touch on these series at all.

### What this does and does not say

- It does **not** say the indicators are wrong. The gate helps; the band level
  is worth ~2.2 Sharpe as an *execution* reference versus entering at the same
  bar's open. Both are real, and both are small.
- It **does** say that a fixed 1.8-sigma target against an unbounded loss is a
  losing shape unless the entry is right about three times in four, and this
  entry is right about two times in three.
- A breakout mirror is **not** reported here. Negating the position is a clean
  mirror only for close-to-close P&L (there it flips −0.32 to +0.32 gross, which
  still does not clear 2.3 bps of cost). With resting orders it is a different
  strategy needing its own exit rule, and inventing one would be reporting a
  strategy nobody specified.
