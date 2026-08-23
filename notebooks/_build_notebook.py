"""Generates the Kaggle parameter-sweep notebook."""
import json

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().split('\n'),
    })


def code(text):
    lines = text.strip('\n').split('\n')
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    })


# Fix trailing newlines in source lists (nbformat wants \n on all but last)
def finalise(cell_list):
    for c in cell_list:
        src = c["source"]
        c["source"] = [l + '\n' for l in src[:-1]] + [src[-1]] if src else []
    return cell_list


md("""
# Sherm Quanty — Mean Reversion & Momentum Parameter Sweep

**Purpose.** Find parameters by measurement instead of by intuition.

Every number currently in the Sherm Quanty config was invented — lookback
windows, entry thresholds, holding periods, stop distances. This notebook
replaces guesses with measured values, using real forex history.

**Run this on Kaggle** (Internet must be ON in the notebook settings, right
panel → Internet → On). The development sandbox has no route to market data,
which is why this exists as a notebook rather than as a script in the repo.

---

## What this does

1. Downloads real daily OHLC for 8 symbols (7 FX majors + gold)
2. **Validates the data is genuinely real** — a synthetic fallback would make
   every result meaningless, so we check rather than assume
3. Measures realised volatility per symbol (this answers, with data, how much
   wider gold's stop needs to be than a currency's)
4. Sweeps two edge families across lookback × threshold × holding period
5. Tests four gate settings, including the "no mean reversion in high
   volatility" hypothesis
6. Splits train/test so the chosen parameters are validated on data the search
   never saw
7. Applies a multiple-comparisons correction, because testing many
   combinations produces winners by luck

## What this deliberately does NOT do

- No position sizing, no risk limits, no slot logic. Those are downstream of
  knowing whether an edge exists at all.
- No machine learning. A rule-based baseline has to exist first, otherwise
  there is no bar for a model to clear.
""")

md("""
## 1. Setup
""")

code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from itertools import product

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

print('pandas', pd.__version__)
print('numpy ', np.__version__)
""")

md("""
## 2. Download real market data

The 7 FX majors plus gold. `GC=F` is COMEX gold futures, used as the stand-in
for spot XAUUSD — close enough for measuring volatility and testing whether an
edge exists, though the futures basis means it is not identical to what a
broker quotes.
""")

code("""
SYMBOLS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'JPY=X',
    'USDCHF': 'CHF=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'CAD=X',
    'NZDUSD': 'NZDUSD=X',
    'XAUUSD': 'GC=F',
}

START = '2010-01-01'
END   = '2024-12-31'

raw = {}
for name, ticker in SYMBOLS.items():
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close']].dropna()
    df.index = pd.to_datetime(df.index)
    raw[name] = df
    print(f'{name:8s} {len(df):5d} rows   {df.index[0].date()} to {df.index[-1].date()}')
""")

md("""
### 2a. Validate the data is real

This matters more than it looks. If a download silently fails and something
substitutes generated data, every number downstream becomes noise fitted to
noise — and it will look completely plausible.

Three checks, each of which a random walk fails:

- **Fat tails.** Real FX returns have kurtosis well above 3 (the normal value).
  Big moves happen far more often than a bell curve predicts.
- **Volatility clustering.** Calm follows calm, turbulent follows turbulent.
  Measured as autocorrelation of absolute returns, which is strongly positive
  in real markets and ~0 in a random walk.
- **Known events.** March 2020 must show a volatility spike. If it does not,
  the data is not real.
""")

code("""
def validate_real(name, df):
    ret = np.log(df['close']).diff().dropna()

    kurt = stats.kurtosis(ret, fisher=False)           # normal = 3.0
    vol_cluster = ret.abs().autocorr(lag=1)            # random walk ~ 0

    covid = ret.loc['2020-03-01':'2020-03-31']
    baseline = ret.loc['2019-01-01':'2019-12-31']
    covid_ratio = (covid.std() / baseline.std()) if len(covid) and baseline.std() else np.nan

    checks = {
        'fat_tails':      kurt > 3.5,
        'vol_clustering': vol_cluster > 0.05,
        'covid_spike':    covid_ratio > 1.3,
    }
    return kurt, vol_cluster, covid_ratio, checks
""")

code("""
print(f"{'symbol':8s} {'kurtosis':>9s} {'volclust':>9s} {'covid x':>8s}   verdict")
print('-' * 60)
all_real = True
for name, df in raw.items():
    kurt, vc, cr, checks = validate_real(name, df)
    ok = all(checks.values())
    all_real &= ok
    failed = '' if ok else '  FAILED: ' + ','.join(k for k, v in checks.items() if not v)
    print(f'{name:8s} {kurt:9.2f} {vc:9.3f} {cr:8.2f}   {"REAL" if ok else "SUSPECT"}{failed}')

print()
if all_real:
    print('All symbols pass. Data is real; results below are meaningful.')
else:
    print('*** STOP. At least one symbol looks synthetic or corrupt. ***')
    print('*** Do not trust any parameter derived below.             ***')
""")

md("""
## 3. Measure volatility per symbol

This replaces an invented number with a measured one.

The config currently gives gold a 2.0x stop multiplier, on the basis that
"gold is 2-3x more volatile than a currency". That was asserted from memory,
never checked. Here it gets checked.

The ratio computed below is what the multiplier should actually be.
""")

code("""
vol_table = []
for name, df in raw.items():
    ret = np.log(df['close']).diff().dropna()
    ann = ret.std() * np.sqrt(252)
    daily_range = ((df['high'] - df['low']) / df['close']).mean()

    # ATR(14) as a percentage of price — the practical measure for stops
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14).mean() / df['close']).mean()

    vol_table.append({
        'symbol': name,
        'ann_vol_pct': ann * 100,
        'avg_daily_range_pct': daily_range * 100,
        'atr14_pct': atr_pct * 100,
    })

vol_df = pd.DataFrame(vol_table).set_index('symbol').sort_values('ann_vol_pct')

# Everything relative to the least volatile major, so the numbers are
# directly usable as stop multipliers.
majors = vol_df.drop('XAUUSD')
base = majors['atr14_pct'].median()
vol_df['stop_multiplier'] = vol_df['atr14_pct'] / base

print(vol_df.round(3))
print()
print(f"Median major ATR14: {base:.3f}% of price")
gold_mult = vol_df.loc['XAUUSD', 'stop_multiplier']
print(f"Gold multiplier MEASURED: {gold_mult:.2f}x   (config currently asserts 2.00x)")
""")

md("""
## 4. Features

All indicators are parameterised by lookback, because the lookback is exactly
what we are trying to determine. Nothing here hardcodes 21 days.

Everything is strictly backward-looking. A feature computed for day *T* uses
only data up to and including day *T*'s close — otherwise the backtest sees
the future and every result is fiction.
""")

code("""
def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df, lookback):
    \"\"\"Features for one symbol at one lookback setting.\"\"\"
    out = pd.DataFrame(index=df.index)
    close = df['close']
    out['close'] = close
    out['ret_1d'] = np.log(close).diff()

    # --- Family A: distance from the mean ---
    ma  = close.rolling(lookback).mean()
    sd  = close.rolling(lookback).std()
    out['z_std'] = (close - ma) / sd                 # == Bollinger position

    a = atr(df, lookback)
    out['z_atr'] = (close - ma) / a                  # ATR-normalised variant

    # --- Family B: oscillators ---
    out['rsi'] = rsi(close, max(2, lookback // 5))
    lo = df['low'].rolling(lookback).min()
    hi = df['high'].rolling(lookback).max()
    out['stoch'] = 100 * (close - lo) / (hi - lo)

    # --- Momentum ---
    out['mom'] = close.pct_change(lookback)
    out['mom_rsi'] = rsi(close, 14)

    # --- Gate inputs ---
    out['vol_short'] = out['ret_1d'].rolling(5).std()
    out['vol_long']  = out['ret_1d'].rolling(20).std()
    out['vol_ratio'] = out['vol_short'] / out['vol_long']
    out['sma200'] = close.rolling(200).mean()
    out['above_sma200'] = (close > out['sma200']).astype(int)

    return out
""")

md("""
## 5. Gates

Testing the hypothesis that mean reversion needs calm conditions, and should
switch off when volatility expands (which suggests a trend has begun).

Four settings, treated as a swept parameter rather than an assumption. `none`
is the control — without it there is no way to tell whether gating helps at
all.
""")

code("""
GATES = {
    'none':          lambda f: pd.Series(True, index=f.index),
    'vol_calm':      lambda f: f['vol_ratio'] < 1.0,    # vol not expanding
    'vol_very_calm': lambda f: f['vol_ratio'] < 0.8,    # strictly calm
    'vol_expanding': lambda f: f['vol_ratio'] > 1.2,    # inverse — sanity check
}
""")

md("""
The `vol_expanding` gate is deliberately the opposite of the hypothesis. If
the hypothesis is right it should perform *worse* than `vol_calm`. If instead
it performs better, the hypothesis is backwards — and that is worth finding
out now rather than after deploying.
""")

md("""
## 6. Strategies

Two families of mean reversion plus momentum. Each returns entry signals as
+1 (long), -1 (short) or 0 (flat).
""")

code("""
def signal_distance(f, threshold, measure='z_std'):
    \"\"\"Family A. Fade the move when price is far from its mean.\"\"\"
    z = f[measure]
    return np.select([z < -threshold, z > threshold], [1, -1], default=0)


def signal_oscillator(f, threshold, measure='rsi'):
    \"\"\"Family B. Fade the move at oscillator extremes.\"\"\"
    o = f[measure]
    lo, hi = threshold, 100 - threshold
    return np.select([o < lo, o > hi], [1, -1], default=0)


def signal_momentum(f, threshold, measure='mom'):
    \"\"\"Trend continuation. Long strength, short weakness.\"\"\"
    m = f[measure]
    confirm_long  = f['mom_rsi'] > 50
    confirm_short = f['mom_rsi'] < 50
    return np.select(
        [(m > threshold) & confirm_long, (m < -threshold) & confirm_short],
        [1, -1], default=0)


STRATEGIES = {
    'mr_distance_std': dict(fn=signal_distance,   measure='z_std',
                            thresholds=[1.5, 2.0, 2.5]),
    'mr_distance_atr': dict(fn=signal_distance,   measure='z_atr',
                            thresholds=[1.5, 2.0, 2.5]),
    'mr_rsi':          dict(fn=signal_oscillator, measure='rsi',
                            thresholds=[10, 20, 30]),
    'mr_stoch':        dict(fn=signal_oscillator, measure='stoch',
                            thresholds=[10, 20, 30]),
    'momentum':        dict(fn=signal_momentum,   measure='mom',
                            thresholds=[0.0, 0.01, 0.02]),
}
""")

md("""
## 7. Backtest engine

Deliberately simple: enter on the signal, exit after N days, record the
return. No stops, no sizing, no compounding.

That is intentional. We are asking one question — *does this signal predict
direction?* — and adding stops or position sizing would blend the answer with
questions about risk management that come later. Trades are equal-weighted so
each one contributes the same evidence.

**Costs are applied.** A round-trip spread is subtracted from every trade. At
these horizons costs are a large fraction of the edge, and a backtest without
them is systematically flattering.
""")

code("""
# Typical round-trip cost as a fraction of price. Deliberately conservative:
# real retail spreads vary, and an edge that only survives at zero cost is
# not an edge.
COST = {
    'EURUSD': 0.00010, 'GBPUSD': 0.00012, 'USDJPY': 0.00010,
    'USDCHF': 0.00012, 'AUDUSD': 0.00012, 'USDCAD': 0.00013,
    'NZDUSD': 0.00015, 'XAUUSD': 0.00040,
}


def backtest(f, signals, holding, cost, gate_mask=None):
    \"\"\"Equal-weighted, non-overlapping trades. Returns a Series of trade P&L.\"\"\"
    sig = pd.Series(signals, index=f.index)
    if gate_mask is not None:
        sig = sig.where(gate_mask, 0)

    fwd = (f['close'].shift(-holding) / f['close'] - 1)

    trades = []
    i = 0
    idx = f.index
    n = len(idx)
    sig_v = sig.values
    fwd_v = fwd.values

    # Non-overlapping: once in a trade, ignore signals until it closes.
    # Overlapping trades would count the same market move several times and
    # inflate the apparent sample size.
    while i < n - holding:
        s = sig_v[i]
        if s != 0 and not np.isnan(fwd_v[i]):
            trades.append(s * fwd_v[i] - cost)
            i += holding
        else:
            i += 1
    return pd.Series(trades, dtype=float)


def summarise(trades):
    n = len(trades)
    if n < 10:
        return dict(n=n, win_rate=np.nan, avg_return=np.nan,
                    total=np.nan, sharpe=np.nan, p_value=np.nan)
    wins = (trades > 0).mean()
    avg  = trades.mean()
    sharpe = (avg / trades.std() * np.sqrt(252 / max(1, n))) if trades.std() else np.nan
    _, p = stats.ttest_1samp(trades, 0)
    return dict(n=n, win_rate=wins, avg_return=avg,
                total=trades.sum(), sharpe=sharpe, p_value=p)
""")

md("""
## 8. Walk-forward split

Parameters are chosen on the **train** period only. The winner is then checked
against **test** — data the search has never touched.

This is the guard against the luck problem. Testing 60 combinations produces
some that look good by chance; roughly 5% of pure noise clears a 5%
significance bar. A setting that works in train *and* holds up in test is much
harder to explain as luck.

If a parameter looks brilliant in train and falls apart in test, it was noise.
That outcome is a success of the method, not a failure.
""")

code("""
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'

LOOKBACKS = [10, 21, 40, 60, 90]
HOLDINGS  = [2, 5, 10, 15]

print(f'Train: {START} to {TRAIN_END}')
print(f'Test:  {TEST_START} to {END}')
print(f'Grid per strategy: {len(LOOKBACKS)} lookbacks x 3 thresholds '
      f'x {len(HOLDINGS)} holds x {len(GATES)} gates '
      f'= {len(LOOKBACKS)*3*len(HOLDINGS)*len(GATES)} combinations')
""")

md("""
## 9. Run the sweep

Every combination is run on all 8 symbols and the trades pooled. Pooling means
a parameter has to work broadly rather than on one lucky pair.
""")

code("""
def run_sweep(period_start, period_end):
    rows = []
    for strat_name, spec in STRATEGIES.items():
        for lookback, threshold, holding, gate_name in product(
                LOOKBACKS, spec['thresholds'], HOLDINGS, GATES):

            pooled = []
            for sym, df in raw.items():
                f = build_features(df, lookback)
                f = f.loc[period_start:period_end]
                if len(f) < 250:
                    continue
                sig  = spec['fn'](f, threshold, spec['measure'])
                mask = GATES[gate_name](f)
                pooled.append(backtest(f, sig, holding, COST[sym], mask))

            if not pooled:
                continue
            trades = pd.concat(pooled, ignore_index=True)
            rows.append({
                'strategy': strat_name, 'lookback': lookback,
                'threshold': threshold, 'holding': holding, 'gate': gate_name,
                **summarise(trades),
            })
    return pd.DataFrame(rows)


train_results = run_sweep(START, TRAIN_END)
print(f'{len(train_results)} combinations tested on train period')
train_results.head()
""")

md("""
## 10. Multiple-comparisons correction

Having tested several hundred combinations, some will look significant purely
by chance. The Benjamini-Hochberg procedure controls the false discovery rate
— roughly, it answers "of the results I am calling significant, what fraction
are likely to be flukes?"

Without this step, the best-looking result is very likely to be the luckiest
rather than the best.
""")

code("""
def bh_correct(pvalues, fdr=0.10):
    p = np.asarray(pvalues, dtype=float)
    ok = ~np.isnan(p)
    out = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    passed = 0
    for rank, i in enumerate(order, start=1):
        if p[i] <= fdr * rank / m:
            passed = rank
    out[order[:passed]] = True
    return out
""")

code("""
valid = train_results[train_results['n'] >= 30].copy()
valid['significant'] = bh_correct(valid['p_value'].values, fdr=0.10)

print(f'{len(valid)} combinations with >= 30 trades')
print(f'{valid["significant"].sum()} survive BH correction at 10% FDR')
print()
print('Top 15 by average return (train period, significant only):')
top = (valid[valid['significant']]
       .sort_values('avg_return', ascending=False)
       .head(15))
print(top[['strategy','lookback','threshold','holding','gate',
           'n','win_rate','avg_return','sharpe','p_value']].to_string(index=False))
""")

md("""
## 11. Out-of-sample validation

The real test. Take the settings that looked best on train, and run them on
test — data the search never saw.

Read the `oos_ratio` column carefully:

- **near 1.0** — the edge held up. Trustworthy.
- **near 0 or negative** — it was noise. The train result was luck.
- **much greater than 1** — probably also luck, in the other direction. Do not
  celebrate it.
""")

code("""
test_results = run_sweep(TEST_START, END)

merged = valid[valid['significant']].merge(
    test_results,
    on=['strategy', 'lookback', 'threshold', 'holding', 'gate'],
    suffixes=('_train', '_test'))

merged['oos_ratio'] = merged['avg_return_test'] / merged['avg_return_train']
merged['held_up'] = (merged['avg_return_test'] > 0) & (merged['oos_ratio'] > 0.5)

survivors = merged.sort_values('avg_return_train', ascending=False)

print(f'{len(merged)} significant train combinations checked out of sample')
print(f'{merged["held_up"].sum()} held up out of sample')
print()
cols = ['strategy','lookback','threshold','holding','gate',
        'n_train','win_rate_train','avg_return_train',
        'n_test','win_rate_test','avg_return_test','oos_ratio','held_up']
print(survivors[cols].head(20).to_string(index=False))
""")

md("""
## 12. Does the volatility gate actually help?

Your hypothesis was that mean reversion needs calm conditions. This compares
gate settings directly, holding everything else constant.

If `vol_calm` beats `none`, the hypothesis is supported. If `none` wins, mean
reversion works regardless of volatility and the gate only removes trades. If
`vol_expanding` wins, the hypothesis is backwards.
""")

code("""
mr_only = valid[valid['strategy'].str.startswith('mr_')]

gate_summary = (mr_only.groupby('gate')
                .agg(combos=('n', 'size'),
                     median_trades=('n', 'median'),
                     mean_win_rate=('win_rate', 'mean'),
                     mean_return=('avg_return', 'mean'),
                     pct_significant=('significant', 'mean'))
                .sort_values('mean_return', ascending=False))

print('Mean-reversion performance by gate (train period):')
print(gate_summary.round(4))
print()

best = gate_summary.index[0]
if best == 'none':
    print('VERDICT: gating did not help. Mean reversion worked regardless of')
    print('volatility; the gate only removed trades.')
elif best in ('vol_calm', 'vol_very_calm'):
    print(f'VERDICT: hypothesis SUPPORTED. {best} outperformed ungated.')
    print('Mean reversion does prefer calm conditions.')
else:
    print('VERDICT: hypothesis appears BACKWARDS — mean reversion did better')
    print('when volatility was expanding. Worth investigating before relying on it.')
""")

md("""
## 13. Momentum vs mean reversion by regime

Your intuition was that expanding volatility signals a trend, so momentum
should work where mean reversion does not. This checks whether the two edges
are complementary — which matters, because two edges that fire in the same
conditions are one edge in two costumes.
""")

code("""
comparison = (valid.assign(family=np.where(
                    valid['strategy'] == 'momentum', 'momentum', 'mean_reversion'))
              .groupby(['family', 'gate'])
              .agg(mean_return=('avg_return', 'mean'),
                   mean_win_rate=('win_rate', 'mean'),
                   n_significant=('significant', 'sum'))
              .round(4))
print(comparison)
""")

md("""
## 14. Recommended parameters

What to bring back to the repo. These replace the invented values in
`config.py`.

Anything that failed out-of-sample validation is excluded — a parameter that
only worked in the search period is not a parameter, it is a coincidence.
""")

code("""
held = survivors[survivors['held_up']]

print('=' * 70)
print('RECOMMENDED PARAMETERS')
print('=' * 70)

if len(held) == 0:
    print()
    print('NO combination survived out-of-sample validation.')
    print()
    print('This is a real result, not a bug. It means that on this data, with')
    print('these costs, neither edge family showed a durable effect. Options:')
    print('  - widen the parameter grid (different lookbacks may help)')
    print('  - test intraday timeframes; these edges may not live at daily')
    print('  - revisit whether costs are set too conservatively')
    print('  - accept that the edge is not there and look elsewhere')
    print()
    print('Do NOT pick the best train-period result and use it anyway.')
else:
    for family in ['mean_reversion', 'momentum']:
        subset = held[held['strategy'] == 'momentum'] if family == 'momentum' \\
                 else held[held['strategy'].str.startswith('mr_')]
        if len(subset) == 0:
            print(f'\\n{family}: nothing survived.')
            continue
        best = subset.iloc[0]
        print(f'\\n{family.upper()}')
        print(f"  strategy      {best['strategy']}")
        print(f"  lookback      {best['lookback']}")
        print(f"  threshold     {best['threshold']}")
        print(f"  holding       {best['holding']} days")
        print(f"  gate          {best['gate']}")
        print(f"  train         {best['n_train']:.0f} trades, "
              f"{best['win_rate_train']:.1%} win, {best['avg_return_train']:.4%} avg")
        print(f"  test          {best['n_test']:.0f} trades, "
              f"{best['win_rate_test']:.1%} win, {best['avg_return_test']:.4%} avg")

print()
print('=' * 70)
print('MEASURED STOP MULTIPLIERS  (replaces the asserted gold 2.0x)')
print('=' * 70)
print(vol_df[['atr14_pct', 'stop_multiplier']].round(3))
""")

md("""
## 15. Export

Saves the full sweep so results can be examined without re-running, and a
compact summary to bring back to the repo.
""")

code("""
train_results.to_csv('sweep_train.csv', index=False)
test_results.to_csv('sweep_test.csv', index=False)
survivors.to_csv('sweep_survivors.csv', index=False)
vol_df.to_csv('volatility_measured.csv')

print('Written:')
print('  sweep_train.csv         all train combinations')
print('  sweep_test.csv          all test combinations')
print('  sweep_survivors.csv     significant + out-of-sample checked')
print('  volatility_measured.csv per-symbol volatility and stop multipliers')
""")

md("""
## 16. Visual check

Parameter surfaces. A real edge shows a smooth region of good performance —
neighbouring parameter values should behave similarly, because the market does
not care that you rounded 21 to 20.

An isolated bright spot surrounded by poor results is the signature of
overfitting: it means that exact combination got lucky, and a slightly
different one would not have.
""")

code("""
mr_best_gate = gate_summary.index[0]
subset = valid[(valid['strategy'] == 'mr_distance_std') &
               (valid['gate'] == mr_best_gate)]

if len(subset) > 0:
    fig, axes = plt.subplots(1, len(HOLDINGS), figsize=(20, 4), sharey=True)
    for ax, hold in zip(np.atleast_1d(axes), HOLDINGS):
        s = subset[subset['holding'] == hold]
        if len(s) == 0:
            continue
        pivot = s.pivot_table(index='lookback', columns='threshold',
                              values='avg_return')
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                       vmin=-0.004, vmax=0.004)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f'{hold}-day hold')
        ax.set_xlabel('threshold')
    np.atleast_1d(axes)[0].set_ylabel('lookback')
    fig.suptitle(f'Mean reversion (distance/std), gate={mr_best_gate} — '
                 f'avg return per trade, train period')
    plt.colorbar(im, ax=axes, fraction=0.02)
    plt.show()
else:
    print('No data to plot.')
""")

md("""
---

## What to bring back

1. **The recommended parameters** from section 14 — or the finding that nothing
   survived, which is equally useful and much cheaper to learn now.
2. **The measured stop multipliers** from section 3, replacing the asserted 2.0x
   for gold.
3. **The gate verdict** from section 12 — whether the volatility hypothesis held.

A note on reading section 14: if nothing survives out-of-sample, resist the
urge to use the best train result anyway. The whole point of the split is to
make that temptation resistible. Better to know the edge is not there than to
deploy a coincidence.
""")

nb = {
    "cells": finalise(cells),
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = '/home/user/SHERM-QUANTY/notebooks/01_parameter_sweep.ipynb'
with open(path, 'w') as fh:
    json.dump(nb, fh, indent=1)
print(f'Wrote {path} ({len(cells)} cells)')
