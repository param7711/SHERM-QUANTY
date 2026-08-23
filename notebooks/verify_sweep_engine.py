"""
Verify the notebook's engine is CORRECT, using data where the answer is known.

This cannot tell us what the right parameters are — that needs real market
data. It can tell us whether the code that will derive them works, which is
the part that would otherwise ship unverified.

Three properties worth proving:
  1. The data validator rejects synthetic data (else a silent fallback poisons
     everything downstream)
  2. The backtest finds an edge that is definitely there
  3. The backtest finds nothing in a random walk
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

# Pull the notebook's own code so we test what will actually run, rather than
# a reimplementation that might differ.
nb = json.load(open('/home/user/SHERM-QUANTY/notebooks/01_parameter_sweep.ipynb'))
src = '\n'.join(
    ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')

ns = {}
exec("import warnings; warnings.filterwarnings('ignore')", ns)
exec('import numpy as np, pandas as pd\nfrom scipy import stats\n'
     'from itertools import product', ns)

# Execute only the definition cells (skip anything needing network or globals
# built by earlier data cells).
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    code = ''.join(cell['source'])
    if any(tok in code for tok in ('yf.download', 'raw[', 'raw.items()',
                                   'train_results', 'valid[', 'vol_df',
                                   'plt.', 'to_csv', 'run_sweep(')):
        continue
    try:
        exec(code, ns)
    except Exception as e:
        print(f'  [skip cell] {type(e).__name__}: {e}')

required = ['build_features', 'backtest', 'summarise', 'signal_distance',
            'signal_momentum', 'signal_oscillator', 'GATES', 'bh_correct',
            'validate_real', 'rsi', 'atr']
missing = [r for r in required if r not in ns]
print(f'Loaded from notebook: {len(required) - len(missing)}/{len(required)} '
      f'{"(missing: " + ",".join(missing) + ")" if missing else ""}')
print()

build_features = ns['build_features']
backtest       = ns['backtest']
summarise      = ns['summarise']
signal_distance = ns['signal_distance']
GATES          = ns['GATES']
bh_correct     = ns['bh_correct']
validate_real  = ns['validate_real']

passed = 0
total  = 0


def check(name, ok, detail=''):
    global passed, total
    total += 1
    passed += bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def make_ohlc(close, index):
    """Wrap a close series into an OHLC frame the feature code accepts."""
    noise = np.abs(np.random.default_rng(0).normal(0, 0.0008, len(close)))
    return pd.DataFrame({
        'open':  close,
        'high':  close * (1 + noise),
        'low':   close * (1 - noise),
        'close': close,
    }, index=index)


print('=' * 68)
print('1. Does the validator reject synthetic data?')
print('=' * 68)

rng = np.random.default_rng(42)
n = 3000
dates = pd.bdate_range('2010-01-01', periods=n)

# Pure geometric Brownian motion — exactly what the sandbox fallback makes.
gbm = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
gbm_df = make_ohlc(pd.Series(gbm, index=dates), dates)
kurt, vc, cr, checks = validate_real('GBM', gbm_df)
# A random walk has normal tails and no volatility clustering.
check('random walk flagged SUSPECT', not all(checks.values()),
      f'kurtosis={kurt:.2f} volclust={vc:.3f}')

# Now data WITH fat tails and volatility clustering, like a real market.
#
# A proper GARCH(1,1). My first attempt used a mean-reverting vol process
# whose shocks were tiny relative to its mean, so vol was near-constant and
# produced no clustering at all — the test failed on its own generator, not
# on the notebook. GARCH feeds each squared shock back into the next
# period's variance, with persistence (alpha+beta) close to 1, which is what
# actually generates both stylised facts.
omega, alpha, beta = 1e-6, 0.10, 0.88     # persistence 0.98
var = np.zeros(n)
eps = np.zeros(n)
var[0] = omega / (1 - alpha - beta)
for i in range(1, n):
    var[i] = omega + alpha * eps[i-1]**2 + beta * var[i-1]
    eps[i] = rng.normal(0, 1) * np.sqrt(var[i])
realistic = 1.10 * np.exp(np.cumsum(eps))
real_df = make_ohlc(pd.Series(realistic, index=dates), dates)
kurt2, vc2, _, checks2 = validate_real('REALISTIC', real_df)
check('clustered/fat-tailed passes those two checks',
      checks2['fat_tails'] and checks2['vol_clustering'],
      f'kurtosis={kurt2:.2f} volclust={vc2:.3f}')

print()
print('=' * 68)
print('2. Does the backtest find an edge that IS there?')
print('=' * 68)

# Construct a series that genuinely mean-reverts: an Ornstein-Uhlenbeck
# process around a slow trend. Price pulled back toward its mean, so a
# distance-based fade MUST be profitable if the engine is correct.
ou = np.zeros(n)
for i in range(1, n):
    ou[i] = ou[i-1] + 0.08 * (0 - ou[i-1]) + rng.normal(0, 0.010)
mr_close = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)) + ou)
mr_df = make_ohlc(pd.Series(mr_close, index=dates), dates)

f = build_features(mr_df, 21)
sig = signal_distance(f, 2.0, 'z_std')
trades = backtest(f, sig, 5, cost=0.0)
res = summarise(trades)
check('finds edge in mean-reverting data',
      res['win_rate'] > 0.55 and res['avg_return'] > 0,
      f"n={res['n']} win={res['win_rate']:.1%} avg={res['avg_return']:.4%}")

print()
print('=' * 68)
print('3. Does it correctly find NOTHING in a random walk?')
print('=' * 68)

f_rw = build_features(gbm_df, 21)
sig_rw = signal_distance(f_rw, 2.0, 'z_std')
trades_rw = backtest(f_rw, sig_rw, 5, cost=0.0)
res_rw = summarise(trades_rw)
# A coin flip. Anything far from 50% would mean the engine invents signal.
check('random walk gives ~coin flip',
      0.42 < res_rw['win_rate'] < 0.58,
      f"n={res_rw['n']} win={res_rw['win_rate']:.1%} avg={res_rw['avg_return']:.4%}")

print()
print('=' * 68)
print('4. Mechanics')
print('=' * 68)

# Costs must actually reduce returns.
t_free = backtest(f, sig, 5, cost=0.0)
t_cost = backtest(f, sig, 5, cost=0.0010)
check('costs reduce returns',
      t_cost.mean() < t_free.mean(),
      f'free={t_free.mean():.4%} costed={t_cost.mean():.4%} '
      f'delta={t_free.mean()-t_cost.mean():.4%}')

# Non-overlapping: with a 10-day hold over N bars, trades cannot exceed N/10.
t_overlap = backtest(f, sig, 10, cost=0.0)
check('trades are non-overlapping',
      len(t_overlap) <= len(f) / 10 + 1,
      f'{len(t_overlap)} trades over {len(f)} bars, 10-day hold')

# A gate must reduce the trade count, not silently do nothing.
mask_calm = GATES['vol_calm'](f)
t_gated = backtest(f, sig, 5, cost=0.0, gate_mask=mask_calm)
check('gate reduces trade count',
      len(t_gated) < len(t_free),
      f'ungated={len(t_free)} gated={len(t_gated)} '
      f'({mask_calm.mean():.0%} of bars pass)')

# Direction: shorts must be scored inversely. On a strong uptrend, a
# short-only signal must LOSE. This is the bug class that silently inverted
# the shadow ledger earlier in the project.
up = pd.Series(1.10 * np.exp(np.cumsum(np.full(n, 0.0008))), index=dates)
up_df = make_ohlc(up, dates)
f_up = build_features(up_df, 21)
short_only = np.where(np.arange(len(f_up)) % 20 == 0, -1, 0)
t_short = backtest(f_up, short_only, 5, cost=0.0)
check('shorts scored inversely',
      t_short.mean() < 0,
      f'short-only in an uptrend returns {t_short.mean():.4%}')

# No lookahead: a feature at bar T must not change when future bars are added.
f_full  = build_features(mr_df, 21)
f_trunc = build_features(mr_df.iloc[:2000], 21)
common  = f_trunc.index[-50:]
drift = (f_full.loc[common, 'z_std'] - f_trunc.loc[common, 'z_std']).abs().max()
check('features do not look ahead',
      drift < 1e-9,
      f'max divergence on shared bars = {drift:.2e}')

print()
print('=' * 68)
print('5. Multiple-comparisons correction')
print('=' * 68)

# Under pure noise, BH at 10% FDR should pass roughly none of 500 p-values.
noise_p = rng.uniform(0, 1, 500)
n_pass = bh_correct(noise_p, fdr=0.10).sum()
check('BH rejects pure noise', n_pass <= 5, f'{n_pass}/500 passed')

# With genuine signal mixed in, it should find most of it.
mixed = np.concatenate([rng.uniform(0, 0.0001, 50), rng.uniform(0, 1, 450)])
n_found = bh_correct(mixed, fdr=0.10)[:50].sum()
check('BH detects real signal', n_found >= 40, f'{n_found}/50 true effects found')

print()
print('=' * 68)
print(f'{passed}/{total} checks passed')
print('=' * 68)
if passed == total:
    print('The sweep engine is correct: it finds edges that exist, finds')
    print('nothing in noise, applies costs, avoids lookahead, and handles')
    print('direction properly.')
    print()
    print('It does NOT tell us what the parameters should be. That needs real')
    print('market data and has to run on Kaggle.')
else:
    print('Engine has defects — fix before running on real data.')
