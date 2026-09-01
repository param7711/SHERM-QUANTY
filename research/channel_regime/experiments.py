"""
The actual hypothesis test.

Runs, in order:
  1  baseline        default parameters, every instrument, vs buy & hold
  2  attribution     which leg of the switch produces the P&L
  3  grid            648-point parameter sweep -> a plateau or a lucky spike?
  4  costs           breakeven transaction cost
  5  stability       year-by-year returns
  6  walkforward     anchored out-of-sample parameter selection
  7  pbo             probability of backtest overfitting (CSCV, Bailey et al.)
  8  dsr             deflated Sharpe, corrected for the number of trials
  9  nulls           permutation and matched-Gaussian null distributions

Everything is written to results/channel_regime/ as CSV.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import data as D
from backtest import Params, run, simulate, metrics, trade_log

RESULTS = os.path.join(D.ROOT, 'results', 'channel_regime')
INSTRUMENTS = ['SP500', 'NASDAQ', 'WTI', 'GOOG', 'EURUSD']
DEFAULT = Params()
COST_BPS = 5.0          # per unit of position change, one way
SEED = 7

GRID = {
    'n':          [50, 80, 100, 150, 200, 250],
    'k':          [1.5, 2.0, 2.5, 3.0],
    'sma_len':    [10, 20, 50],
    'entry_frac': [0.7, 0.9, 1.0],
    'exit_z':     [-0.5, 0.0, 0.5],
}


def _grid_params():
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        yield Params(**dict(zip(keys, combo)))


def _save(df: pd.DataFrame, name: str):
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, f'{name}.csv'), index=False)


def _sharpe(r: np.ndarray, bpy: int) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0


# ---------------------------------------------------------------------------
# 1-2  baseline and leg attribution
# ---------------------------------------------------------------------------

def stage_baseline() -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        close = D.load(inst)['close']
        bpy = D.BARS_PER_YEAR[inst]
        variants = {
            'switch (full hypothesis)': DEFAULT,
            'mean-reversion leg only':  Params(use_tf=False),
            'trend leg only':           Params(use_mr=False),
        }
        for label, p in variants.items():
            m = run(close, p, bpy, cost_bps=COST_BPS)
            m['instrument'], m['variant'] = inst, label
            rows.append(m)
        bh = metrics(close.pct_change().fillna(0.0), bpy, label='buy & hold')
        bh.update({'instrument': inst, 'variant': 'buy & hold',
                   'exposure': 1.0, 'n_trades': 1})
        rows.append(bh)

    cols = ['instrument', 'variant', 'sharpe', 'gross_sharpe', 'cagr', 'vol',
            'max_dd', 'calmar', 't_stat', 'exposure', 'n_trades', 'hit_rate',
            'profit_factor', 'avg_bars', 'years']
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    _save(df, '01_baseline')
    return df


def stage_leg_attribution() -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        close = D.load(inst)['close']
        bpy = D.BARS_PER_YEAR[inst]
        sig = simulate(close, DEFAULT, cost_bps=COST_BPS)
        trades = trade_log(sig)
        for leg_name, leg_id in (('MR', 1), ('TF', 2)):
            mask = sig['leg_held'] == leg_id
            leg_ret = sig['ret_net'].where(mask, 0.0)
            lt = trades[trades['leg'] == leg_name]
            rows.append({
                'instrument': inst,
                'leg': leg_name,
                'bar_share': float(mask.mean()),
                'total_pnl': float(sig['ret_net'][mask].sum()),
                'sharpe_when_on': _sharpe(sig['ret_net'][mask].to_numpy(), bpy),
                'contribution_sharpe': _sharpe(leg_ret.to_numpy(), bpy),
                'n_trades': int(len(lt)),
                'hit_rate': float((lt['pnl'] > 0).mean()) if len(lt) else np.nan,
                'avg_pnl': float(lt['pnl'].mean()) if len(lt) else np.nan,
                'avg_bars': float(lt['bars'].mean()) if len(lt) else np.nan,
            })
    df = pd.DataFrame(rows)
    _save(df, '02_leg_attribution')
    return df


# ---------------------------------------------------------------------------
# 3  parameter grid
# ---------------------------------------------------------------------------

def grid_returns(inst: str, cost_bps: float = COST_BPS):
    close = D.load(inst)['close']
    params = list(_grid_params())
    mat = np.zeros((len(close), len(params)), dtype=np.float32)
    for j, p in enumerate(params):
        mat[:, j] = simulate(close, p, cost_bps=cost_bps)['ret_net'].to_numpy()
    return close.index, mat, params


def stage_grid():
    rows, cache = [], {}
    for inst in INSTRUMENTS:
        bpy = D.BARS_PER_YEAR[inst]
        idx, mat, params = grid_returns(inst)
        cache[inst] = (idx, mat, params)
        for p in params:
            pass
        sharpes = np.array([_sharpe(mat[:, j], bpy) for j in range(mat.shape[1])])
        for p, s in zip(params, sharpes):
            rows.append({'instrument': inst, 'sharpe': s, 'n': p.n, 'k': p.k,
                         'sma_len': p.sma_len, 'entry_frac': p.entry_frac,
                         'exit_z': p.exit_z})
        print(f'  grid done: {inst}')
    df = pd.DataFrame(rows)
    _save(df, '03_grid')

    summary = (df.groupby('instrument')['sharpe']
                 .agg(configs='size', best='max', median='median',
                      mean='mean', worst='min',
                      pct_positive=lambda s: float((s > 0).mean()),
                      pct_above_0_5=lambda s: float((s > 0.5).mean()))
                 .reset_index())
    _save(summary, '03_grid_summary')
    return df, summary, cache


# ---------------------------------------------------------------------------
# 4-5  costs and stability
# ---------------------------------------------------------------------------

def stage_costs() -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        close = D.load(inst)['close']
        bpy = D.BARS_PER_YEAR[inst]
        for c in [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]:
            m = run(close, DEFAULT, bpy, cost_bps=c)
            rows.append({'instrument': inst, 'cost_bps': c,
                         'sharpe': m['sharpe'], 'cagr': m['cagr'],
                         'n_trades': m['n_trades']})
    df = pd.DataFrame(rows)
    _save(df, '04_costs')
    return df


def stage_stability() -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        close = D.load(inst)['close']
        bpy = D.BARS_PER_YEAR[inst]
        sig = simulate(close, DEFAULT, cost_bps=COST_BPS)
        for year, chunk in sig.groupby(sig.index.year):
            if len(chunk) < bpy // 4:
                continue
            rows.append({
                'instrument': inst, 'year': int(year),
                'strategy': float((1 + chunk['ret_net']).prod() - 1),
                'buy_hold': float((1 + chunk['bh']).prod() - 1),
                'sharpe': _sharpe(chunk['ret_net'].to_numpy(), bpy),
                'bars': len(chunk),
            })
    df = pd.DataFrame(rows)
    _save(df, '05_stability')
    return df


# ---------------------------------------------------------------------------
# 6  anchored walk-forward
# ---------------------------------------------------------------------------

def stage_walkforward(cache) -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        bpy = D.BARS_PER_YEAR[inst]
        idx, mat, params = cache[inst]
        t, folds = mat.shape[0], 6
        edges = np.linspace(0, t, folds + 1).astype(int)
        oos_stream = []
        for f in range(1, folds):
            is_end, oos_end = edges[f], edges[f + 1]
            is_block, oos_block = mat[:is_end], mat[is_end:oos_end]
            is_sharpe = np.array([_sharpe(is_block[:, j], bpy)
                                  for j in range(mat.shape[1])])
            best = int(np.argmax(is_sharpe))
            oos_stream.append(oos_block[:, best])
            rows.append({
                'instrument': inst, 'fold': f,
                'is_bars': int(is_end), 'oos_bars': int(oos_end - is_end),
                'is_sharpe_best': float(is_sharpe[best]),
                'oos_sharpe_selected': _sharpe(oos_block[:, best], bpy),
                'oos_sharpe_median_config': float(np.median(
                    [_sharpe(oos_block[:, j], bpy) for j in range(mat.shape[1])])),
                'selected': params[best].key(),
                'oos_start': str(idx[is_end].date()),
                'oos_end': str(idx[oos_end - 1].date()),
            })
        stitched = np.concatenate(oos_stream)
        rows.append({
            'instrument': inst, 'fold': 'ALL_OOS', 'is_bars': np.nan,
            'oos_bars': int(stitched.size), 'is_sharpe_best': np.nan,
            'oos_sharpe_selected': _sharpe(stitched, bpy),
            'oos_sharpe_median_config': np.nan,
            'selected': 'stitched walk-forward', 'oos_start': '', 'oos_end': '',
        })
    df = pd.DataFrame(rows)
    _save(df, '06_walkforward')
    return df


# ---------------------------------------------------------------------------
# 7  probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------

def _block_moments(mat: np.ndarray, blocks: int):
    t = mat.shape[0]
    edges = np.linspace(0, t, blocks + 1).astype(int)
    n = np.array([edges[b + 1] - edges[b] for b in range(blocks)], dtype=float)
    s1 = np.array([mat[edges[b]:edges[b + 1]].sum(axis=0) for b in range(blocks)])
    s2 = np.array([(mat[edges[b]:edges[b + 1]] ** 2).sum(axis=0)
                   for b in range(blocks)])
    return n, s1, s2


def _sharpe_from_moments(n, s1, s2, sel, bpy):
    nn = n[sel].sum()
    m1 = s1[sel].sum(axis=0) / nn
    m2 = s2[sel].sum(axis=0) / nn
    var = np.maximum(m2 - m1 ** 2, 1e-24) * nn / (nn - 1)
    return m1 / np.sqrt(var) * np.sqrt(bpy)


def stage_pbo(cache, blocks: int = 12) -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        bpy = D.BARS_PER_YEAR[inst]
        _, mat, _ = cache[inst]
        n, s1, s2 = _block_moments(mat.astype(np.float64), blocks)
        all_b = set(range(blocks))
        lambdas, degradations = [], []
        for combo in itertools.combinations(range(blocks), blocks // 2):
            is_sel, oos_sel = list(combo), sorted(all_b - set(combo))
            is_s = _sharpe_from_moments(n, s1, s2, is_sel, bpy)
            oos_s = _sharpe_from_moments(n, s1, s2, oos_sel, bpy)
            best = int(np.argmax(is_s))
            rank = float((oos_s <= oos_s[best]).sum()) / (len(oos_s) + 1)
            rank = min(max(rank, 1e-6), 1 - 1e-6)
            lambdas.append(np.log(rank / (1 - rank)))
            degradations.append(oos_s[best] - is_s[best])
        lam = np.array(lambdas)
        rows.append({
            'instrument': inst, 'trials': mat.shape[1], 'splits': len(lam),
            'pbo': float((lam <= 0).mean()),
            'median_logit': float(np.median(lam)),
            'mean_oos_minus_is_sharpe': float(np.mean(degradations)),
        })
    df = pd.DataFrame(rows)
    _save(df, '07_pbo')
    return df


# ---------------------------------------------------------------------------
# 8  deflated Sharpe ratio
# ---------------------------------------------------------------------------

def _dsr(best_ret: np.ndarray, all_sharpes_ann: np.ndarray, bpy: int) -> dict:
    r = best_ret[np.isfinite(best_ret)]
    t = r.size
    sr = r.mean() / r.std(ddof=1)
    sk = stats.skew(r)
    ku = stats.kurtosis(r, fisher=False)

    sr_trials = all_sharpes_ann / np.sqrt(bpy)
    var_sr = sr_trials.var(ddof=1)
    trials = sr_trials.size
    gamma = 0.5772156649
    z1 = stats.norm.ppf(1 - 1.0 / trials)
    z2 = stats.norm.ppf(1 - 1.0 / (trials * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * z1 + gamma * z2)

    denom = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return {
        'observed_sharpe_ann': float(sr * np.sqrt(bpy)),
        'expected_max_sharpe_under_null_ann': float(sr0 * np.sqrt(bpy)),
        'trials': int(trials), 'skew': float(sk), 'kurtosis': float(ku),
        'deflated_sharpe_prob': float(
            stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom)),
    }


def stage_dsr(cache) -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        bpy = D.BARS_PER_YEAR[inst]
        _, mat, params = cache[inst]
        sharpes = np.array([_sharpe(mat[:, j], bpy) for j in range(mat.shape[1])])
        best = int(np.argmax(sharpes))
        d = _dsr(mat[:, best].astype(np.float64), sharpes, bpy)
        d.update({'instrument': inst, 'best_config': params[best].key()})
        rows.append(d)
    df = pd.DataFrame(rows)
    _save(df, '08_dsr')
    return df


# ---------------------------------------------------------------------------
# 10  repairs: does any nearby version of the idea work better?
# ---------------------------------------------------------------------------

VARIANTS = {
    'A  as specified (SMA breaks 2s band)': Params(),
    'B  mean-reversion leg only':           Params(use_tf=False),
    'C  trend leg only (SMA trigger)':      Params(use_mr=False),
    'D  trend trigger = price, same band':  Params(trend_trigger='price'),
    'E  trend trigger = SMA, band 1.0s':    Params(k_trend=1.0),
    'F  MR only when channel is flat':      Params(use_tf=False, slope_max=2.0),
    'G  flat-gated MR + steep-gated trend': Params(trend_trigger='price',
                                                  slope_max=2.0,
                                                  slope_min_trend=4.0),
    'H  trend only, price + steep channel': Params(use_mr=False,
                                                   trend_trigger='price',
                                                   slope_min_trend=4.0),
    'I  MR with a 2-sigma stop':            Params(use_tf=False, stop_sigma=2.0),
    'J  MR with a 20-bar time stop':        Params(use_tf=False, max_hold=20),
}


def stage_variants() -> pd.DataFrame:
    rows = []
    for label, p in VARIANTS.items():
        row = {'variant': label}
        sharpes = []
        for inst in INSTRUMENTS:
            m = run(D.load(inst)['close'], p, D.BARS_PER_YEAR[inst],
                    cost_bps=COST_BPS)
            row[inst] = m['sharpe']
            row[f'{inst}_exp'] = m['exposure']
            sharpes.append(m['sharpe'])
        row['mean_sharpe'] = float(np.mean(sharpes))
        row['n_positive'] = int(sum(s > 0 for s in sharpes))
        rows.append(row)
    df = pd.DataFrame(rows)
    _save(df, '10_variants')
    return df


# ---------------------------------------------------------------------------
# 9  null distributions
# ---------------------------------------------------------------------------

def _rebuild(close: pd.Series, rets: np.ndarray) -> pd.Series:
    return pd.Series(close.iloc[0] * np.cumprod(1.0 + rets), index=close.index)


def stage_nulls(reps: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for inst in INSTRUMENTS:
        close = D.load(inst)['close']
        bpy = D.BARS_PER_YEAR[inst]
        real = run(close, DEFAULT, bpy, cost_bps=COST_BPS)['sharpe']
        rets = close.pct_change().fillna(0.0).to_numpy()
        mu, sd = rets.mean(), rets.std()

        perm, gbm = [], []
        for _ in range(reps):
            perm.append(_sharpe(simulate(
                _rebuild(close, rng.permutation(rets)), DEFAULT,
                cost_bps=COST_BPS)['ret_net'].to_numpy(), bpy))
            gbm.append(_sharpe(simulate(
                _rebuild(close, rng.normal(mu, sd, rets.size)), DEFAULT,
                cost_bps=COST_BPS)['ret_net'].to_numpy(), bpy))
        perm, gbm = np.array(perm), np.array(gbm)
        rows.append({
            'instrument': inst, 'observed_sharpe': real, 'reps': reps,
            'perm_mean': float(perm.mean()), 'perm_std': float(perm.std(ddof=1)),
            'perm_p_value': float((perm >= real).mean()),
            'perm_pct_positive': float((perm > 0).mean()),
            'gbm_mean': float(gbm.mean()), 'gbm_std': float(gbm.std(ddof=1)),
            'gbm_p_value': float((gbm >= real).mean()),
        })
        print(f'  nulls done: {inst}')
    df = pd.DataFrame(rows)
    _save(df, '09_nulls')
    return df


# ---------------------------------------------------------------------------
# 11  is the P&L just long beta?
# ---------------------------------------------------------------------------

def _newey_west_t(y: np.ndarray, x: np.ndarray, lags: int = 5):
    """OLS of y on [1, x] with a Newey-West t-stat on the intercept."""
    n = y.size
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    s = (X * resid[:, None])
    omega = s.T @ s
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1)
        gamma = s[L:].T @ s[:-L]
        omega += w * (gamma + gamma.T)
    cov = xtx_inv @ omega @ xtx_inv
    return float(beta[0]), float(beta[1]), float(beta[0] / np.sqrt(cov[0, 0]))


def stage_beta() -> pd.DataFrame:
    """Split the return stream by direction and strip out market exposure."""
    rows = []
    for inst in INSTRUMENTS:
        close = D.load(inst)['close']
        bpy = D.BARS_PER_YEAR[inst]
        for label, p in (('switch (as specified)', DEFAULT),
                         ('mean-reversion leg only', Params(use_tf=False))):
            sig = simulate(close, p, cost_bps=COST_BPS)
            r = sig['ret_net'].to_numpy()
            m = sig['bh'].to_numpy()
            a, b, t_a = _newey_west_t(r, m)
            longs = sig['ret_net'].where(sig['held'] > 0, 0.0).to_numpy()
            shorts = sig['ret_net'].where(sig['held'] < 0, 0.0).to_numpy()
            rows.append({
                'instrument': inst, 'variant': label,
                'sharpe': _sharpe(r, bpy),
                'beta': b,
                'alpha_ann': a * bpy,
                'alpha_t_stat_nw': t_a,
                'long_share': float((sig['held'] > 0).mean()),
                'short_share': float((sig['held'] < 0).mean()),
                'long_pnl': float(longs.sum()),
                'short_pnl': float(shorts.sum()),
                'long_sharpe': _sharpe(longs, bpy),
                'short_sharpe': _sharpe(shorts, bpy),
            })
    df = pd.DataFrame(rows)
    _save(df, '11_beta')
    return df


# ---------------------------------------------------------------------------
# 12  sign-flip null (keeps volatility clustering, kills direction)
# ---------------------------------------------------------------------------

def stage_signflip(reps: int = 500) -> pd.DataFrame:
    """A fairer null than i.i.d. permutation.

    Permuting returns destroys volatility clustering as well as any
    predictability, which narrows the null and flatters the strategy.
    Randomly flipping the sign of each return keeps the |r| sequence --
    and therefore the vol clusters the channel actually reacts to --
    while removing every directional pattern.
    """
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for inst in INSTRUMENTS:
        close = D.load(inst)['close']
        bpy = D.BARS_PER_YEAR[inst]
        real = run(close, DEFAULT, bpy, cost_bps=COST_BPS)['sharpe']
        rets = close.pct_change().fillna(0.0).to_numpy()

        null = []
        for _ in range(reps):
            flips = rng.choice([-1.0, 1.0], size=rets.size)
            null.append(_sharpe(simulate(
                _rebuild(close, rets * flips), DEFAULT,
                cost_bps=COST_BPS)['ret_net'].to_numpy(), bpy))
        null = np.array(null)
        rows.append({
            'instrument': inst, 'observed_sharpe': real, 'reps': reps,
            'null_mean': float(null.mean()),
            'null_std': float(null.std(ddof=1)),
            'p_value': float((null >= real).mean()),
            'null_p95': float(np.percentile(null, 95)),
        })
        print(f'  sign-flip done: {inst}')
    df = pd.DataFrame(rows)
    _save(df, '12_signflip')
    return df



# ---------------------------------------------------------------------------
# 13  the specified indicators: DevLucem Lin Reg ++ and the 0.3s inner band
# ---------------------------------------------------------------------------

# Linear Regression ++ [Dev Lucem], Pine v5, verified against the published
# source: source=close, length=100, deviation=2.0, offset=0, smoothing=1,
# and the band half-width is sqrt(sum(resid^2)/length) -- population, not
# n-2. Its own alerts fire on a band CROSS, so the fade entry is at |z| = k
# exactly (entry_frac = 1.0), not at 0.9k.
DEVLUCEM = dict(n=100, k=2.0, entry_frac=1.0, dev_ddof=0)

# The 20-SMA inner band from the RegDet BB indicator: mean = SMA(close, 20),
# inner bands at 0.3 population stdev of close over the same 20 bars.
INNER_SD = 0.3

SPECS = {
    'v1  legacy: SMA vs regression band, n-2 sigma':
        Params(trend_trigger='sma', entry_frac=0.9, exit_z=0.0, dev_ddof=2),
    'S0  DevLucem fade, exit at the regression centre':
        Params(**DEVLUCEM, use_tf=False, exit_z=0.0),
    'S1  DevLucem fade, exit inside the 0.3s inner band':
        Params(**DEVLUCEM, use_tf=False, mr_exit='inner', inner_sd=INNER_SD),
    'S2  inner band as the trend trigger, fade otherwise':
        Params(**DEVLUCEM, trend_trigger='inner', inner_sd=INNER_SD),
    'S3  S1 plus a 2.0s cap on new fades':
        Params(**DEVLUCEM, use_tf=False, mr_exit='inner', inner_sd=INNER_SD,
               cap_sd=2.0),
}


def stage_specs() -> pd.DataFrame:
    rows = []
    for label, p in SPECS.items():
        row = {'spec': label}
        sh = []
        for inst in INSTRUMENTS:
            m = run(D.load(inst)['close'], p, D.BARS_PER_YEAR[inst],
                    cost_bps=COST_BPS)
            row[inst] = m['sharpe']
            if inst == 'SP500':
                row['trades'] = m['n_trades']
                row['exposure'] = m['exposure']
            sh.append(m['sharpe'])
        row['daily_mean'] = float(np.mean(sh[:4]))   # EURUSD is hourly
        row['all_mean'] = float(np.mean(sh))
        rows.append(row)
    df = pd.DataFrame(rows)
    _save(df, '13_specs')
    return df


def stage_cap() -> pd.DataFrame:
    """How far from the 20-SMA should a fade stop being allowed?"""
    rows = []
    for exit_style in ('center', 'inner'):
        for cap in (0.0, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
            p = Params(**DEVLUCEM, use_tf=False, mr_exit=exit_style,
                       inner_sd=INNER_SD, cap_sd=cap, cap_blocks_mr=True)
            row = {'mr_exit': exit_style, 'cap_sd': cap if cap else np.nan}
            sh, tr = [], []
            for inst in INSTRUMENTS:
                m = run(D.load(inst)['close'], p, D.BARS_PER_YEAR[inst],
                        cost_bps=COST_BPS)
                row[inst] = m['sharpe']
                sh.append(m['sharpe'])
                tr.append(m['n_trades'])
            row['daily_mean'] = float(np.mean(sh[:4]))
            row['n_positive'] = int(sum(x > 0 for x in sh[:4]))
            row['trades'] = int(np.sum(tr))
            rows.append(row)
    df = pd.DataFrame(rows)
    _save(df, '14_cap_sweep')
    return df


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    ap.add_argument('--reps', type=int, default=500)
    args = ap.parse_args()

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 50)
    want, cache = args.stage, None

    if want in ('all', 'baseline'):
        print('\n=== 1. Baseline (default params, 5 bps per side) ===')
        print(stage_baseline().round(3).to_string(index=False))
        print('\n=== 2. Leg attribution ===')
        print(stage_leg_attribution().round(3).to_string(index=False))

    if want in ('all', 'grid', 'walkforward', 'pbo', 'dsr'):
        print('\n=== 3. Parameter grid ===')
        _, summary, cache = stage_grid()
        print(summary.round(3).to_string(index=False))

    if want in ('all', 'costs'):
        print('\n=== 4. Cost sensitivity ===')
        print(stage_costs().round(3).to_string(index=False))

    if want in ('all', 'stability'):
        print('\n=== 5. Year by year ===')
        print(stage_stability().round(3).to_string(index=False))

    if want in ('all', 'walkforward'):
        print('\n=== 6. Anchored walk-forward ===')
        print(stage_walkforward(cache).round(3).to_string(index=False))

    if want in ('all', 'pbo'):
        print('\n=== 7. Probability of backtest overfitting (CSCV) ===')
        print(stage_pbo(cache).round(3).to_string(index=False))

    if want in ('all', 'dsr'):
        print('\n=== 8. Deflated Sharpe ===')
        print(stage_dsr(cache).round(3).to_string(index=False))

    if want in ('all', 'variants'):
        print('\n=== 10. Variants / repairs (net Sharpe per instrument) ===')
        v = stage_variants()
        print(v[['variant'] + INSTRUMENTS + ['mean_sharpe', 'n_positive']]
              .round(3).to_string(index=False))

    if want in ('all', 'nulls'):
        print('\n=== 9. Null distributions ===')
        print(stage_nulls(reps=args.reps).round(3).to_string(index=False))

    if want in ('all', 'beta'):
        print('\n=== 11. Market exposure and direction split ===')
        print(stage_beta().round(3).to_string(index=False))

    if want in ('all', 'specs'):
        print('\n=== 13. Specified indicators (DevLucem + 0.3s inner band) ===')
        print(stage_specs().round(3).to_string(index=False))
        print('\n=== 14. Cap sweep: how far from the SMA to stop fading ===')
        print(stage_cap().round(3).to_string(index=False))

    if want in ('all', 'signflip'):
        print('\n=== 12. Sign-flip null (vol clustering preserved) ===')
        print(stage_signflip(reps=args.reps).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
