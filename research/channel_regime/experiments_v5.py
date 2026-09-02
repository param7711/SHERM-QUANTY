"""
v5 -- the band-to-band fade, tested on the timeframe it was drawn on.

SPEC (from the settings panels, not from prose)
    RegDet+VC Bollinger : length 20, SMA basis, source close,
                          OUTER 2.1 sd, INNER 0.3 sd, offset 0
    LRC_SH              : length 400
    short at the outer upper band, cover at the inner upper band
    long  at the outer lower band, sell  at the inner lower band
    trade only while the BLUE line (the 20-SMA) is inside LRC_SH
    no stop loss

PRIMARY UNIVERSE is NSE hourly, because the chart is a 1-hour chart and
a 20-bar band means something different on every timeframe.  Daily US and
daily India are carried alongside as a transfer test, not as the headline.

THE UNKNOWN.  LRC_SH exposes only Length; its deviation multiplier is
hard-coded in a script I have not read.  The channel width IS the gate,
so the width is swept everywhere and nothing is fitted to it.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as DUS
import data_in as DIN
import data_1h as D1H
from bbfade import (FadeParams, simulate, summarise, sharpe, trade_table,
                    bands, signals)
from experiments import _newey_west_t

RESULTS = os.path.join(DUS.ROOT, 'results', 'channel_regime')
os.makedirs(RESULTS, exist_ok=True)

PPY_1H = 7 * 250          # seven hourly bars a session
PPY_D = 252

IDX_1H = ['NIFTY', 'BANKNIFTY']
STK_1H = [s for s in D1H.STOCKS]
US_D = ['SP500', 'NASDAQ', 'GOOG', 'EURUSD']       # WTI is close-only: no touches
IN_D = ['NIFTY50', 'BANKNIFTY', 'NIFTY500', 'RELIANCE', 'HDFCBANK',
        'ICICIBANK', 'INFY', 'ITC', 'LT', 'AXISBANK', 'KOTAKBANK',
        'BHARTIARTL', 'MARUTI', 'HINDUNILVR', 'ASIANPAINT', 'BAJFINANCE']

# (key, group, loader, bars-per-year)
UNIVERSE = ([(s, 'NSE 1h index', 'h', PPY_1H) for s in IDX_1H]
            + [(s, 'NSE 1h stock', 'h', PPY_1H) for s in STK_1H]
            + [(s, 'US daily', 'u', PPY_D) for s in US_D]
            + [(s, 'NSE daily', 'i', PPY_D) for s in IN_D])
PRIMARY = [u for u in UNIVERSE if u[2] == 'h']

SPEC = dict(bb_len=20, bb_ma='sma', outer=2.1, inner=0.3, lrc_len=400)
DEVS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
BASE_DEV = 2.0

_C = {}


def load(name, kind):
    k = (name, kind)
    if k not in _C:
        d = (D1H.load(name, 'back') if kind == 'h'
             else DUS.load(name) if kind == 'u' else DIN.load(name))
        _C[k] = d[['open', 'high', 'low', 'close']].dropna()
    return _C[k]


def save(df, name):
    df.to_csv(os.path.join(RESULTS, name), index=False)
    print(f'  -> {name}  ({len(df)} rows)')
    return df


def sh(r, ppy):
    return sharpe(r, ppy)


# ---------------------------------------------------------- 30 headline

def stage_headline():
    rows = []
    for name, grp, kind, ppy in UNIVERSE:
        df = load(name, kind)
        for dev in DEVS:
            s = summarise(df, FadeParams(**SPEC, lrc_dev=dev), ppy=ppy)
            s |= {'inst': name, 'group': grp, 'lrc_dev': dev,
                  'start': str(df.index[0])[:10], 'end': str(df.index[-1])[:10]}
            rows.append(s)
    out = pd.DataFrame(rows)
    cols = ['inst', 'group', 'lrc_dev', 'sharpe', 'sharpe_gross', 'cagr',
            'exposure', 'inside_pct', 'trades', 'win_rate', 'avg_bars',
            'max_bars', 'long_trades', 'short_trades', 'long_pnl', 'short_pnl',
            'both_band_bars', 'bars', 'start', 'end']
    save(out[[c for c in cols if c in out.columns]], '30_v5_headline.csv')
    piv = out.pivot_table(index=['group'], columns='lrc_dev',
                          values='sharpe', aggfunc='mean').round(3)
    print(piv.to_string())
    return out


# ------------------------------------------------------------- 31 gate

def stage_gate():
    rows = []
    for name, grp, kind, ppy in UNIVERSE:
        df = load(name, kind)
        b = summarise(df, FadeParams(**SPEC, use_filter=False), ppy=ppy)
        r = {'inst': name, 'group': grp, 'sharpe_nogate': b['sharpe'],
             'trades_nogate': b['trades'], 'exposure_nogate': b['exposure']}
        for dev in DEVS:
            g = summarise(df, FadeParams(**SPEC, lrc_dev=dev), ppy=ppy)
            r[f'S_dev{dev}'] = g['sharpe']
            r[f'inside_dev{dev}'] = g['inside_pct']
            r[f'delta_dev{dev}'] = g['sharpe'] - b['sharpe']
        rows.append(r)
    out = pd.DataFrame(rows)
    save(out, '31_v5_gate.csv')
    for grp, g in out.groupby('group'):
        d = ' '.join(f'{v:+.3f}' for v in
                     [g[f'delta_dev{x}'].mean() for x in DEVS])
        i = ' '.join(f'{g[f"inside_dev{x}"].mean():.0%}' for x in DEVS)
        print(f'  {grp:<14} nogate S={g["sharpe_nogate"].mean():+.3f} '
              f'| gate delta {d} | inside {i}')
    return out


# ------------------------------------------------------------ 32 costs

def stage_costs():
    rows = []
    for name, grp, kind, ppy in UNIVERSE:
        df = load(name, kind)
        for bps in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]:
            for gate in (True, False):
                s = summarise(df, FadeParams(**SPEC, lrc_dev=BASE_DEV,
                                             use_filter=gate, cost_bps=bps),
                              ppy=ppy)
                rows.append({'inst': name, 'group': grp, 'cost_bps': bps,
                             'gate': gate, 'sharpe': s['sharpe'],
                             'trades': s['trades'], 'cagr': s['cagr']})
    out = pd.DataFrame(rows)
    save(out, '32_v5_costs.csv')
    print(out[out.gate].pivot_table(index='group', columns='cost_bps',
                                    values='sharpe', aggfunc='mean')
          .round(3).to_string())
    return out


# ------------------------------------------------------ 33 sensitivity

def stage_sensitivity():
    rows = []
    for name, grp, kind, ppy in PRIMARY:
        df = load(name, kind)
        for outer in [1.5, 1.8, 2.0, 2.1, 2.3, 2.5, 3.0]:
            for inner in [0.0, 0.3, 0.6, 1.0]:
                for blen in [10, 14, 20, 30, 50]:
                    s = summarise(df, FadeParams(bb_len=blen, outer=outer,
                                                 inner=inner, lrc_len=400,
                                                 lrc_dev=BASE_DEV), ppy=ppy)
                    rows.append({'inst': name, 'group': grp, 'outer': outer,
                                 'inner': inner, 'bb_len': blen,
                                 'sharpe': s['sharpe'], 'trades': s['trades'],
                                 'exposure': s['exposure']})
    out = pd.DataFrame(rows)
    save(out, '33_v5_sensitivity.csv')
    piv = out.pivot_table(index='outer', columns='inner', values='sharpe',
                          aggfunc='mean').round(3)
    print('  mean Sharpe by outer (rows) x inner (cols), all bb_len:')
    print(piv.to_string())
    save(out.groupby(['outer', 'inner', 'bb_len'])['sharpe'].mean()
            .reset_index(), '33_v5_sensitivity_summary.csv')
    return out


# --------------------------------------------------------- 34 mechanics

def stage_mechanics():
    variants = {
        'spec': dict(),
        'close trigger': dict(trigger='close'),
        'same-bar exit allowed': dict(allow_same_bar=True),
        'gate on exits too': dict(filter_exits=True),
        'long only': dict(long_only=True),
        'short only': dict(short_only=True),
        '2-bar fill delay': dict(lag=2),
        'no gate': dict(use_filter=False),
    }
    rows = []
    for name, grp, kind, ppy in PRIMARY:
        df = load(name, kind)
        for v, kw in variants.items():
            s = summarise(df, FadeParams(**SPEC, lrc_dev=BASE_DEV, **kw), ppy=ppy)
            rows.append({'inst': name, 'group': grp, 'variant': v,
                         'sharpe': s['sharpe'], 'trades': s['trades'],
                         'exposure': s['exposure'], 'win_rate': s['win_rate'],
                         'avg_bars': s['avg_bars'], 'max_bars': s['max_bars']})
    out = pd.DataFrame(rows)
    save(out, '34_v5_mechanics.csv')
    print(out.groupby('variant')[['sharpe', 'trades', 'exposure']]
          .mean().round(3).to_string())
    return out


# ------------------------------------------------------------- 35 alpha

def stage_alpha():
    rows = []
    for name, grp, kind, ppy in UNIVERSE:
        df = load(name, kind)
        s = simulate(df, FadeParams(**SPEC, lrc_dev=BASE_DEV)).dropna(
            subset=['net', 'ret'])
        if len(s) < 500:
            continue
        a, b, t = _newey_west_t(s['net'].to_numpy(), s['ret'].to_numpy(), lags=20)
        rows.append({'inst': name, 'group': grp, 'alpha_ann': a * ppy,
                     'beta': b, 'alpha_t_nw': t, 'sharpe': sh(s['net'], ppy),
                     'bh_sharpe': sh(s['ret'], ppy)})
    out = pd.DataFrame(rows)
    save(out, '35_v5_alpha.csv')
    print(out.groupby('group')[['alpha_ann', 'beta', 'alpha_t_nw', 'sharpe']]
          .mean().round(3).to_string())
    return out


# ------------------------------------------------------------- 36 nulls

def _gate_sim(df, p, gate_arr, ppy):
    b = bands(df, p)
    hi, lo = df['high'].astype(float), df['low'].astype(float)
    ok = (b[['upper', 'lower', 'inner_up', 'inner_dn']].notna().all(axis=1)
          & (b['sd'] > 0)).to_numpy()
    hu = (hi >= b['upper']).to_numpy() & ok
    hd = (lo <= b['lower']).to_numpy() & ok
    xs = (lo <= b['inner_up']).to_numpy() & ok
    xl = (hi >= b['inner_dn']).to_numpy() & ok
    pos = np.zeros(len(df)); cur = 0.0
    for i in range(len(df)):
        if not ok[i]:
            pos[i] = cur; continue
        moved = False
        if cur > 0 and xl[i]:
            cur, moved = 0.0, True
        elif cur < 0 and xs[i]:
            cur, moved = 0.0, True
        if cur == 0 and gate_arr[i] and not moved:
            if hu[i] and hd[i]:
                pass
            elif hd[i]:
                cur = 1.0
            elif hu[i]:
                cur = -1.0
        pos[i] = cur
    held = pd.Series(pos, index=df.index).shift(1).fillna(0.0)
    ret = df['close'].astype(float).pct_change()
    cost = held.diff().abs().fillna(held.abs()) * p.cost_bps / 1e4
    return sh((held * ret - cost).dropna(), ppy)


def stage_nulls(n_flip=2000, n_rot=300, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for name, grp, kind, ppy in PRIMARY:
        df = load(name, kind)
        p = FadeParams(**SPEC, lrc_dev=BASE_DEV)
        s = simulate(df, p).dropna(subset=['net'])
        obs = sh(s['net'], ppy)

        pos, ret, cst = (s['pos'].to_numpy(), s['ret'].to_numpy(),
                         s['cost'].to_numpy())
        blk = np.r_[0, np.cumsum(np.abs(np.diff(pos)) > 0)]
        nb = int(blk.max()) + 1
        na = np.empty(n_flip)
        for i in range(n_flip):
            r = pos * rng.choice([-1.0, 1.0], nb)[blk] * ret - cst
            na[i] = r.mean() / r.std() * np.sqrt(ppy)

        ins = bands(df, p)['inside'].fillna(False).to_numpy()
        nbv = np.empty(n_rot)
        lo_, hi_ = int(0.05 * len(df)), int(0.95 * len(df))
        for i in range(n_rot):
            nbv[i] = _gate_sim(df, p, np.roll(ins, rng.integers(lo_, hi_)), ppy)

        rows.append({'inst': name, 'group': grp, 'sharpe': obs,
                     'signflip_p': float((na >= obs).mean()),
                     'signflip_p95': float(np.percentile(na, 95)),
                     'rotgate_p': float((nbv >= obs).mean()),
                     'rotgate_mean': float(np.nanmean(nbv))})
        print(f'   {name:<12} S={obs:+.3f}  flip p={rows[-1]["signflip_p"]:.3f}'
              f'  rot-gate p={rows[-1]["rotgate_p"]:.3f} '
              f'(rot mean {rows[-1]["rotgate_mean"]:+.3f})')
    return save(pd.DataFrame(rows), '36_v5_nulls.csv')


# --------------------------------------------------------- 37 stability

def stage_stability():
    rows, yr = [], []
    for name, grp, kind, ppy in PRIMARY:
        df = load(name, kind)
        s = simulate(df, FadeParams(**SPEC, lrc_dev=BASE_DEV)).dropna(subset=['net'])
        n = len(s)
        for i, (a, b_) in enumerate([(0, n // 3), (n // 3, 2 * n // 3),
                                     (2 * n // 3, n)]):
            seg = s.iloc[a:b_]
            rows.append({'inst': name, 'group': grp, 'window': f'third {i+1}',
                         'from': str(seg.index[0])[:10], 'to': str(seg.index[-1])[:10],
                         'sharpe': sh(seg['net'], ppy)})
        cut = s.index[0] + pd.DateOffset(years=3)
        oos = s[s.index > cut]
        rows.append({'inst': name, 'group': grp, 'window': 'OOS (yr 4+)',
                     'from': str(oos.index[0])[:10] if len(oos) else '',
                     'to': str(oos.index[-1])[:10] if len(oos) else '',
                     'sharpe': sh(oos['net'], ppy) if len(oos) > 500 else np.nan})
        for y, g in s.groupby(s.index.year):
            if len(g) > 400:
                yr.append({'inst': name, 'group': grp, 'year': int(y),
                           'sharpe': sh(g['net'], ppy), 'ret': float(g['net'].sum())})
    out = pd.DataFrame(rows)
    save(out, '37_v5_stability.csv')
    print(out.pivot_table(index='group', columns='window', values='sharpe',
                          aggfunc='mean').round(3).to_string())
    y = pd.DataFrame(yr)
    save(y, '37_v5_by_year.csv')
    print(y.groupby('year')[['sharpe', 'ret']].mean().round(3).to_string())
    return out


# ------------------------------------------------- 38 random-walk control

def stage_randomwalk(n_paths=200, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for gate in (True, False):
        v = []
        for _ in range(n_paths):
            n = 11000
            px = 100 * np.exp(np.cumsum(rng.normal(0, 0.0035, n)))
            d = pd.DataFrame({'open': px,
                              'high': px * (1 + np.abs(rng.normal(0, .0018, n))),
                              'low': px * (1 - np.abs(rng.normal(0, .0018, n))),
                              'close': px},
                             index=pd.date_range('2013-01-01', periods=n, freq='h'))
            v.append(summarise(d, FadeParams(**SPEC, lrc_dev=BASE_DEV,
                                             use_filter=gate), ppy=PPY_1H)['sharpe'])
        v = np.array(v)
        out.append({'gate': gate, 'paths': n_paths, 'mean': v.mean(),
                    'sd': v.std(), 'p05': np.percentile(v, 5),
                    'p95': np.percentile(v, 95),
                    'frac_positive': float((v > 0).mean())})
        print(f'   random walk gate={gate}: mean {v.mean():+.3f}  '
              f'p95 {np.percentile(v,95):+.3f}  frac>0 {(v>0).mean():.0%}')
    return save(pd.DataFrame(out), '38_v5_randomwalk.csv')


STAGES = {'headline': stage_headline, 'gate': stage_gate, 'costs': stage_costs,
          'sens': stage_sensitivity, 'mech': stage_mechanics,
          'alpha': stage_alpha, 'nulls': stage_nulls, 'stab': stage_stability,
          'rw': stage_randomwalk}

if __name__ == '__main__':
    for k in (sys.argv[1:] or list(STAGES)):
        print(f'== {k}')
        STAGES[k]()
