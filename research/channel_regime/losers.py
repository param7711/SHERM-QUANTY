"""
Everything about the losing trades.

A mean Sharpe hides the shape of a loss.  Two very different strategies
produce the same negative number: one bleeds a little on almost every
trade, the other wins steadily and is destroyed by a handful of
catastrophes.  They need opposite fixes, so the first question is which
one this is.

What gets measured here, per trade:

    pnl          realised, net of costs
    bars         how long it was held
    mae / mfe    the worst and best it ever got DURING the trade, which is
                 what a stop or a trail would actually have caught
    sd_pct       the band width at entry, as a fraction of price -- the
                 volatility regime the trade was opened into
    hour         which bar of the NSE session, because the 09:15 bar
                 carries the whole overnight gap and a band system meets
                 that gap with a resting order already in the market
    gap          the overnight jump on the entry bar, where there was one

Concentration is then read off the sorted loss curve, and the "drop the
worst k" test says how few trades would have to be removed to change the
verdict.  If a handful of trades carry the loss, the entry is fine and
the risk control is missing; if the loss is spread evenly, the entry is
the problem and no risk control saves it.
"""

import numpy as np
import pandas as pd

from bbfade import FadeParams, simulate_limit


def enrich(df: pd.DataFrame, p: FadeParams, inst: str) -> pd.DataFrame:
    """Attach the path-dependent facts the trade record cannot hold."""
    t = simulate_limit(df, p).attrs['trades']
    if not len(t):
        return t
    hi = df['high'].to_numpy(float)
    lo = df['low'].to_numpy(float)
    op = df['open'].to_numpy(float)
    cl = df['close'].to_numpy(float)
    mae, mfe, gap = [], [], []
    for r in t.itertuples():
        a, b = r.entry_i, r.exit_i
        h, l = hi[a:b + 1].max(), lo[a:b + 1].min()
        if r.side > 0:
            mae.append((l - r.entry_px) / r.entry_px)
            mfe.append((h - r.entry_px) / r.entry_px)
        else:
            mae.append((r.entry_px - h) / r.entry_px)
            mfe.append((r.entry_px - l) / r.entry_px)
        gap.append((op[a] - cl[a - 1]) / cl[a - 1] if a > 0 else np.nan)
    t = t.copy()
    t['inst'] = inst
    t['mae'] = mae                      # always <= 0: the worst it got
    t['mfe'] = mfe                      # always >= 0: the best it got
    t['gap'] = gap
    t['hour'] = pd.DatetimeIndex(t['entry']).hour
    t['year'] = pd.DatetimeIndex(t['entry']).year
    t['win'] = t['pnl'] > 0
    t['reached_target'] = t['mfe'] > 0  # the exit is a target, so this is
    return t                            # true for every closed trade


def concentration(pnl: np.ndarray) -> dict:
    """How much of the damage sits in how few trades."""
    loss = -pnl[pnl < 0]
    loss = np.sort(loss)[::-1]
    tot = loss.sum()
    n = loss.size
    out = {'n_losers': int(n), 'total_loss': float(tot)}
    for q in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, int(round(q * n)))
        out[f'worst_{int(q*100)}pct_share'] = float(loss[:k].sum() / tot)
    # Gini of the loss distribution: 0 = every loser identical,
    # 1 = one trade carries everything
    x = np.sort(loss)
    out['loss_gini'] = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
    return out


def drop_worst(pnl: np.ndarray, upto: int = 60) -> pd.DataFrame:
    """Remove the k worst trades and see when the total turns positive."""
    s = np.sort(pnl)
    rows = []
    for k in range(0, min(upto, s.size) + 1):
        rows.append({'dropped': k, 'total': float(s[k:].sum()),
                     'pct_of_trades': k / s.size})
    return pd.DataFrame(rows)


def bootstrap_concentration(pnl: np.ndarray, n: int = 5000, seed: int = 3) -> dict:
    """Is the concentration itself stable, or an artefact of this sample?"""
    rng = np.random.default_rng(seed)
    sh = np.empty(n)
    for i in range(n):
        d = rng.choice(pnl, pnl.size, replace=True)
        l = -d[d < 0]
        if l.size < 5:
            sh[i] = np.nan
            continue
        l = np.sort(l)[::-1]
        sh[i] = l[:max(1, int(round(0.05 * l.size)))].sum() / l.sum()
    return {'worst5_share_mean': float(np.nanmean(sh)),
            'worst5_share_p05': float(np.nanpercentile(sh, 5)),
            'worst5_share_p95': float(np.nanpercentile(sh, 95))}
