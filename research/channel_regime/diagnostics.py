"""
Where does the money actually go?

A negative Sharpe says a strategy loses.  It does not say why, and the
"why" is the only part that tells you what to build next.  There are only
four places the money can go, and each has its own test:

    1. the ENTRY has no edge          -> event study, and the rotated-entry
                                         control below
    2. the PAYOFF SHAPE is negative   -> a fixed target against an
                                         unbounded loss loses even on a
                                         coin-flip entry; the synthetic
                                         market measures exactly that
    3. COSTS eat a real but small edge-> the cost ladder
    4. it was BAD LUCK               -> the trade bootstrap

The controls are built so that only one thing changes at a time.

ROTATED-ENTRY CONTROL.  Keep the number of trades, the long/short mix and
the exit rule exactly as they are, and move only WHEN the entries happen,
by a random circular shift.  Both arms enter at the bar's open so the two
are priced the same way.  If the real entries score like the rotated ones,
the bands are not choosing anything -- the loss is geometry.  If the real
entries score WORSE, the bands are actively picking the wrong moments.

SYNTHETIC MARKET.  A stationary block bootstrap of the instrument's own
hourly returns: same volatility, same fat tails, same intraday range
distribution, but any serial structure beyond the block length is
destroyed.  Running the full strategy there gives the Sharpe this
machinery produces in a market with nothing to find.  It is the number the
real result has to be compared against -- not zero.
"""

import numpy as np
import pandas as pd

from bbfade import FadeParams, simulate_limit, bands, sharpe


# ------------------------------------------------------------- ratios

def _drawdown(eq: np.ndarray) -> tuple:
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    i = int(np.argmin(dd))
    under = dd < 0
    longest, run = 0, 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return float(dd.min()), i, longest


def ratios(net: pd.Series, trades: pd.DataFrame, ppy: int) -> dict:
    r = net.fillna(0.0).to_numpy()
    eq = np.cumprod(1.0 + r)
    yrs = len(r) / ppy
    cagr = eq[-1] ** (1 / yrs) - 1.0
    mdd, _, longest = _drawdown(eq)
    dn = r[r < 0]
    up = r[r > 0]
    sd = r.std()
    dsd = dn.std() if dn.size else np.nan
    # Ulcer index: RMS of the drawdown path, which punishes deep AND long
    peak = np.maximum.accumulate(eq)
    ulcer = float(np.sqrt(np.mean((eq / peak - 1.0) ** 2)) * 100)
    w = trades['pnl'] > 0 if len(trades) else pd.Series(dtype=bool)
    gw = float(trades.loc[w, 'pnl'].sum()) if len(trades) else np.nan
    gl = float(-trades.loc[~w, 'pnl'].sum()) if len(trades) else np.nan
    aw = float(trades.loc[w, 'pnl'].mean()) if w.any() else np.nan
    al = float(trades.loc[~w, 'pnl'].mean()) if (~w).any() else np.nan
    wr = float(w.mean()) if len(trades) else np.nan
    exp_ = wr * aw + (1 - wr) * al if len(trades) else np.nan
    return {
        'sharpe': float(r.mean() / sd * np.sqrt(ppy)) if sd else np.nan,
        'sortino': float(r.mean() / dsd * np.sqrt(ppy)) if dsd else np.nan,
        'calmar': float(cagr / abs(mdd)) if mdd else np.nan,
        'cagr': float(cagr),
        'vol_ann': float(sd * np.sqrt(ppy)),
        'max_dd': mdd,
        'dd_bars': longest,
        'ulcer': ulcer,
        'skew': float(pd.Series(r).skew()),
        'kurtosis': float(pd.Series(r).kurtosis()),
        'tail_ratio': float(np.percentile(r, 95) / abs(np.percentile(r, 5)))
        if np.percentile(r, 5) else np.nan,
        'profit_factor': float(gw / gl) if gl else np.nan,
        'win_rate': wr,
        'avg_win': aw,
        'avg_loss': al,
        'payoff_ratio': float(aw / abs(al)) if al else np.nan,
        'expectancy': float(exp_) if exp_ == exp_ else np.nan,
        # Kelly on the trade distribution: negative means never size it up
        'kelly': float(wr - (1 - wr) / (aw / abs(al))) if (al and aw) else np.nan,
        'trades': int(len(trades)),
        'pct_time_in': float((net.index.size and (trades['bars'].sum() / len(net)))
                             if len(trades) else 0.0),
    }


# ------------------------------------------------ MC 1: trade bootstrap

def trade_bootstrap(trades: pd.DataFrame, n: int = 20000, seed: int = 1) -> dict:
    """Was it bad luck?  Resample the realised trades with replacement."""
    if len(trades) < 30:
        return {}
    rng = np.random.default_rng(seed)
    p = trades['pnl'].to_numpy()
    draws = rng.choice(p, size=(n, p.size), replace=True)
    tot = draws.sum(axis=1)
    mu = draws.mean(axis=1)
    sd = draws.std(axis=1)
    t = mu / (sd / np.sqrt(p.size))
    return {'boot_mean_total': float(tot.mean()),
            'boot_p_total_positive': float((tot > 0).mean()),
            'boot_ci_lo': float(np.percentile(tot, 2.5)),
            'boot_ci_hi': float(np.percentile(tot, 97.5)),
            'trade_t_stat': float(p.mean() / (p.std(ddof=1) / np.sqrt(p.size)))}


# ------------------------------------------- MC 2: rotated-entry control

def _run_entries(df, p, entry_i, sides, ppy):
    """Same exit machinery, but the entry bars are handed in.

    Entry is at the bar's OPEN in both arms so the real and the rotated
    control are priced identically and only the TIMING differs.
    """
    b = bands(df, p).shift(1)
    o = df['open'].to_numpy(float); h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float);  c = df['close'].to_numpy(float)
    IU, ID = b['inner_up'].to_numpy(), b['inner_dn'].to_numpy()
    n = len(df); fee = p.cost_bps / 1e4
    want = np.zeros(n); want[entry_i] = sides
    ret = np.zeros(n); cur = 0.0; epx = np.nan; pnl = []
    for i in range(1, n):
        if cur == 0.0:
            if want[i] != 0.0 and np.isfinite(IU[i]) and np.isfinite(ID[i]):
                cur, epx = want[i], o[i]
                ret[i] = cur * (c[i] - epx) / epx - fee
        else:
            prev = c[i - 1]
            tgt = ID[i] if cur > 0 else IU[i]
            hit = (h[i] >= ID[i]) if cur > 0 else (l[i] <= IU[i])
            if hit:
                ret[i] = cur * (tgt - prev) / prev - fee
                pnl.append(cur * (tgt - epx) / epx - 2 * fee)
                cur = 0.0
            else:
                ret[i] = cur * (c[i] - prev) / prev
    s = pd.Series(ret, index=df.index)
    return sharpe(s, ppy), float(np.sum(pnl)), len(pnl)


def rotated_entry_control(df, p, ppy, n_iter=200, seed=2):
    """Move only WHEN the trades happen; keep count, side mix and exits."""
    sim = simulate_limit(df, p)
    t = sim.attrs['trades']
    if len(t) < 30:
        return {}
    ei = np.array([df.index.get_loc(x) for x in t['entry']])
    sd = t['side'].to_numpy()
    real_s, real_p, real_n = _run_entries(df, p, ei, sd, ppy)
    rng = np.random.default_rng(seed)
    out = np.empty(n_iter); tot = np.empty(n_iter)
    n = len(df)
    for k in range(n_iter):
        shift = rng.integers(int(0.05 * n), int(0.95 * n))
        out[k], tot[k], _ = _run_entries(df, p, (ei + shift) % n, sd, ppy)
    return {'real_open_entry_sharpe': real_s, 'real_open_entry_pnl': real_p,
            'rot_mean_sharpe': float(np.nanmean(out)),
            'rot_sd_sharpe': float(np.nanstd(out)),
            'rot_mean_pnl': float(np.nanmean(tot)),
            'p_real_worse_than_rot': float((out > real_s).mean()),
            'edge_vs_rotated': real_s - float(np.nanmean(out))}


# --------------------------------------- MC 3: synthetic (no-edge) market

def block_bootstrap_ohlc(df, rng, block=20):
    """Resample hourly returns in blocks, carrying each bar's own range.

    The high/low are carried with their own return so the bar shapes stay
    realistic; only the ORDER of bars is destroyed, which is the point.
    """
    c = df['close'].to_numpy(float)
    r = np.diff(np.log(c))
    hi = (df['high'].to_numpy(float) / df['close'].to_numpy(float))[1:]
    lo = (df['low'].to_numpy(float) / df['close'].to_numpy(float))[1:]
    op = (df['open'].to_numpy(float) / df['close'].to_numpy(float))[1:]
    m = r.size
    nb = int(np.ceil(m / block))
    starts = rng.integers(0, m - block, nb)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m]
    px = c[0] * np.exp(np.cumsum(r[idx]))
    return pd.DataFrame({'open': px * op[idx], 'high': px * hi[idx],
                         'low': px * lo[idx], 'close': px}, index=df.index[1:])


def synthetic_market(df, p, ppy, n_iter=150, seed=4, block=20):
    """What does this machinery earn where there is nothing to find?"""
    rng = np.random.default_rng(seed)
    out = np.empty(n_iter)
    for k in range(n_iter):
        d = block_bootstrap_ohlc(df, rng, block)
        out[k] = simulate_limit(d, p)['net'].pipe(sharpe, ppy)
    real = sharpe(simulate_limit(df, p)['net'], ppy)
    return {'real_sharpe': float(real),
            'synth_mean': float(np.nanmean(out)),
            'synth_sd': float(np.nanstd(out)),
            'synth_p05': float(np.nanpercentile(out, 5)),
            'synth_p95': float(np.nanpercentile(out, 95)),
            'p_synth_beats_real': float((out > real).mean()),
            'real_minus_synth': float(real - np.nanmean(out))}
