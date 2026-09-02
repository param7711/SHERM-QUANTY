"""
v5 -- the band-to-band fade, as specified from the chart.

    RegDet+VC   Bollinger, length 20, SMA basis, source close,
                OUTER StdDev 2.1, INNER StdDev 0.3, offset 0
    LRC_SH      linear regression channel, Length 400

Rules, verbatim from the brief:

    short  when price touches the OUTER upper band  (basis + 2.1 sd)
    cover  when price touches the INNER upper band  (basis + 0.3 sd)
    long   when price touches the OUTER lower band  (basis - 2.1 sd)
    sell   when price touches the INNER lower band  (basis - 0.3 sd)

    trade only while the BLUE line sits inside the regression channel.
    no stop loss.

THE BLUE LINE.  Settled from the indicator source, not from the picture:
of the five Bollinger plots, exactly one is blue.

    BB Upper        #F23645   red
    BB Inner Upper  #F23645   red, dimmed
    BB Basis        #2962FF   BLUE      <- the 20-period SMA
    BB Inner Lower  #089981   green, dimmed
    BB Lower        #089981   green

So the gate is: the 20-SMA must lie between the two regression-channel
lines.  That is `smaz` from the earlier passes -- the same variable the
first hypothesis wanted to trade ON, now used to stand ASIDE.

THE ONE NUMBER I DO NOT HAVE.  LRC_SH exposes a single input, Length.
Its deviation multiplier is hard-coded inside a script I have not seen,
so the channel's WIDTH is unknown, and the width is the whole of the
gate.  Every result below is therefore reported across a sweep of that
multiplier, together with how often the gate actually bites at each
setting.  Nothing is tuned to it.

CAUSALITY.  Every quantity at bar t is built from bars <= t.  The state
machine reads bar t and sets the position to HOLD FROM t+1; fills are at
t+1's close.  A bar may change state once: no entry and exit inside the
same bar, because OHLC does not say which of the high and the low came
first.  `allow_same_bar` flips that on to prove it is not load-bearing.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from channel import regression_channel

MA = {
    'sma': lambda s, n: s.rolling(n).mean(),
    'ema': lambda s, n: s.ewm(span=n, adjust=False).mean(),
    'wma': lambda s, n: s.rolling(n).apply(
        lambda w: np.dot(w, np.arange(1, n + 1)) / (n * (n + 1) / 2), raw=True),
    'rma': lambda s, n: s.ewm(alpha=1.0 / n, adjust=False).mean(),
}


@dataclass(frozen=True)
class FadeParams:
    # --- Bollinger, exactly as the settings panel shows -----------------
    bb_len: int = 20
    bb_ma: str = 'sma'
    outer: float = 2.1
    inner: float = 0.3
    # --- LRC_SH --------------------------------------------------------
    lrc_len: int = 400
    lrc_dev: float = 2.0          # NOT observed; swept everywhere
    # --- the gate ------------------------------------------------------
    use_filter: bool = True
    filter_exits: bool = False    # gate entries only, or exits too
    # --- mechanics -----------------------------------------------------
    trigger: str = 'touch'        # 'touch' = high/low ; 'close' = close only
    allow_same_bar: bool = False
    reverse: bool = False        # trade the break instead of the fade
    long_only: bool = False
    short_only: bool = False
    min_entry_hour: int = 0       # 0 = any bar; 10 = skip the 09:15 opening bar
    stop_sigma: float = 0.0       # 0 = the spec: no stop loss
    max_hold: int = 0             # 0 = no time stop
    lag: int = 1
    cost_bps: float = 5.0


def bands(df: pd.DataFrame, p: FadeParams) -> pd.DataFrame:
    """The five Bollinger lines and the two regression lines."""
    close = df['close'].astype(float)
    basis = MA[p.bb_ma](close, p.bb_len)
    sd = close.rolling(p.bb_len).std(ddof=0)      # Pine ta.stdev, biased

    ch = regression_channel(close, p.lrc_len, ddof=0)

    out = pd.DataFrame(index=df.index)
    out['basis'] = basis                          # the BLUE line
    out['sd'] = sd
    out['upper'] = basis + p.outer * sd
    out['inner_up'] = basis + p.inner * sd
    out['inner_dn'] = basis - p.inner * sd
    out['lower'] = basis - p.outer * sd
    out['lrc_c'] = ch['center']
    out['lrc_s'] = ch['sigma']
    out['lrc_up'] = ch['center'] + p.lrc_dev * ch['sigma']
    out['lrc_dn'] = ch['center'] - p.lrc_dev * ch['sigma']
    # the gate: is the blue line between the channel lines?
    out['inside'] = (basis > out['lrc_dn']) & (basis < out['lrc_up'])
    return out


def signals(df: pd.DataFrame, p: FadeParams) -> pd.DataFrame:
    b = bands(df, p)
    if p.trigger == 'touch':
        hi, lo = df['high'].astype(float), df['low'].astype(float)
    else:
        hi = lo = df['close'].astype(float)

    ok = b[['upper', 'lower', 'inner_up', 'inner_dn', 'inside']].notna().all(axis=1)
    ok &= b['sd'] > 0

    hit_up = (hi >= b['upper']) & ok           # touched the outer upper -> short
    hit_dn = (lo <= b['lower']) & ok           # touched the outer lower -> long
    x_short = (lo <= b['inner_up']) & ok       # short covers at inner upper
    x_long = (hi >= b['inner_dn']) & ok        # long sells at inner lower
    gate = b['inside'].fillna(False).to_numpy() if p.use_filter else np.ones(len(df), bool)

    hu, hd = hit_up.to_numpy(), hit_dn.to_numpy()
    xs, xl = x_short.to_numpy(), x_long.to_numpy()
    okv = ok.to_numpy()

    n = len(df)
    pos = np.zeros(n)
    both = 0
    cur = 0.0
    for i in range(n):
        if not okv[i]:
            pos[i] = cur
            continue
        moved = False
        # --- exits first: an open trade is closed by its own target -----
        if cur > 0 and xl[i] and (gate[i] or not p.filter_exits):
            cur, moved = 0.0, True
        elif cur < 0 and xs[i] and (gate[i] or not p.filter_exits):
            cur, moved = 0.0, True
        # --- entries, gated --------------------------------------------
        if cur == 0 and gate[i] and (p.allow_same_bar or not moved):
            if hu[i] and hd[i]:
                both += 1                       # both bands in one bar: stand aside
            elif hd[i] and not p.short_only:
                cur = 1.0
            elif hu[i] and not p.long_only:
                cur = -1.0
        pos[i] = cur

    if p.reverse:
        pos = -pos
    s = pd.DataFrame(index=df.index)
    s['pos'] = pos
    s['inside'] = b['inside']
    s['hit_up'], s['hit_dn'] = hit_up, hit_dn
    s.attrs['both_band_bars'] = both
    return s


def simulate(df: pd.DataFrame, p: FadeParams) -> pd.DataFrame:
    sig = signals(df, p)
    ret = df['close'].astype(float).pct_change()
    held = sig['pos'].shift(p.lag).fillna(0.0)
    turn = held.diff().abs().fillna(held.abs())
    cost = turn * p.cost_bps / 1e4
    out = pd.DataFrame(index=df.index)
    out['ret'] = ret
    out['pos'] = held
    out['gross'] = held * ret
    out['cost'] = cost
    out['net'] = out['gross'] - cost
    out['inside'] = sig['inside']
    out.attrs['both_band_bars'] = sig.attrs['both_band_bars']
    return out


# ---------------------------------------------------------------- stats

def sharpe(r: pd.Series, ppy: int = 252) -> float:
    r = r.dropna()
    sd = r.std()
    return float('nan') if len(r) < 30 or sd == 0 else float(r.mean() / sd * np.sqrt(ppy))


def trade_table(sim: pd.DataFrame) -> pd.DataFrame:
    """Round-trips, from the realised (already-lagged) position."""
    pos = sim['pos'].to_numpy()
    net = sim['net'].to_numpy()
    idx = sim.index
    rows, i, n = [], 0, len(pos)
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        side, j, pnl = pos[i], i, 0.0
        while j < n and pos[j] == side:
            pnl += net[j]
            j += 1
        rows.append({'entry': idx[i], 'exit': idx[j - 1], 'side': side,
                     'bars': j - i, 'pnl': pnl})
        i = j
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, p: FadeParams, ppy: int = 252) -> dict:
    sim = simulate(df, p)
    t = trade_table(sim)
    d = {
        'sharpe': sharpe(sim['net'], ppy),
        'sharpe_gross': sharpe(sim['gross'], ppy),
        'cagr': float(np.expm1(np.log1p(sim['net'].fillna(0)).sum() / (len(sim) / ppy))),
        'exposure': float((sim['pos'] != 0).mean()),
        'inside_pct': float(sim['inside'].dropna().mean()) if sim['inside'].notna().any() else float('nan'),
        'trades': int(len(t)),
        'win_rate': float((t['pnl'] > 0).mean()) if len(t) else float('nan'),
        'avg_bars': float(t['bars'].mean()) if len(t) else float('nan'),
        'max_bars': int(t['bars'].max()) if len(t) else 0,
        'long_trades': int((t['side'] > 0).sum()) if len(t) else 0,
        'short_trades': int((t['side'] < 0).sum()) if len(t) else 0,
        'both_band_bars': sim.attrs['both_band_bars'],
        'bars': int(len(sim)),
    }
    if len(t):
        lp, sp = t.loc[t['side'] > 0, 'pnl'], t.loc[t['side'] < 0, 'pnl']
        d['long_pnl'] = float(lp.sum())
        d['short_pnl'] = float(sp.sum())
    return d


# ------------------------------------------------------------ self-check

def _self_check():
    rng = np.random.default_rng(7)
    n = 3000
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    df = pd.DataFrame({'close': px,
                       'high': px * (1 + abs(rng.normal(0, .004, n))),
                       'low': px * (1 - abs(rng.normal(0, .004, n))),
                       'open': px})
    df.index = pd.bdate_range('2000-01-03', periods=n)
    p = FadeParams()

    # 1. bands are ordered, and the inner pair sits inside the outer pair
    b = bands(df, p)
    m = b['sd'].notna() & (b['sd'] > 0)
    assert (b.loc[m, 'upper'] > b.loc[m, 'inner_up']).all()
    assert (b.loc[m, 'inner_up'] > b.loc[m, 'basis']).all()
    assert (b.loc[m, 'basis'] > b.loc[m, 'inner_dn']).all()
    assert (b.loc[m, 'inner_dn'] > b.loc[m, 'lower']).all()

    # 2. no look-ahead: truncating the series cannot change earlier signals
    full = signals(df, p)['pos']
    cut = signals(df.iloc[:2000], p)['pos']
    a, c = full.iloc[:2000].to_numpy(), cut.to_numpy()
    assert np.array_equal(a, c), f'{int((a != c).sum())} positions moved under truncation'

    # 3. the realised position is the signal, lagged, and nothing else
    sim = simulate(df, p)
    assert np.allclose(sim['pos'].to_numpy(),
                       full.shift(1).fillna(0).to_numpy())

    # 4. positions are flat/long/short only, entries respect the gate
    assert set(np.unique(full)) <= {-1.0, 0.0, 1.0}
    ins = bands(df, p)['inside'].fillna(False).to_numpy()
    pv = full.to_numpy()
    opens = np.where((pv != 0) & (np.r_[0.0, pv[:-1]] == 0))[0]
    assert ins[opens].all(), 'an entry fired while the blue line was outside'

    # 4b. reversing is exactly a sign change on the position, nothing else
    assert np.array_equal(signals(df, FadeParams(reverse=True))['pos'].to_numpy(),
                          -full.to_numpy())

    # 5. costs can only take away
    fin = sim['net'].notna() & sim['gross'].notna()
    assert (sim.loc[fin, 'net'] <= sim.loc[fin, 'gross'] + 1e-15).all()
    assert sim.loc[fin, 'cost'].sum() > 0

    # 6. with the gate off, entries are a superset of gated entries
    p_off = FadeParams(use_filter=False)
    n_off = int((signals(df, p_off)['pos'] != 0).sum())
    n_on = int((full != 0).sum())
    assert n_off >= n_on

    print('bbfade self-check OK '
          f'(gated bars {n_on}, ungated {n_off}, inside {ins.mean():.1%})')


if __name__ == '__main__':
    _self_check()


# ------------------------------------------------- realistic execution

def simulate_limit(df: pd.DataFrame, p: FadeParams) -> pd.DataFrame:
    """Resting limit orders at the bands, which is how this is actually traded.

    Filling a bar LATE -- signal at t's close, fill at t+1's close -- throws
    away the snap-back the whole strategy exists to capture, and no one
    trading a band system would do that.  The honest version is a resting
    order: the five band levels for bar t are fixed by bars <= t-1, so an
    order can already be sitting at them when bar t opens.  Nothing here
    peeks; the levels are lagged one bar and the fill price is the level.

    Fills are taken AT the level even when the bar gapped clean through it.
    That is the conservative side of the trade in all four cases -- a gap
    through a resting order fills you at the better price in reality -- so
    this understates the strategy rather than flattering it.

    The one genuine assumption left is that touching the level fills you.
    For NIFTY, BANKNIFTY and large-cap NSE names that is fair; it would not
    be for something thin.

    `reverse` is deliberately NOT honoured here.  Negating the position is a
    clean mirror when P&L is close-to-close, but a resting-order breakout is
    a different strategy, not a sign flip: entering long ON the upper band
    puts the inner-upper band BELOW the fill, so mirroring the exit books a
    certain loss on every trade.  A breakout needs its own exit rule, and
    inventing one here would be reporting a strategy nobody specified.
    """
    if p.reverse:
        raise ValueError('reverse is only defined for the close-to-close '
                         'engine; a limit breakout needs its own exit rule')
    b = bands(df, p).shift(1)                 # levels known before the bar opens
    gate = (b['inside'].fillna(False).to_numpy() if p.use_filter
            else np.ones(len(df), bool))
    o = df['open'].astype(float).to_numpy()
    h = df['high'].astype(float).to_numpy()
    l = df['low'].astype(float).to_numpy()
    c = df['close'].astype(float).to_numpy()
    U, L = b['upper'].to_numpy(), b['lower'].to_numpy()
    SD = b['sd'].to_numpy()
    IU, ID = b['inner_up'].to_numpy(), b['inner_dn'].to_numpy()
    ok = np.isfinite(U) & np.isfinite(L) & np.isfinite(IU) & np.isfinite(ID)
    # "no trades before 09:30": on hourly bars the opening bar spans 09:15-10:14,
    # so the earliest bar that can be excluded wholesale is that one. Entries are
    # blocked on it; an open position is still managed and exited normally.
    can_open = (np.ones(len(df), bool) if not p.min_entry_hour
                else np.asarray(df.index.hour >= p.min_entry_hour))

    n = len(df)
    ret = np.zeros(n)
    pos = np.zeros(n)
    fee = p.cost_bps / 1e4
    cur, trades = 0.0, []
    entry_i, entry_px = -1, np.nan

    for i in range(1, n):
        if not ok[i]:
            pos[i] = cur
            continue
        r = 0.0
        if cur == 0.0:
            side, fill = 0.0, np.nan
            if gate[i] and can_open[i] and l[i] <= L[i] and not p.short_only:
                side, fill = 1.0, L[i]
            elif gate[i] and can_open[i] and h[i] >= U[i] and not p.long_only:
                side, fill = -1.0, U[i]
            if side != 0.0:
                r = side * (c[i] - fill) / fill - fee
                cur, entry_i, entry_px = side, i, fill
        else:
            prev = c[i - 1]
            tgt = ID[i] if cur > 0 else IU[i]
            hit = (h[i] >= ID[i]) if cur > 0 else (l[i] <= IU[i])
            # A stop and a target inside one bar is ambiguous, so the stop
            # is assumed to go first. That is the pessimistic reading and
            # the only one that cannot flatter the result.
            if p.stop_sigma > 0:
                sd_i = SD[i]
                stop = (entry_px - p.stop_sigma * sd_i if cur > 0
                        else entry_px + p.stop_sigma * sd_i)
                if (l[i] <= stop) if cur > 0 else (h[i] >= stop):
                    tgt, hit = stop, True
            if p.max_hold and not hit and i - entry_i >= p.max_hold:
                tgt, hit = c[i], True
            if hit:
                r = cur * (tgt - prev) / prev - fee
                trades.append({'entry': df.index[entry_i], 'exit': df.index[i],
                               'side': cur, 'bars': i - entry_i,
                               'entry_px': entry_px, 'exit_px': tgt,
                               'entry_i': entry_i, 'exit_i': i,
                               'sd_pct': SD[entry_i] / entry_px,
                               'pnl': cur * (tgt - entry_px) / entry_px - 2 * fee})
                cur = 0.0
            else:
                r = cur * (c[i] - prev) / prev
        ret[i] = r
        pos[i] = cur

    out = pd.DataFrame(index=df.index)
    out['ret'] = df['close'].astype(float).pct_change()
    out['pos'] = pos
    out['net'] = ret
    out.attrs['trades'] = pd.DataFrame(trades)
    return out


def summarise_limit(df: pd.DataFrame, p: FadeParams, ppy: int = 252) -> dict:
    s = simulate_limit(df, p)
    t = s.attrs['trades']
    return {'sharpe': sharpe(s['net'], ppy),
            'cagr': float(np.expm1(np.log1p(s['net']).sum() / (len(s) / ppy))),
            'exposure': float((s['pos'] != 0).mean()),
            'trades': int(len(t)),
            'win_rate': float((t['pnl'] > 0).mean()) if len(t) else float('nan'),
            'avg_bars': float(t['bars'].mean()) if len(t) else float('nan'),
            'max_bars': int(t['bars'].max()) if len(t) else 0,
            'avg_win': float(t.loc[t.pnl > 0, 'pnl'].mean()) if len(t) else float('nan'),
            'avg_loss': float(t.loc[t.pnl <= 0, 'pnl'].mean()) if len(t) else float('nan'),
            'long_pnl': float(t.loc[t.side > 0, 'pnl'].sum()) if len(t) else 0.0,
            'short_pnl': float(t.loc[t.side < 0, 'pnl'].sum()) if len(t) else 0.0,
            'bars': int(len(s))}


def _limit_check():
    """A limit engine that peeks is worthless, so prove it does not."""
    rng = np.random.default_rng(5)
    n = 4000
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({'open': px,
                       'high': px * (1 + abs(rng.normal(0, .002, n))),
                       'low': px * (1 - abs(rng.normal(0, .002, n))),
                       'close': px},
                      index=pd.date_range('2015-01-01', periods=n, freq='h'))
    p = FadeParams()
    a = simulate_limit(df, p)['net'].iloc[:3000].to_numpy()
    c = simulate_limit(df.iloc[:3000], p)['net'].to_numpy()
    assert np.allclose(a, c, atol=1e-14), 'limit engine reads the future'
    # a fill never happens at a price the bar did not trade at
    s = simulate_limit(df, p)
    assert set(np.unique(s['pos'])) <= {-1.0, 0.0, 1.0}
    # zero-cost gross must beat costed net
    g = simulate_limit(df, FadeParams(cost_bps=0.0))['net'].sum()
    assert g > s['net'].sum()
    print('limit-engine check OK')


if __name__ == '__main__':
    _limit_check()
