"""
Signal state machine and vectorised P&L for the regime-switching hypothesis.

Hypothesis as stated:
  - the regression channel says whether the market is trending or ranging;
  - inside the channel -> mean-revert (buy the lower band, short the upper
    band, take profit back at the centre line);
  - when the 20-SMA breaks out of the channel -> follow the trend (long
    above the upper band, short below the lower band).

Execution convention: the signal is formed from bar t's close and is
filled at bar t+1 (`lag=1`); P&L is close-to-close. Costs are charged on
every unit of position change. No look-ahead anywhere.
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from channel import channel_features

FLAT, MR_LONG, MR_SHORT, TF_LONG, TF_SHORT = 0, 1, 2, 3, 4
LEG_NONE, LEG_MR, LEG_TF = 0, 1, 2


@dataclass(frozen=True)
class Params:
    n: int = 100            # channel lookback (bars)
    k: float = 2.0          # channel half-width, in residual sigmas
    sma_len: int = 20       # moving average that triggers the trend leg
    entry_frac: float = 0.9  # mean-reversion entry at |z| >= entry_frac * k
    exit_z: float = 0.0     # mean-reversion exit when z crosses this level
    use_mr: bool = True
    use_tf: bool = True
    max_hold: int = 0       # bars; 0 = no time stop
    stop_sigma: float = 0.0  # adverse move past entry, in sigmas; 0 = no stop
    trend_trigger: str = 'sma'  # 'sma' (as hypothesised) or 'price'
    k_trend: float = 0.0    # breakout threshold; 0 = reuse k
    slope_max: float = 0.0  # gate MR on a flat channel; 0 = no gate
    slope_min_trend: float = 0.0  # gate the trend leg on a steep channel

    def key(self) -> str:
        return (f"n{self.n}_k{self.k}_sma{self.sma_len}_e{self.entry_frac}"
                f"_x{self.exit_z}_mr{int(self.use_mr)}_tf{int(self.use_tf)}"
                f"_h{self.max_hold}_s{self.stop_sigma}"
                f"_trig{self.trend_trigger}_kt{self.k_trend}"
                f"_sl{self.slope_max}_{self.slope_min_trend}")


def generate_positions(close: pd.Series, p: Params) -> pd.DataFrame:
    """Target position at each bar's close, plus which leg produced it."""
    feat = channel_features(close, p.n, p.sma_len)
    z = feat['z'].to_numpy()
    smaz = feat['smaz'].to_numpy()
    t = len(z)

    pos = np.zeros(t)
    leg = np.zeros(t, dtype=np.int8)
    state = FLAT
    entry_z = 0.0
    bars_in = 0
    entry_k = p.entry_frac * p.k
    k_trend = p.k_trend if p.k_trend > 0 else p.k
    # 'sma': the 20-SMA has to clear the band, exactly as hypothesised.
    # 'price': price itself clears the band -- a strictly easier trigger,
    # used to show what the trend leg does once it can actually fire.
    trend_metric = smaz if p.trend_trigger == 'sma' else z
    # Drift of the fitted line across the whole window, in residual sigmas:
    # the channel's own read on how directional the market is.
    tstr = (feat['slope'] * p.n / feat['sigma']).to_numpy()

    for i in range(t):
        zi, si = z[i], trend_metric[i]
        if not np.isfinite(zi) or not np.isfinite(si):
            state, bars_in = FLAT, 0
            continue

        ti = tstr[i]
        steep_ok_up = p.slope_min_trend <= 0 or ti >= p.slope_min_trend
        steep_ok_dn = p.slope_min_trend <= 0 or ti <= -p.slope_min_trend
        flat_ok = p.slope_max <= 0 or abs(ti) <= p.slope_max

        trend_up = p.use_tf and si > k_trend and steep_ok_up
        trend_dn = p.use_tf and si < -k_trend and steep_ok_dn

        if trend_up:
            if state != TF_LONG:
                state, entry_z, bars_in = TF_LONG, zi, 0
        elif trend_dn:
            if state != TF_SHORT:
                state, entry_z, bars_in = TF_SHORT, zi, 0
        else:
            # Back inside the channel: any trend position is closed here.
            if state in (TF_LONG, TF_SHORT):
                state, bars_in = FLAT, 0

            if state == MR_LONG:
                stop = p.stop_sigma > 0 and zi <= entry_z - p.stop_sigma
                timed = p.max_hold > 0 and bars_in >= p.max_hold
                if zi >= p.exit_z or stop or timed:
                    state, bars_in = FLAT, 0
            elif state == MR_SHORT:
                stop = p.stop_sigma > 0 and zi >= entry_z + p.stop_sigma
                timed = p.max_hold > 0 and bars_in >= p.max_hold
                if zi <= -p.exit_z or stop or timed:
                    state, bars_in = FLAT, 0

            if state == FLAT and p.use_mr and flat_ok:
                if zi <= -entry_k:
                    state, entry_z, bars_in = MR_LONG, zi, 0
                elif zi >= entry_k:
                    state, entry_z, bars_in = MR_SHORT, zi, 0

        if state == FLAT:
            pos[i], leg[i] = 0.0, LEG_NONE
        elif state == MR_LONG:
            pos[i], leg[i] = 1.0, LEG_MR
        elif state == MR_SHORT:
            pos[i], leg[i] = -1.0, LEG_MR
        elif state == TF_LONG:
            pos[i], leg[i] = 1.0, LEG_TF
        else:
            pos[i], leg[i] = -1.0, LEG_TF

        bars_in = bars_in + 1 if state != FLAT else 0

    out = feat.copy()
    out['pos'] = pos
    out['leg'] = leg
    return out


def simulate(close: pd.Series, p: Params, cost_bps: float = 5.0,
             lag: int = 1) -> pd.DataFrame:
    """Per-bar net returns for one parameter set."""
    sig = generate_positions(close, p)
    ret = close.pct_change().fillna(0.0)

    held = sig['pos'].shift(lag).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    gross = held * ret
    cost = turnover * (cost_bps / 1e4)

    sig['ret_gross'] = gross
    sig['cost'] = cost
    sig['ret_net'] = gross - cost
    sig['held'] = held
    sig['leg_held'] = sig['leg'].shift(lag).fillna(0).astype(int)
    sig['bh'] = ret
    return sig


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def metrics(ret: pd.Series, bars_per_year: int, held: pd.Series = None,
            label: str = '') -> dict:
    r = ret.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0 or r.std() == 0:
        return {'label': label, 'bars': int(r.size), 'cagr': 0.0, 'vol': 0.0,
                'sharpe': 0.0, 'sortino': 0.0, 'max_dd': 0.0, 'calmar': 0.0,
                't_stat': 0.0, 'exposure': 0.0, 'total_return': 0.0,
                'skew': 0.0, 'best_bar': 0.0, 'worst_bar': 0.0}

    equity = np.cumprod(1.0 + r)
    years = r.size / bars_per_year
    cagr = equity[-1] ** (1 / years) - 1 if equity[-1] > 0 else -1.0
    vol = r.std(ddof=1) * np.sqrt(bars_per_year)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(bars_per_year)
    downside = r[r < 0]
    sortino = (r.mean() / downside.std(ddof=1) * np.sqrt(bars_per_year)
               if downside.size > 1 and downside.std(ddof=1) > 0 else np.nan)
    mdd = _max_drawdown(equity)

    out = {
        'label': label,
        'bars': int(r.size),
        'years': round(years, 2),
        'total_return': float(equity[-1] - 1),
        'cagr': float(cagr),
        'vol': float(vol),
        'sharpe': float(sharpe),
        'sortino': float(sortino) if np.isfinite(sortino) else np.nan,
        'max_dd': float(mdd),
        'calmar': float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        't_stat': float(r.mean() / r.std(ddof=1) * np.sqrt(r.size)),
        'skew': float(pd.Series(r).skew()),
        'best_bar': float(r.max()),
        'worst_bar': float(r.min()),
    }
    if held is not None:
        h = held.to_numpy(dtype=float)
        out['exposure'] = float(np.mean(np.abs(h) > 0))
        out['long_share'] = float(np.mean(h > 0))
        out['short_share'] = float(np.mean(h < 0))
    return out


def trade_log(sig: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-bar position series into discrete trades."""
    held = sig['held'].to_numpy()
    legs = sig['leg_held'].to_numpy()
    net = sig['ret_net'].to_numpy()
    idx = sig.index

    trades = []
    i, t = 0, len(held)
    while i < t:
        if held[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < t and held[j + 1] == held[i] and legs[j + 1] == legs[i]:
            j += 1
        pnl = float(np.prod(1.0 + net[i:j + 1]) - 1.0)
        trades.append({
            'entry': idx[i], 'exit': idx[j], 'bars': j - i + 1,
            'direction': int(np.sign(held[i])),
            'leg': 'MR' if legs[i] == LEG_MR else 'TF',
            'pnl': pnl,
        })
        i = j + 1
    return pd.DataFrame(trades)


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'hit_rate': np.nan, 'avg_pnl': np.nan,
                'profit_factor': np.nan, 'avg_bars': np.nan}
    wins = trades.loc[trades['pnl'] > 0, 'pnl']
    losses = trades.loc[trades['pnl'] <= 0, 'pnl']
    return {
        'n_trades': int(len(trades)),
        'hit_rate': float((trades['pnl'] > 0).mean()),
        'avg_pnl': float(trades['pnl'].mean()),
        'avg_win': float(wins.mean()) if len(wins) else np.nan,
        'avg_loss': float(losses.mean()) if len(losses) else np.nan,
        'profit_factor': float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf,
        'avg_bars': float(trades['bars'].mean()),
    }


def run(close: pd.Series, p: Params, bars_per_year: int,
        cost_bps: float = 5.0, lag: int = 1) -> dict:
    sig = simulate(close, p, cost_bps=cost_bps, lag=lag)
    m = metrics(sig['ret_net'], bars_per_year, held=sig['held'], label=p.key())
    m.update(trade_stats(trade_log(sig)))
    m.update({f'param_{k}': v for k, v in asdict(p).items()})
    m['cost_bps'] = cost_bps
    m['gross_sharpe'] = metrics(sig['ret_gross'], bars_per_year)['sharpe']
    return m


def _self_check():
    import data as D
    close = D.load('SP500')['close']

    p = Params()
    sig = simulate(close, p)

    # No look-ahead: the realised position must be a lagged copy of the signal.
    assert (sig['held'].iloc[1:].to_numpy()
            == sig['pos'].iloc[:-1].to_numpy()).all()

    # Truncating the series must not alter any earlier position.
    trunc = simulate(close.iloc[:3000], p)
    assert np.allclose(trunc['pos'].to_numpy(),
                       sig['pos'].iloc[:3000].to_numpy())

    # Costs must strictly reduce P&L whenever there is any turnover.
    assert sig['cost'].sum() > 0
    assert sig['ret_net'].sum() < sig['ret_gross'].sum()

    # Positions are only ever -1, 0, +1.
    assert set(np.unique(sig['pos'])) <= {-1.0, 0.0, 1.0}

    res = run(close, p, 252)
    print(f"SP500 baseline  sharpe={res['sharpe']:.2f}  cagr={res['cagr']:.2%} "
          f"dd={res['max_dd']:.1%}  trades={res['n_trades']}  "
          f"exposure={res['exposure']:.1%}")
    print('backtest.py self-check passed (causality, costs, position bounds)')


if __name__ == '__main__':
    _self_check()
