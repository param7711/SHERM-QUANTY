"""
Why the fade works on some series and not others.

Four measurements, all on the same footing across every instrument:

1. Variance ratio VR(k) = Var(k-bar return) / (k * Var(1-bar return)).
   VR < 1 means multi-bar moves are smaller than a random walk implies --
   the series reverts. VR > 1 means it trends. This is the property a
   band-fade is betting on, measured directly and without any strategy.

2. Lag-1 autocorrelation of returns. The same bet at one-bar horizon.

3. The event study that IS the strategy: after price closes beyond the
   DevLucem band, what does it do next? Mean forward return over the
   holding horizon, long and short sides separately, with a t-stat.
   No parameters, no costs, no state machine -- just the conditional mean.

4. The leverage effect: corr(return, next-bar realised vol) and whether
   down-moves revert harder than up-moves. This is the usual explanation
   for index mean-reversion (fading a panic earns the variance premium),
   so it is worth measuring rather than asserting.
"""

import numpy as np
import pandas as pd
from scipy import stats

from channel import regression_channel


def variance_ratio(r: pd.Series, k: int) -> float:
    r = r.dropna()
    v1 = r.var(ddof=1)
    vk = r.rolling(k).sum().dropna().var(ddof=1)
    return float(vk / (k * v1)) if v1 > 0 else np.nan


def band_event_study(close: pd.Series, length: int = 100, dev: float = 2.0,
                     horizon: int = 5) -> dict:
    """Forward returns conditioned on a band break, the fade's raw edge."""
    z = regression_channel(close, length, ddof=0)['z']
    fwd = close.shift(-horizon) / close - 1.0

    lo = (z <= -dev) & fwd.notna()
    hi = (z >= dev) & fwd.notna()
    base = fwd.dropna()

    def side(mask, sign):
        x = fwd[mask]
        if len(x) < 30:
            return np.nan, np.nan, len(x)
        # Excess over the unconditional drift: a long fade in a rising
        # market must beat simply being long, not beat zero.
        excess = sign * (x - base.mean())
        t = stats.ttest_1samp(excess, 0.0)
        return float(excess.mean()), float(t.statistic), int(len(x))

    lo_m, lo_t, lo_n = side(lo, +1)
    hi_m, hi_t, hi_n = side(hi, -1)
    return {
        'lower_break_n': lo_n, 'lower_excess': lo_m, 'lower_t': lo_t,
        'upper_break_n': hi_n, 'upper_excess': hi_m, 'upper_t': hi_t,
        'base_fwd': float(base.mean()),
    }


def leverage_effect(close: pd.Series, vol_win: int = 5) -> dict:
    r = np.log(close / close.shift(1))
    fwd_vol = r.rolling(vol_win).std().shift(-vol_win)
    ok = r.notna() & fwd_vol.notna()
    lev = float(np.corrcoef(r[ok], fwd_vol[ok])[0, 1])

    # Do down-moves revert harder than up-moves? Compare the mean 5-bar
    # forward return after a 1-sd down day vs after a 1-sd up day.
    sd = r.std()
    fwd5 = close.shift(-5) / close - 1.0
    down = fwd5[(r <= -sd) & fwd5.notna()]
    up = fwd5[(r >= sd) & fwd5.notna()]
    return {
        'leverage_corr': lev,
        'after_down_day': float(down.mean()) if len(down) > 30 else np.nan,
        'after_up_day': float(up.mean()) if len(up) > 30 else np.nan,
        'reversion_asymmetry': (float(down.mean() - up.mean())
                                if len(down) > 30 and len(up) > 30 else np.nan),
    }


def profile(close: pd.Series, name: str, klass: str, horizon: int = 5) -> dict:
    r = close.pct_change()
    out = {'instrument': name, 'class': klass, 'bars': len(close)}
    out['ac1'] = float(r.autocorr(1))
    out['ac5'] = float(r.autocorr(5))
    for k in (2, 5, 10, 20):
        out[f'vr{k}'] = variance_ratio(r, k)
    out.update(band_event_study(close, horizon=horizon))
    out.update(leverage_effect(close))
    return out
