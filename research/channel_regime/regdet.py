"""
RegDet — Sherm Quanty regime detector, ported from regdet_v6_bb_IL.pine.

Line-for-line port of the Pine v6 indicator at its shipped defaults, with
the volatility cap ON (volCapOn=true, which is that file's default). At
w=1.0 the HMM term drops out exactly, so the Pine script is the whole
detector and nothing is approximated here.

Emits the same five labels: H_BULL, L_BULL, SIDEWAYS, L_BEAR, H_BEAR.

The indicator's own header is emphatic that these are STATE, not signal —
"the 5-label forward-return ordering is BROKEN at every horizon". This
port exists to test exactly that claim as a regime filter, not to assume
the opposite.

Pine conventions preserved:
  * ta.stdev / ta.sma default to biased=true -> population sd (ddof=0).
  * zscore() accumulates only on non-na inputs and returns 0.0 when the
    baseline is not yet defined; each call site owns its accumulator.
  * warm-up gate is bar_index > max(SWING_WIN, MOM_5D_LEN) + 30.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

LABELS = ('H_BEAR', 'L_BEAR', 'SIDEWAYS', 'L_BULL', 'H_BULL')


@dataclass(frozen=True)
class RegDetParams:
    # Bars per day. On a daily chart bpdAuto collapses to 1 for every
    # session length the input allows, so 1 is the daily-bar value.
    bpd: int = 1
    context_days: int = 12
    conf_l: float = 0.50
    confirm_bars: int = 2
    z_hi: float = 0.50
    eff_hi: float = 0.35
    z_hi_exit: float = 0.35
    eff_hi_exit: float = 0.25
    eff_win: int = 9
    bar_dir_tau: float = 1.0
    # Deviation 2 — the volatility cap. This file ships it ON.
    vol_cap_on: bool = True
    vol_win_days: float = 3.0
    vol_hi: float = 0.50
    vol_hi_exit: float = 1.00
    baseline: str = 'expanding'   # or 'rolling'
    baseline_len: int = 500

    def __post_init__(self):
        if self.z_hi_exit > self.z_hi or self.eff_hi_exit > self.eff_hi:
            raise ValueError('Inverted hysteresis: exits must be <= entries.')
        if self.vol_cap_on and self.vol_hi_exit < self.vol_hi:
            raise ValueError('Inverted volatility hysteresis: '
                             'VOL_HI_EXIT must be >= VOL_HI.')


def _zscore_expanding(x: pd.Series) -> np.ndarray:
    """Pine's zscore() with the expanding baseline.

    The accumulators advance only on non-na bars, so the running mean and
    population sd are taken over the non-na history to date -- not over
    bar count.
    """
    v = x.to_numpy(dtype=float)
    ok = np.isfinite(v)
    filled = np.where(ok, v, 0.0)
    n = np.cumsum(ok)
    s = np.cumsum(filled)
    s2 = np.cumsum(filled * filled)

    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, s2 / np.maximum(n, 1) - mu * mu, np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))

    out = np.zeros_like(v)
    live = ok & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    out[live] = (v[live] - mu[live]) / sd[live]
    return out


def _zscore_rolling(x: pd.Series, length: int) -> np.ndarray:
    mu = x.rolling(length).mean()
    sd = x.rolling(length).std(ddof=0)
    out = (x - mu) / sd.replace(0, np.nan)
    return out.fillna(0.0).to_numpy()


def compute(close: pd.Series, p: RegDetParams = RegDetParams()) -> pd.DataFrame:
    """Return the detector's per-bar state, labels included."""
    zs = (_zscore_expanding if p.baseline == 'expanding'
          else lambda s: _zscore_rolling(s, p.baseline_len))

    mom_1d = max(2, 1 * p.bpd)
    mom_3d = max(2, 3 * p.bpd)
    mom_5d = max(2, 5 * p.bpd)
    swing = max(2, p.context_days * p.bpd)
    vol_len = max(2, int(round(p.vol_win_days * p.bpd)))

    r = np.log(close / close.shift(1))
    f_ret = r
    f_mom1d = r.rolling(mom_1d).sum()
    f_mom3d = r.rolling(mom_3d).sum()
    f_mom5d = r.rolling(mom_5d).sum()
    ma = close.rolling(swing).mean()
    f_dma = (close - ma) / ma.where(ma > 0)

    z_ret, z_m1d = zs(f_ret), zs(f_mom1d)
    z_m3d, z_m5d = zs(f_mom3d), zs(f_mom5d)
    z_dma = zs(f_dma)

    w_sum = 3.2
    composite = (1.0 * z_ret + 0.4 * z_m1d + 0.4 * z_m3d
                 + 0.4 * z_m5d + 1.0 * z_dma) / w_sum
    score = zs(pd.Series(composite, index=close.index))

    # Shift-stabilised softmax over (+s/tau, 0, -s/tau).
    e = score / max(p.bar_dir_tau, 1e-12)
    aa = np.abs(e)
    eb, es, er = np.exp(e - aa), np.exp(-aa), np.exp(-e - aa)
    tot = eb + es + er
    bull, side, bear = eb / tot, es / tot, er / tot
    confidence = np.maximum(np.maximum(bull, side), bear)

    dir_raw = np.where((bull >= side) & (bull >= bear), 1,
                       np.where((bear >= side) & (bear >= bull), -1, 0))
    dir_raw = np.where(confidence < p.conf_l, 0, dir_raw)

    # Intensity inputs.
    trend_z = z_m3d
    net_move = (close - close.shift(p.eff_win)).abs()
    path_len = close.diff().abs().rolling(p.eff_win).sum()
    eff = (net_move / path_len.where(path_len > 0)).clip(0.0, 1.0).fillna(0.0).to_numpy()

    rvol = r.rolling(vol_len).std(ddof=0)
    zvol = zs(rvol)

    calm_enter = np.ones_like(zvol, dtype=bool) if not p.vol_cap_on else zvol <= p.vol_hi
    calm_hold = np.ones_like(zvol, dtype=bool) if not p.vol_cap_on else zvol <= p.vol_hi_exit

    can_enter = (np.abs(trend_z) >= p.z_hi) & (eff >= p.eff_hi) & calm_enter
    can_hold = (np.abs(trend_z) >= p.z_hi_exit) & (eff >= p.eff_hi_exit) & calm_hold
    sgn = np.sign(trend_z).astype(int)

    t = len(close)
    dir_emit = np.zeros(t, dtype=int)
    intens = np.zeros(t, dtype=int)
    cand, run_len, emit, inten = 0, 0, None, 0

    for i in range(t):
        d = int(dir_raw[i])
        if emit is None:
            emit, cand, run_len = d, d, 0
        elif d == emit:
            emit, cand, run_len = d, d, 0
        else:
            if d == cand:
                run_len += 1
            else:
                cand, run_len = d, 1
            if run_len >= p.confirm_bars:
                emit, run_len = cand, 0
        dir_emit[i] = emit

        if inten != 0 and can_hold[i] and sgn[i] == inten:
            pass
        else:
            inten = int(sgn[i]) if (can_enter[i] and sgn[i] != 0) else 0
        intens[i] = inten

    warm = np.arange(t) > max(swing, mom_5d) + 30

    regime = np.where(
        dir_emit == 1, np.where(intens == 1, 'H_BULL', 'L_BULL'),
        np.where(dir_emit == -1, np.where(intens == -1, 'H_BEAR', 'L_BEAR'),
                 'SIDEWAYS'))
    regime = np.where(warm, regime, None)

    return pd.DataFrame({
        'score': score, 'confidence': confidence, 'trend_z': trend_z,
        'eff': eff, 'zvol': zvol, 'dir_emit': dir_emit, 'intens': intens,
        'regime': regime, 'warm': warm,
    }, index=close.index)


def bollinger(close: pd.Series, length: int = 20, inner_mult: float = 0.3,
              outer_mult: float = 2.0) -> pd.DataFrame:
    """The indicator's five-line Bollinger block.

    Pine: bbSd = ta.stdev(bbSrc, bbLength) -- biased, so ddof=0. One basis
    and ONE standard deviation shared by both pairs; the pairs differ only
    in multiplier, which is why inner and outer are directly comparable.

    Note the file's own words: this block is display only. It feeds no part
    of the regime label.
    """
    basis = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    return pd.DataFrame({
        'basis': basis, 'sd': sd,
        'inner_up': basis + inner_mult * sd,
        'inner_dn': basis - inner_mult * sd,
        'outer_up': basis + outer_mult * sd,
        'outer_dn': basis - outer_mult * sd,
        'bbz': (close - basis) / sd.replace(0, np.nan),
    }, index=close.index)


def _self_check():
    import data as D

    close = D.load('SP500')['close']
    st = compute(close)

    # Softmax masses are a distribution; confidence is their max.
    assert (st['confidence'].between(1 / 3 - 1e-9, 1.0)).all()

    # Causality: truncating the future must not change any past label.
    trunc = compute(close.iloc[:3000])
    a, b = st['regime'].iloc[:3000], trunc['regime']
    # Warm-up bars are NaN on both sides, and NaN never equals NaN.
    same = (a.to_numpy() == b.to_numpy()) | (a.isna() & b.isna()).to_numpy()
    assert same.all(), f'{(~same).sum()} labels changed under truncation'

    # Hysteresis guards fire.
    for bad in (dict(z_hi_exit=0.9), dict(vol_hi_exit=0.1)):
        try:
            RegDetParams(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'guard did not fire for {bad}')

    occ = st.loc[st['warm'], 'regime'].value_counts(normalize=True)
    print('  SP500 daily regime occupancy (vol cap on):')
    for lab in LABELS:
        print(f'    {lab:9s} {occ.get(lab, 0.0):6.1%}')

    bb = bollinger(close)
    live = bb['sd'].notna() & (bb['sd'] > 0)
    assert (bb.loc[live, 'inner_up'] < bb.loc[live, 'outer_up']).all()
    assert (bb.loc[live, 'inner_dn'] > bb.loc[live, 'outer_dn']).all()
    print('regdet.py self-check passed (causality, guards, band ordering)')


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _self_check()
