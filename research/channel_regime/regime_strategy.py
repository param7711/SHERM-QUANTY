"""
The hypothesis as the chart actually states it.

Two indicators sit on the chart, so there are two candidate regime
variables, and v1/v2 tested the wrong one. This module tests the right
one: RegDet's five-label state as the trend/range switch, with the
DevLucem channel supplying the fade levels and the RegDet Bollinger
block supplying the 0.3-sigma inner band and the 2.0-sigma outer cap.

  regime      RegDet label (H_BULL / L_BULL / SIDEWAYS / L_BEAR / H_BEAR)
  fade level  price crossing the DevLucem band at |z| = dev
  exit        the regression centre line, or back inside the 0.3s band
  cap         no new fade once |price - SMA20| exceeds outer_mult * sd

Same execution convention as the rest of the study: signal on bar t's
close, filled at t+1, close-to-close, costs on every position change.
"""

import argparse
import os

import numpy as np
import pandas as pd

import data as D
import regdet as RD
from channel import regression_channel
from backtest import metrics, trade_log, trade_stats

RESULTS = os.path.join(D.ROOT, 'results', 'channel_regime')
INSTRUMENTS = ['SP500', 'NASDAQ', 'WTI', 'GOOG', 'EURUSD']
COST_BPS = 5.0

HIGH = ('H_BULL', 'H_BEAR')
LOW = ('L_BULL', 'L_BEAR')
SIDE = ('SIDEWAYS',)


def positions(close: pd.Series, *, length: int = 100, dev: float = 2.0,
              fade_regimes=SIDE + LOW, trend_regimes=(), cap_mult: float = 0.0,
              inner_mult: float = 0.3, outer_mult: float = 2.0,
              exit_at: str = 'center', bb_length: int = 20,
              rd: RD.RegDetParams = RD.RegDetParams(),
              regime_override=None) -> pd.DataFrame:
    ch = regression_channel(close, length, ddof=0)      # DevLucem convention
    bb = RD.bollinger(close, bb_length, inner_mult, outer_mult)

    z = ch['z'].to_numpy()
    bbz = bb['bbz'].to_numpy()
    # regime_override exists for the control tests: feed in a rotated or
    # reshuffled label series to ask whether the detector's TIMING matters
    # or only the amount of exposure it allows.
    regime = (RD.compute(close, rd)['regime'].to_numpy()
              if regime_override is None else np.asarray(regime_override))

    t = len(close)
    pos = np.zeros(t)
    leg = np.empty(t, dtype=object)
    state = 0            # 0 flat, +-1 fade, +-2 trend

    for i in range(t):
        zi, bi, ri = z[i], bbz[i], regime[i]
        if not np.isfinite(zi) or not isinstance(ri, str):
            state = 0
            pos[i], leg[i] = 0.0, None
            continue
        if not np.isfinite(bi):
            bi = 0.0

        if ri in trend_regimes:
            state = 2 if ri == 'H_BULL' else -2
        else:
            if abs(state) == 2:
                state = 0
            if state == 1:
                done = (bi >= -inner_mult) if exit_at == 'inner' else (zi >= 0.0)
                if done:
                    state = 0
            elif state == -1:
                done = (bi <= inner_mult) if exit_at == 'inner' else (zi <= 0.0)
                if done:
                    state = 0
            if state == 0 and ri in fade_regimes:
                capped = cap_mult > 0 and abs(bi) > cap_mult
                if not capped:
                    if zi <= -dev:
                        state = 1
                    elif zi >= dev:
                        state = -1

        pos[i] = np.sign(state)
        leg[i] = None if state == 0 else ('TF' if abs(state) == 2 else 'MR')

    out = pd.DataFrame({'pos': pos, 'z': z, 'bbz': bbz, 'regime': regime},
                       index=close.index)
    out['leg'] = leg
    return out


def simulate(close: pd.Series, cost_bps: float = COST_BPS, **kw) -> pd.DataFrame:
    sig = positions(close, **kw)
    ret = close.pct_change().fillna(0.0)
    held = sig['pos'].shift(1).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    sig['held'] = held
    sig['leg_held'] = sig['leg'].shift(1)
    sig['ret_gross'] = held * ret
    sig['cost'] = turnover * (cost_bps / 1e4)
    sig['ret_net'] = sig['ret_gross'] - sig['cost']
    sig['bh'] = ret
    return sig


def run(close: pd.Series, bars_per_year: int, cost_bps: float = COST_BPS,
        **kw) -> dict:
    sig = simulate(close, cost_bps=cost_bps, **kw)
    m = metrics(sig['ret_net'], bars_per_year, held=sig['held'])
    tl = trade_log(sig.assign(leg_held=sig['leg_held'].map(
        {'MR': 1, 'TF': 2}).fillna(0).astype(int)))
    m.update(trade_stats(tl))
    return m


SPECS = {
    'R0  fade everywhere (no regime filter)':
        dict(fade_regimes=SIDE + LOW + HIGH),
    'R1  fade only in SIDEWAYS':
        dict(fade_regimes=SIDE),
    'R2  fade unless the tape is high-intensity':
        dict(fade_regimes=SIDE + LOW),
    'R3  R2 + follow the trend in H_BULL / H_BEAR':
        dict(fade_regimes=SIDE + LOW, trend_regimes=HIGH),
    'R4  R3 + 2.0s outer-band cap on new fades':
        dict(fade_regimes=SIDE + LOW, trend_regimes=HIGH, cap_mult=2.0),
    'R5  R1 + 2.0s outer-band cap':
        dict(fade_regimes=SIDE, cap_mult=2.0),
    'R6  trend leg only (H_BULL / H_BEAR)':
        dict(fade_regimes=(), trend_regimes=HIGH),
    'R7  R4 but exiting at the 0.3s inner band':
        dict(fade_regimes=SIDE + LOW, trend_regimes=HIGH, cap_mult=2.0,
             exit_at='inner'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost', type=float, default=COST_BPS)
    args = ap.parse_args()

    rows = []
    for label, kw in SPECS.items():
        row = {'spec': label}
        sh = []
        for inst in INSTRUMENTS:
            m = run(D.load(inst)['close'], D.BARS_PER_YEAR[inst],
                    cost_bps=args.cost, **kw)
            row[inst] = m['sharpe']
            if inst == 'SP500':
                row['trades'] = m['n_trades']
                row['exposure'] = m['exposure']
            sh.append(m['sharpe'])
        row['daily_mean'] = float(np.mean(sh[:4]))
        row['n_pos_daily'] = int(sum(x > 0 for x in sh[:4]))
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, '15_regdet_specs.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
