"""
Hourly FX bars -- the market with no session and no gap.

SOURCE.  github.com/ejtraderLabs/historical-data, one CSV per pair per
timeframe, already on an hourly grid.  Prices arrive scaled by 1e5
(1.27801 stored as 127801); the strategy is scale-invariant so the
figures are unaffected, but they are divided back for readability.

WHY FX IS THE RIGHT CONTROL.  NSE hourly bars carry two things a band
system cannot separate: an overnight gap on the first bar of every
session, and a strong upward drift that punishes the short leg.  FX runs
continuously five days a week and has no meaningful drift in either
direction.  If the long-only result on Indian stocks came from the drift,
long-only should NOT help here.
"""

import os
import glob
import pandas as pd

SRC = '/home/user/ejtraderlabs/historical-data'
PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF',
         'EURJPY', 'GBPJPY', 'AUDJPY', 'EURGBP', 'EURCHF']

# The weekend break is a real hole in the record, but it is 2 days in an
# otherwise continuous week -- not the daily 17.75-hour hole the NSE has.
MAX_GAP_HOURS = 96


def load(pair: str, years: float = 0.0) -> pd.DataFrame:
    f = os.path.join(SRC, pair, f'{pair}h1.csv')
    d = pd.read_csv(f, parse_dates=['Date']).set_index('Date').sort_index()
    d = d.rename(columns=str.lower)[['open', 'high', 'low', 'close']] / 1e5
    d = d[~d.index.duplicated(keep='last')]
    d = d[(d > 0).all(axis=1)]
    # drop any bar that is not a real bar: zero range and zero movement
    d = d[(d['high'] >= d['low']) & (d['close'] <= d['high']) & (d['close'] >= d['low'])]
    if years:
        d = d[d.index > d.index[-1] - pd.DateOffset(years=int(years))]
    return d


def scan() -> pd.DataFrame:
    rows = []
    for p in PAIRS:
        d = load(p)
        r = d['close'].pct_change()
        gaps = d.index.to_series().diff().dt.total_seconds() / 3600
        rows.append({'pair': p, 'bars': len(d),
                     'start': str(d.index[0])[:10], 'end': str(d.index[-1])[:10],
                     'max_abs_ret': float(r.abs().max()),
                     'n_gt2pct': int((r.abs() > 0.02).sum()),
                     'max_gap_h': float(gaps.max()),
                     'gaps_over_96h': int((gaps > MAX_GAP_HOURS).sum())})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(scan().to_string(index=False))
