"""
Indian market data — NIFTY, BANK NIFTY and major NIFTY constituents.

Source: BennyThadikaran/eod2_data, a public GitHub repository of NSE
end-of-day bars, reachable through this session's anonymous git lane (the
NSE archives themselves, like every other market-data host, are 403 at the
egress proxy). Cloned, not scraped, and pinned by commit in `SOURCE_SHA`.

Every series is screened for unadjusted corporate actions before use: a
split or bonus that the vendor has not adjusted shows up as a one-day
return near -50%, -80% or -90% and would hand any band-break strategy a
free, entirely fictional signal. `quality_scan()` reports the suspects and
`load()` refuses a series that fails.
"""

import os
import subprocess

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = '/home/user/eod2_data/daily'
OUT_DIR = os.path.join(ROOT, 'data', 'research_in')

INDICES = {
    'NIFTY50':   'nifty 50.csv',
    'BANKNIFTY': 'nifty bank.csv',
    'NIFTY500':  'nifty 500.csv',
}

# The heaviest NIFTY 50 constituents, across sectors.
STOCKS = {
    'RELIANCE':   'reliance.csv',
    'HDFCBANK':   'hdfcbank.csv',
    'ICICIBANK':  'icicibank.csv',
    'INFY':       'infy.csv',
    'TCS':        'tcs.csv',
    'ITC':        'itc.csv',
    'LT':         'lt.csv',
    'SBIN':       'sbin.csv',
    'BHARTIARTL': 'bhartiartl.csv',
    'KOTAKBANK':  'kotakbank.csv',
    'AXISBANK':   'axisbank.csv',
    'HINDUNILVR': 'hindunilvr.csv',
    'MARUTI':     'maruti.csv',
    'ASIANPAINT': 'asianpaint.csv',
    'BAJFINANCE': 'bajfinance.csv',
}

ALL = {**INDICES, **STOCKS}
ASSET_CLASS = {k: 'equity_index' for k in INDICES}
ASSET_CLASS.update({k: 'single_stock' for k in STOCKS})
BARS_PER_YEAR = {k: 252 for k in ALL}

# A corporate action the vendor did not adjust lands within a few percent
# of these ratios. Real one-day index/stock moves do not.
SPLIT_RATIOS = (0.5, 0.2, 0.1, 0.25, 1 / 3, 2 / 3)
SPLIT_TOL = 0.03


def source_sha() -> str:
    try:
        return subprocess.run(['git', '-C', '/home/user/eod2_data', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return 'unknown'


def _read(fname: str) -> pd.DataFrame:
    path = os.path.join(SRC_DIR, fname)
    df = pd.read_csv(path, usecols=range(6),
                     names=['date', 'open', 'high', 'low', 'close', 'volume'],
                     header=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df.dropna(subset=['close']).astype(float)


def quality_scan(name: str, df: pd.DataFrame) -> dict:
    r = df['close'].pct_change()
    big = r[r.abs() > 0.20]
    suspects = []
    for d, v in big.items():
        ratio = 1 + v
        if any(abs(ratio - k) < SPLIT_TOL for k in SPLIT_RATIOS):
            suspects.append((str(d.date()), round(float(v), 4)))
    return {
        'instrument': name,
        'bars': len(df),
        'start': df.index[0].date().isoformat(),
        'end': df.index[-1].date().isoformat(),
        'ann_vol': round(float(np.log(df['close']).diff().std() * np.sqrt(252)), 4),
        'moves_over_20pct': int(len(big)),
        'split_suspects': len(suspects),
        'suspect_dates': suspects[:4],
        'nonpositive_close': int((df['close'] <= 0).sum()),
    }


def _split_dates(df: pd.DataFrame) -> list:
    r = df['close'].pct_change()
    out = []
    for d, v in r[r.abs() > 0.20].items():
        if any(abs(1 + v - k) < SPLIT_TOL for k in SPLIT_RATIOS):
            out.append(d)
    return out


def trim_after_last_split(df: pd.DataFrame) -> tuple:
    """Start the series the bar after the last unadjusted corporate action.

    Truncating beats discarding: INFY keeps 22 clean years after its 2004
    split, ITC keeps 21 after its 2005 one. The alternative -- feeding a
    -93% print to a band-break rule -- is a fictional signal.
    """
    splits = _split_dates(df)
    if not splits:
        return df, None
    cut = max(splits)
    return df.loc[df.index > cut], cut


def materialise(names=None, max_suspects: int = 0) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name in (names or ALL):
        fname = ALL[name]
        if not os.path.exists(os.path.join(SRC_DIR, fname)):
            rows.append({'instrument': name, 'bars': 0, 'error': 'file missing'})
            continue
        raw = _read(fname)
        df, cut = trim_after_last_split(raw)
        q = quality_scan(name, df)
        q['class'] = ASSET_CLASS[name]
        q['trimmed_at'] = str(cut.date()) if cut is not None else ''
        q['dropped_bars'] = len(raw) - len(df)
        q['usable'] = (q['split_suspects'] <= max_suspects
                       and q['nonpositive_close'] == 0 and len(df) > 750)
        if q['usable']:
            df.to_parquet(os.path.join(OUT_DIR, f'{name}.parquet'))
        rows.append(q)
    return pd.DataFrame(rows)


def load(name: str) -> pd.DataFrame:
    path = os.path.join(OUT_DIR, f'{name}.parquet')
    if not os.path.exists(path):
        materialise([name])
    if not os.path.exists(path):
        raise FileNotFoundError(f'{name} failed the quality scan; see materialise()')
    return pd.read_parquet(path)


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    s = materialise()
    print(f'source: BennyThadikaran/eod2_data @ {source_sha()[:12]}\n')
    print(s[['instrument', 'class', 'bars', 'start', 'end', 'ann_vol',
             'moves_over_20pct', 'split_suspects', 'trimmed_at', 'dropped_bars',
             'usable']].to_string(index=False))
    bad = s[~s['usable'].fillna(False)]
    if len(bad):
        print('\nREJECTED:')
        print(bad[['instrument', 'split_suspects', 'suspect_dates']].to_string(index=False))
