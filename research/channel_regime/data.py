"""
Real-market data loader for the regression-channel regime hypothesis test.

IMPORTANT — data provenance.
This sandbox's egress policy blocks every market-data host (Yahoo, Stooq,
NSE archives, AlphaVantage, Tiingo all return 403 at the proxy), so
`data/download_data.py` would silently fall back to its synthetic GBM
generator. Synthetic GBM has no serial dependence by construction, so
BOTH legs of the hypothesis are guaranteed to lose money on it — testing
an edge there is meaningless.

Instead this module assembles genuinely observed price history that ships
*inside* PyPI packages (no network fetch at load time):

  SP500    S&P 500 daily OHLCV      1999-01-04 .. 2018-12-31   (arch)
  NASDAQ   Nasdaq Comp daily OHLCV  1999-01-04 .. 2018-12-31   (arch)
  WTI      WTI crude daily close    1986-01-02 .. 2019-01-03   (arch/FRED)
  GOOG     Google daily OHLCV       2004-08-19 .. 2013-03-01   (backtesting.py)
  EURUSD   EUR/USD hourly OHLCV     2017-04-19 .. 2018-02-07   (backtesting.py)

That is five real series across index / commodity / single-stock / FX and
two timeframes — enough to ask whether the hypothesis survives outside the
series it was eyeballed on. It is NOT NIFTY/BANKNIFTY: the conclusions
transfer as "does this class of rule work on real price series", and the
final read must be re-run on Indian index data before any capital moves.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, 'data', 'research')

# Bars per year, per instrument, used for annualising.
BARS_PER_YEAR = {
    'SP500':  252,
    'NASDAQ': 252,
    'WTI':    252,
    'GOOG':   252,
    'EURUSD': 252 * 24,
}

ASSET_CLASS = {
    'SP500':  'equity_index',
    'NASDAQ': 'equity_index',
    'WTI':    'commodity',
    'GOOG':   'single_stock',
    'EURUSD': 'fx_intraday',
}


def _from_arch(module_name: str) -> pd.DataFrame:
    import importlib
    mod = importlib.import_module(f'arch.data.{module_name}')
    return mod.load()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).lower().replace(' ', '_') for c in df.columns})
    if 'adj_close' in df.columns:
        # Split/dividend adjusted series is the one to trade on.
        ratio = df['adj_close'] / df['close']
        for col in ('open', 'high', 'low'):
            if col in df.columns:
                df[col] = df[col] * ratio
        df['close'] = df['adj_close']
        df = df.drop(columns=['adj_close'])
    keep = [c for c in ('open', 'high', 'low', 'close', 'volume') if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    return df.sort_index().dropna(subset=['close'])


def load_all() -> dict:
    """Return {name: OHLC(V) dataframe} of real observed prices."""
    out = {}

    out['SP500'] = _normalise(_from_arch('sp500'))
    out['NASDAQ'] = _normalise(_from_arch('nasdaq'))

    wti = _from_arch('wti')
    wti.columns = ['close']
    out['WTI'] = _normalise(wti)

    from backtesting.test import GOOG, EURUSD
    out['GOOG'] = _normalise(GOOG.copy())
    out['EURUSD'] = _normalise(EURUSD.copy())

    return out


def materialise(out_dir: str = OUT_DIR) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, df in load_all().items():
        path = os.path.join(out_dir, f'{name}.parquet')
        df.to_parquet(path)
        ann = np.log(df['close']).diff().std() * np.sqrt(BARS_PER_YEAR[name])
        rows.append({
            'instrument': name,
            'class':      ASSET_CLASS[name],
            'bars':       len(df),
            'start':      df.index[0].date().isoformat(),
            'end':        df.index[-1].date().isoformat(),
            'ann_vol':    round(float(ann), 4),
            'has_ohlc':   'open' in df.columns,
            'path':       path,
        })
    return pd.DataFrame(rows)


def load(name: str, out_dir: str = OUT_DIR) -> pd.DataFrame:
    path = os.path.join(out_dir, f'{name}.parquet')
    if not os.path.exists(path):
        materialise(out_dir)
    return pd.read_parquet(path)


if __name__ == '__main__':
    summary = materialise()
    print(summary.to_string(index=False))
