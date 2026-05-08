"""
Step 2 — Historical data acquisition.
Downloads daily OHLCV for all MVP instruments via yfinance.
Saves as data/raw/{instrument}_daily.parquet.

When yfinance is unavailable (e.g. sandboxed env), generates calibrated
synthetic data per instrument so downstream steps can proceed. Replace
with real data on a machine with internet access.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

YFINANCE_TICKERS = {
    # Indices
    'NIFTY':       '^NSEI',
    'BANKNIFTY':   '^NSEBANK',
    'FINNIFTY':    'NIFTY_FIN_SERVICE.NS',
    'MIDCPNIFTY':  '^NSMIDCP',
    'VIX':         '^INDIAVIX',
    # Commodities (CME proxies for MCX)
    'GOLD_USD':    'GC=F',
    'SILVER_USD':  'SI=F',
    'CRUDE_USD':   'CL=F',
    'COPPER_USD':  'HG=F',
    # Currencies
    'USDINR':      'INR=X',
    'EURINR':      'EURINR=X',
    'GBPINR':      'GBPINR=X',
    'JPYINR':      'JPYINR=X',
    'EURUSD':      'EURUSD=X',
    'GBPUSD':      'GBPUSD=X',
    'USDJPY':      'JPY=X',
}

START_DATE = '2010-01-01'
END_DATE   = pd.Timestamp.today().strftime('%Y-%m-%d')

# Calibration parameters for synthetic fallback
# (start_price, ann_drift, ann_vol)
SYNTHETIC_CALIBRATION = {
    'NIFTY':       (5200,    0.115, 0.16),
    'BANKNIFTY':   (9000,    0.130, 0.20),
    'FINNIFTY':    (8000,    0.110, 0.18),
    'MIDCPNIFTY':  (6000,    0.090, 0.20),
    'VIX':         (16.0,    0.000, 0.80),  # special: mean-reverting
    'GOLD_USD':    (1100,    0.060, 0.15),
    'SILVER_USD':  (17.0,    0.040, 0.28),
    'CRUDE_USD':   (80.0,    0.000, 0.40),
    'COPPER_USD':  (3.50,    0.025, 0.22),
    'USDINR':      (46.0,    0.040, 0.06),
    'EURINR':      (65.0,    0.020, 0.08),
    'GBPINR':      (75.0,    0.022, 0.09),
    'JPYINR':      (0.50,    0.005, 0.09),
    'EURUSD':      (1.43,   -0.020, 0.08),
    'GBPUSD':      (1.61,   -0.020, 0.08),
    'USDJPY':      (93.0,    0.030, 0.09),
}


def _download_yf(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("empty data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = 'date'
    return df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])


def _generate_synthetic(name: str) -> pd.DataFrame:
    """GBM-based synthetic OHLCV calibrated per instrument."""
    rng = np.random.default_rng(hash(name) % (2**32))
    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    n = len(dates)

    start_price, mu_ann, sigma_ann = SYNTHETIC_CALIBRATION[name]
    dt = 1 / 252
    mu_d = mu_ann * dt
    sigma_d = sigma_ann * np.sqrt(dt)

    if name == 'VIX':
        # Mean-reverting Ornstein-Uhlenbeck around 16, occasional spikes
        prices = np.zeros(n)
        prices[0] = start_price
        for i in range(1, n):
            kappa = 0.05
            spike = 4.0 if rng.random() < 0.005 else 0.0
            prices[i] = max(8.0,
                            prices[i-1] + kappa * (16.0 - prices[i-1])
                            + rng.normal(0, 1.5) + spike)
    else:
        log_returns = rng.normal(mu_d, sigma_d, n)
        prices = start_price * np.exp(np.cumsum(log_returns))

    # Synthesize OHLCV from close
    intraday_vol = sigma_d * 0.5
    high = prices * (1 + np.abs(rng.normal(0, intraday_vol, n)))
    low  = prices * (1 - np.abs(rng.normal(0, intraday_vol, n)))
    open_ = np.concatenate([[prices[0]], prices[:-1] * (1 + rng.normal(0, intraday_vol, n-1))])
    volume = rng.integers(100_000, 10_000_000, n)

    df = pd.DataFrame({
        'open':   open_,
        'high':   np.maximum.reduce([open_, high, prices]),
        'low':    np.minimum.reduce([open_, low,  prices]),
        'close':  prices,
        'volume': volume.astype(np.int64),
    }, index=dates)
    df.index.name = 'date'
    return df


def download_all(force_synthetic: bool = False) -> dict:
    out_dir = 'data/raw'
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for name, ticker in YFINANCE_TICKERS.items():
        df = None
        source = None
        if not force_synthetic:
            try:
                df = _download_yf(ticker)
                source = 'yfinance'
            except Exception as e:
                print(f"  [{name}] yfinance failed ({type(e).__name__}: {e}) — using synthetic")
        if df is None or df.empty:
            df = _generate_synthetic(name)
            source = 'synthetic'

        path = f'{out_dir}/{name}_daily.parquet'
        df.to_parquet(path)
        summary.append({
            'instrument': name,
            'ticker':     ticker,
            'source':     source,
            'start':      df.index[0].date().isoformat(),
            'end':        df.index[-1].date().isoformat(),
            'rows':       len(df),
            'path':       path,
        })

    return summary


def print_summary(summary: list):
    print()
    print(f"{'Instrument':<14} {'Source':<10} {'Start':<12} {'End':<12} {'Rows':>6} {'Flag':<20}")
    print('-' * 80)
    cutoff = pd.Timestamp('2015-01-01').date()
    for row in summary:
        flag = '⚠ INSUFFICIENT_HISTORY' if pd.Timestamp(row['start']).date() > cutoff else ''
        print(f"{row['instrument']:<14} {row['source']:<10} "
              f"{row['start']:<12} {row['end']:<12} {row['rows']:>6} {flag:<20}")
    print()


if __name__ == '__main__':
    print("=== Step 2 — Historical Data Acquisition ===\n")
    s = download_all()
    print_summary(s)

    yf_count        = sum(1 for r in s if r['source'] == 'yfinance')
    synthetic_count = sum(1 for r in s if r['source'] == 'synthetic')
    print(f"yfinance: {yf_count}/{len(s)}   synthetic: {synthetic_count}/{len(s)}")
    if synthetic_count > 0:
        print("\nNOTE: Synthetic data used where yfinance was unreachable.")
        print("Re-run on a machine with internet access (or replace files in data/raw/) for production.")
    print("\n=== Step 2 complete ===")
