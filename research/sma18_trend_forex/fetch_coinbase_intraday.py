"""
Fetches multi-year 15m/30m crypto history from Coinbase Exchange's public
API, to replace yfinance's ~60-day cap on those timeframes with a sample
large enough to actually validate.

yfinance hard-caps 15m/30m at ~60 days for every asset class -- there is no
way around that for futures, because no free source publishes multi-year
intraday futures history (that is a paid-vendor product: Databento, Polygon,
Refinitiv). Crypto is different: exchanges publish their own full candle
history for free, no key required. Coinbase Exchange's public /candles
endpoint was checked directly (see chat) and returns real historical data
back to at least 2021 for major pairs; Kraken and Bybit were also checked
and rejected -- Kraken's public OHLC endpoint silently ignores any 'since'
older than its most recent ~720 candles (LESS history than yfinance, not
more), and Bybit blocks this environment's region outright.

Endpoint: GET /products/{id}/candles?granularity={sec}&start=&end=
  - granularity: 900 (15m) or 1800 (30m), seconds
  - max 300 candles per request -> paginate backward in time
  - returns [time, low, high, open, close, volume], newest first

Run: python fetch_coinbase_intraday.py [years]
Writes: data/15m_cb/{ASSET}.parquet, data/1h_cb/{ASSET}.parquet
(30m and 4h are not fetched -- neither exists as a native Coinbase Exchange
granularity for 30m, and there's no need for 4h either. deep_intraday_
validation.py derives both by resampling: 30m from the deep 15m data, 4h
from the deep 1h data -- the same construction the original study used to
build 4h from yfinance's 1h feed.)
"""

import os, sys, time, warnings
import numpy as np, pandas as pd
import urllib.request, json
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://api.exchange.coinbase.com"
GRAN = {"15m": 900, "1h": 3600}
# Coinbase Exchange has no native 30m candle -- its supported granularities
# are exactly 60/300/900/3600/21600/86400s (1m/5m/15m/1h/6h/1d). Fetching
# granularity=1800 returns HTTP 400 "Unsupported granularity" (confirmed
# directly). 30m is derived by resampling the deep 15m data instead --
# see deep_intraday_validation.py -- which needs no extra API calls at all.
MAX_CANDLES = 300
SLEEP = 0.35   # Coinbase Exchange public rate limit is ~3 req/sec

# Scoped to 6 crypto instruments per user request (was all 14) -- these are
# also the 6 with the deepest market cap/liquidity in the universe, and
# happened to already be the first 6 fetched, so narrowing here wastes no
# prior work.
SELECTED_CRYPTO = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD"]

CB_PRODUCT = {  # UNIVERSE key -> Coinbase Exchange product id
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD", "XRPUSD": "XRP-USD",
    "ADAUSD": "ADA-USD", "DOGEUSD": "DOGE-USD",
}


def _get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def fetch_product(product_id, granularity_sec, start, end):
    """Paginate backward from `end` to `start`, MAX_CANDLES at a time."""
    span = timedelta(seconds=granularity_sec * MAX_CANDLES)
    rows, cursor = [], end
    while cursor > start:
        win_start = max(start, cursor - span)
        url = (f"{BASE}/products/{product_id}/candles?granularity={granularity_sec}"
               f"&start={win_start.isoformat()}&end={cursor.isoformat()}")
        data = _get(url)
        if isinstance(data, dict) and data.get("message"):
            raise RuntimeError(f"{product_id}: {data['message']}")
        if data:
            rows.extend(data)
        cursor = win_start
        time.sleep(SLEEP)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_localize(None)
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def main(years=3):
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=int(years * 365.25))

    for tf, gran in GRAN.items():
        out_dir = os.path.join(HERE, "data", f"{tf}_cb")
        os.makedirs(out_dir, exist_ok=True)
        for name in SELECTED_CRYPTO:
            product = CB_PRODUCT.get(name)
            if not product:
                continue
            path = os.path.join(out_dir, f"{name}.parquet")
            if os.path.exists(path):
                print(f"  {tf} {name:10s} cached", flush=True)
                continue
            t0 = time.time()
            try:
                df = fetch_product(product, gran, start, end)
            except Exception as e:
                print(f"  {tf} {name:10s} FAILED: {e}", flush=True)
                continue
            if df is None or len(df) < 500:
                print(f"  {tf} {name:10s} insufficient ({0 if df is None else len(df)} bars)", flush=True)
                continue
            df.to_parquet(path)
            span_days = (df.index[-1] - df.index[0]).days
            print(f"  {tf} {name:10s} {len(df):>7,} bars  {span_days:>5} days  "
                  f"{df.index[0].date()} -> {df.index[-1].date()}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 3)
