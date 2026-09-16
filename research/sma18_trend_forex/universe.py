"""Multi-asset-class universe for the timeframe x volatility-tier grid.

`vol_tier` is a label only (for grouping in charts/tables) — the grid
still measures each asset's actual annualized volatility from data and
that's what drives the "volatile vs non-volatile" analysis, not this tag.
"""

UNIVERSE = {
    # -- FX majors/crosses (expected low vol) -----------------------------
    "EURUSD":  {"ticker": "EURUSD=X", "asset_class": "FX",       "vol_tier": "low"},
    "GBPUSD":  {"ticker": "GBPUSD=X", "asset_class": "FX",       "vol_tier": "low"},
    "USDJPY":  {"ticker": "USDJPY=X", "asset_class": "FX",       "vol_tier": "low"},
    "AUDUSD":  {"ticker": "AUDUSD=X", "asset_class": "FX",       "vol_tier": "low"},
    "USDCHF":  {"ticker": "USDCHF=X", "asset_class": "FX",       "vol_tier": "low"},
    "EURJPY":  {"ticker": "EURJPY=X", "asset_class": "FX",       "vol_tier": "low"},
    "GBPJPY":  {"ticker": "GBPJPY=X", "asset_class": "FX",       "vol_tier": "low"},
    "USDCAD":  {"ticker": "USDCAD=X", "asset_class": "FX",       "vol_tier": "low"},

    # -- Equity indices (expected medium vol) ------------------------------
    "SPX500":  {"ticker": "^GSPC",    "asset_class": "EQUITY_INDEX", "vol_tier": "medium"},
    "NASDAQ100": {"ticker": "^NDX",   "asset_class": "EQUITY_INDEX", "vol_tier": "medium"},
    "DOW30":   {"ticker": "^DJI",     "asset_class": "EQUITY_INDEX", "vol_tier": "medium"},
    "FTSE100": {"ticker": "^FTSE",    "asset_class": "EQUITY_INDEX", "vol_tier": "medium"},
    "NIKKEI225": {"ticker": "^N225",  "asset_class": "EQUITY_INDEX", "vol_tier": "medium"},

    # -- Commodities (expected medium-high vol) -----------------------------
    "GOLD":    {"ticker": "GC=F",     "asset_class": "COMMODITY", "vol_tier": "medium_high"},
    "CRUDE":   {"ticker": "CL=F",     "asset_class": "COMMODITY", "vol_tier": "medium_high"},
    "SILVER":  {"ticker": "SI=F",     "asset_class": "COMMODITY", "vol_tier": "medium_high"},
    "COPPER":  {"ticker": "HG=F",     "asset_class": "COMMODITY", "vol_tier": "medium_high"},

    # -- Crypto (expected high vol) -----------------------------------------
    "BTCUSD":  {"ticker": "BTC-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "ETHUSD":  {"ticker": "ETH-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "SOLUSD":  {"ticker": "SOL-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
}

TIMEFRAMES = ["15m", "30m", "4h", "1d"]
