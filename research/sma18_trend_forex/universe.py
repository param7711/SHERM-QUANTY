"""Commodity + crypto universe for the SMA18 trend study.

FX and equity indices were dropped after Part 2/3 showed this rule behaves
as a volatility/momentum harvester: FX was the worst class in every cut and
equity indices were middling, so the study now concentrates on the two
classes where the rule actually showed something — and tests those far
harder instead of testing more classes shallowly.

`vol_tier` is a label only (for grouping in charts/tables) — the analysis
measures each instrument's actual annualized volatility from data.
"""

UNIVERSE = {
    # -- Commodities: metals, energy, softs, grains --------------------------
    "GOLD":      {"ticker": "GC=F", "asset_class": "COMMODITY", "vol_tier": "medium"},
    "SILVER":    {"ticker": "SI=F", "asset_class": "COMMODITY", "vol_tier": "medium_high"},
    "PLATINUM":  {"ticker": "PL=F", "asset_class": "COMMODITY", "vol_tier": "medium_high"},
    "PALLADIUM": {"ticker": "PA=F", "asset_class": "COMMODITY", "vol_tier": "high"},
    "COPPER":    {"ticker": "HG=F", "asset_class": "COMMODITY", "vol_tier": "medium"},
    "CRUDE":     {"ticker": "CL=F", "asset_class": "COMMODITY", "vol_tier": "high"},
    "BRENT":     {"ticker": "BZ=F", "asset_class": "COMMODITY", "vol_tier": "high"},
    "NATGAS":    {"ticker": "NG=F", "asset_class": "COMMODITY", "vol_tier": "high"},
    "HEATOIL":   {"ticker": "HO=F", "asset_class": "COMMODITY", "vol_tier": "high"},
    "GASOLINE":  {"ticker": "RB=F", "asset_class": "COMMODITY", "vol_tier": "high"},
    "CORN":      {"ticker": "ZC=F", "asset_class": "COMMODITY", "vol_tier": "medium"},
    "WHEAT":     {"ticker": "ZW=F", "asset_class": "COMMODITY", "vol_tier": "medium_high"},
    "SOYBEAN":   {"ticker": "ZS=F", "asset_class": "COMMODITY", "vol_tier": "medium"},
    "SUGAR":     {"ticker": "SB=F", "asset_class": "COMMODITY", "vol_tier": "medium_high"},
    "COFFEE":    {"ticker": "KC=F", "asset_class": "COMMODITY", "vol_tier": "high"},
    "COTTON":    {"ticker": "CT=F", "asset_class": "COMMODITY", "vol_tier": "medium_high"},

    # -- Crypto --------------------------------------------------------------
    "BTCUSD":    {"ticker": "BTC-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "ETHUSD":    {"ticker": "ETH-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "SOLUSD":    {"ticker": "SOL-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "XRPUSD":    {"ticker": "XRP-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "ADAUSD":    {"ticker": "ADA-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "DOGEUSD":   {"ticker": "DOGE-USD", "asset_class": "CRYPTO", "vol_tier": "high"},
    "LTCUSD":    {"ticker": "LTC-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "BCHUSD":    {"ticker": "BCH-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "LINKUSD":   {"ticker": "LINK-USD", "asset_class": "CRYPTO", "vol_tier": "high"},
    "AVAXUSD":   {"ticker": "AVAX-USD", "asset_class": "CRYPTO", "vol_tier": "high"},
    "DOTUSD":    {"ticker": "DOT-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "XLMUSD":    {"ticker": "XLM-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "ETCUSD":    {"ticker": "ETC-USD",  "asset_class": "CRYPTO", "vol_tier": "high"},
    "ATOMUSD":   {"ticker": "ATOM-USD", "asset_class": "CRYPTO", "vol_tier": "high"},
}

TIMEFRAMES = ["15m", "30m", "4h", "1d"]
