"""
Does liquidity explain which instruments the SMA18 rule works on?

The hypothesis under test: the rule pays on deeply liquid instruments and
fails on thin ones. That is a claim about a *cross-sectional* relationship
between liquidity and edge, so it needs a liquidity number per instrument
and a correlation against the result, not an eyeball over the winners.

Three proxies are used, because no single one is trustworthy here:

1. MEDIAN DAILY DOLLAR VOLUME. The direct measure. Futures volume is quoted
   in contracts, and contract sizes differ by orders of magnitude (gold is
   100 troy oz, corn is 5,000 bushels), so raw volume is meaningless across
   instruments -- every contract is converted to notional USD using its real
   multiplier and price units before comparison. yfinance quotes *-USD crypto
   volume already in USD, so no multiplier applies there.

2. ZERO-RETURN DAY FRACTION. A classic illiquidity proxy (Lesmond/Ogden/
   Trzcinka) that needs no volume data and no units at all: a thin market
   prints unchanged closes more often. This is the cross-check on (1),
   because it cannot be corrupted by a wrong multiplier.

3. AMIHUD ILLIQUIDITY. mean(|return| / dollar volume) -- how much the price
   moves per dollar traded. Higher = thinner.

The decisive test is WITHIN asset class. Crypto is both less liquid than
gold/crude AND much better-performing here, so a pooled correlation across
all 30 instruments would be driven entirely by the class split and would
report whatever the class difference says, not what liquidity says.
"""

import json, math, os, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
VOL_DIR = os.path.join(HERE, "data", "1d_volume")
os.makedirs(VOL_DIR, exist_ok=True)

# Contract multiplier and price-unit divisor per futures ticker, so that
# close * volume * mult / div is notional USD traded.
#   div=100 for the contracts Yahoo quotes in US cents (grains, softs).
CONTRACT = {
    "GC=F": (100, 1), "SI=F": (5000, 1), "PL=F": (50, 1), "PA=F": (100, 1),
    "HG=F": (25000, 1), "CL=F": (1000, 1), "BZ=F": (1000, 1), "NG=F": (10000, 1),
    "HO=F": (42000, 1), "RB=F": (42000, 1),
    "ZC=F": (5000, 100), "ZW=F": (5000, 100), "ZS=F": (5000, 100),
    "SB=F": (112000, 100), "KC=F": (37500, 100), "CT=F": (50000, 100),
}


def _sanitize(o):
    """json.dump writes bare NaN/Infinity, which are not valid JSON and fail
    JSON.parse in the browser. Crypto zero-return fractions are all exactly 0,
    so their correlation is genuinely undefined -- that must serialize as null."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    return o


def fetch(name, ticker):
    path = os.path.join(VOL_DIR, f"{name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    import yfinance as yf
    df = yf.download(ticker, start="1990-01-01", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["close", "volume"]].dropna()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.to_parquet(path)
    return df


def main():
    rows = []
    for name, meta in UNIVERSE.items():
        try:
            df = fetch(name, meta["ticker"])
        except Exception as e:
            print(f"  {name:10s} fetch failed: {e}", flush=True)
            continue
        if df is None or len(df) < 500:
            print(f"  {name:10s} insufficient data", flush=True)
            continue

        close, vol = df["close"].values, df["volume"].values
        is_futures = meta["ticker"] in CONTRACT
        if is_futures:
            mult, div = CONTRACT[meta["ticker"]]
            dollar_vol = close * vol * mult / div      # contracts -> notional USD
        else:
            # yfinance reports *-USD crypto volume in USD ALREADY. Multiplying
            # by close double-counts it (it put BTC at $3.5 quadrillion/day).
            dollar_vol = vol.astype(float)

        # Use the last 3 years only: liquidity is a present-day property, and
        # early history for both futures and crypto is not comparable.
        tail = min(len(df), 756)
        dv = dollar_vol[-tail:]
        dv = dv[dv > 0]
        if len(dv) < 100:
            print(f"  {name:10s} no usable volume", flush=True)
            continue

        # Yahoo's continuous-futures volume is unusable for several contracts:
        # PL=F prints a median of 2 contracts/day with 38% zero-volume days,
        # SI=F prints 87, against real exchange volumes in the tens of
        # thousands. Those instruments are flagged and excluded from the
        # correlations rather than silently ranked as illiquid -- PLATINUM and
        # PALLADIUM are exactly the assets that failed the Markov test, so
        # letting broken volume data stand would manufacture a confirmation
        # of the very hypothesis being tested.
        zero_vol_frac = float(np.mean(vol[-tail:] == 0))
        median_contracts = float(np.median(vol[-tail:]))
        unreliable = bool(is_futures and (median_contracts < 5000 or zero_vol_frac > 0.05))

        ret = np.diff(close) / close[:-1]
        zero_frac = float(np.mean(np.abs(ret[-tail:]) < 1e-9))
        amihud = float(np.mean(np.abs(ret[-tail + 1:]) / dv[-(tail - 1):]) * 1e12) \
            if len(dv) >= tail - 1 else float("nan")

        rows.append({
            "asset": name, "asset_class": meta["asset_class"],
            "median_dollar_vol": float(np.median(dv)),
            "log10_dollar_vol": float(np.log10(np.median(dv))),
            "zero_return_frac": zero_frac,
            "median_contracts": median_contracts,
            "zero_vol_frac": zero_vol_frac,
            "volume_unreliable": unreliable,
            "amihud_x1e12": amihud,
            "n_days": int(len(df)),
        })
        print(f"  {name:10s} ${np.median(dv)/1e9:9.2f}B/day  "
              f"raw={median_contracts:>14,.0f}  zero-vol {zero_vol_frac:5.1%}"
              f"{'   <-- UNRELIABLE, excluded' if unreliable else ''}", flush=True)

    liq = pd.DataFrame(rows)

    # --- join against the results the study already produced ----------------
    mk = pd.DataFrame(json.load(open(os.path.join(OUT_DIR, "markov_validation.json")))["flat"])
    df = liq.merge(mk[["asset", "real_sharpe", "markov_p", "shuffle_p", "real_n_trades"]],
                   on="asset", how="inner")
    df["passes"] = (df["markov_p"] < 0.05).astype(int)

    def corr(sub, a, b):
        s = sub[[a, b]].dropna()
        if len(s) < 5:
            return None
        # rank-then-Pearson == Spearman; scipy is not installed here.
        return {"pearson": round(float(s[a].corr(s[b])), 3),
                "spearman": round(float(s[a].rank().corr(s[b].rank())), 3),
                "n": int(len(s))}

    n_dropped = int(df["volume_unreliable"].sum())
    dropped = df[df.volume_unreliable]["asset"].tolist()
    rel = df[~df.volume_unreliable].copy()

    results = {"pooled": {}, "by_class": {},
               "excluded_for_bad_volume": dropped,
               "n_used": int(len(rel))}
    for measure in ("log10_dollar_vol", "zero_return_frac", "amihud_x1e12"):
        results["pooled"][measure] = {
            "vs_sharpe": corr(rel, measure, "real_sharpe"),
            "vs_markov_p": corr(rel, measure, "markov_p"),
        }
    for cls, sub in rel.groupby("asset_class"):
        results["by_class"][cls] = {
            m: {"vs_sharpe": corr(sub, m, "real_sharpe"),
                "vs_markov_p": corr(sub, m, "markov_p")}
            for m in ("log10_dollar_vol", "zero_return_frac", "amihud_x1e12")
        }
        results["by_class"][cls]["n_pass"] = int(sub["passes"].sum())
        results["by_class"][cls]["n"] = int(len(sub))

    # Split each class at its own median liquidity and compare halves.
    tiers = []
    for cls, sub in rel.groupby("asset_class"):
        med = sub["log10_dollar_vol"].median()
        for label, part in (("top half (more liquid)", sub[sub.log10_dollar_vol >= med]),
                            ("bottom half (less liquid)", sub[sub.log10_dollar_vol < med])):
            tiers.append({
                "asset_class": cls, "tier": label, "n": int(len(part)),
                "mean_sharpe": round(float(part["real_sharpe"].mean()), 3),
                "median_markov_p": round(float(part["markov_p"].median()), 3),
                "n_pass": int(part["passes"].sum()),
                "assets": part.sort_values("log10_dollar_vol", ascending=False)["asset"].tolist(),
            })

    payload = {
        "per_asset": df.round(4).to_dict(orient="records"),
        "correlations": results,
        "tiers": tiers,
        "excluded_for_bad_volume": dropped,
        "note": ("Futures dollar volume uses real contract multipliers and cent-quoted "
                 "price units; crypto volume from yfinance is already USD. Liquidity is "
                 "measured over the most recent ~3 years."),
    }
    with open(os.path.join(OUT_DIR, "liquidity_test.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)

    print(f"\nExcluded for unusable Yahoo volume data: {dropped or 'none'}")
    print("\n=== Liquidity ranking (median notional USD traded per day) ===")
    for _, r in rel.sort_values("median_dollar_vol", ascending=False).iterrows():
        flag = "PASS" if r["passes"] else "fail"
        print(f"  {r['asset']:10s} {r['asset_class']:9s} ${r['median_dollar_vol']/1e9:9.2f}B  "
              f"Sharpe {r['real_sharpe']:+.2f}  p={r['markov_p']:.3f}  {flag}")

    print("\n=== Does liquidity predict the edge? ===")
    print("POOLED across all 30 (confounded by the class split - see below):")
    for m, v in results["pooled"].items():
        print(f"  {m:22s} vs Sharpe: {v['vs_sharpe']}")
    print("\nWITHIN each asset class (this is the test that matters):")
    for cls, v in results["by_class"].items():
        print(f"  {cls} (n={v['n']}, {v['n_pass']} pass):")
        for m in ("log10_dollar_vol", "zero_return_frac", "amihud_x1e12"):
            print(f"    {m:22s} vs Sharpe: {v[m]['vs_sharpe']}   vs p-value: {v[m]['vs_markov_p']}")

    print("\n=== Liquidity halves within each class ===")
    for t in tiers:
        print(f"  {t['asset_class']:9s} {t['tier']:26s} n={t['n']:2d}  "
              f"mean Sharpe {t['mean_sharpe']:+.2f}  median p={t['median_markov_p']:.3f}  "
              f"{t['n_pass']}/{t['n']} pass")


if __name__ == "__main__":
    main()
