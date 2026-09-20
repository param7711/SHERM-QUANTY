"""
Liquidity vs. result on the INTRADAY timeframes only (15m, 30m, 1h).

The daily-timeframe version of this test lives in liquidity_test.py. This one
re-runs the same question against the short-timeframe backtests, where the
cost-and-liquidity story should bite hardest: a rule that trades many times a
day is far more exposed to spread and depth than one holding for a week.

Three things make this a WEAKER test than the daily one, and they should be
read before the correlations:

  * 15m and 30m carry ~60 calendar days of data and 1h carries ~730 (yfinance
    hard caps). A Sharpe computed on 60 days is extremely noisy, and none of
    these timeframes has a Markov p-value behind it -- there is not enough
    history to fit a null model to. So this correlates liquidity against a
    NOISY POINT ESTIMATE, not against an established edge.

  * Liquidity is measured from DAILY volume over three years. It is a
    reasonable stand-in for how deep an instrument is, but it is not
    intraday depth, and it says nothing about spread at 15m.

  * Six commodities are excluded because Yahoo's continuous-futures volume is
    unusable for them (PL=F prints a median of 2 contracts/day). That leaves
    10 commodities and 14 crypto -- small samples where one instrument moves
    the correlation materially.

Run: python intraday_liquidity.py
Outputs: outputs/intraday_liquidity.json
"""

import json, math, os, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid import run_one
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
TIMEFRAMES = ["15m", "30m", "1h"]


def _sanitize(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    return o


def corr(sub, a, b):
    s = sub[[a, b]].dropna()
    if len(s) < 5:
        return None
    return {"pearson": round(float(s[a].corr(s[b])), 3),
            "spearman": round(float(s[a].rank().corr(s[b].rank())), 3),
            "n": int(len(s))}


def main():
    liq = pd.DataFrame(json.load(open(os.path.join(OUT_DIR, "liquidity_test.json")))["per_asset"])
    rows = []
    for tf in TIMEFRAMES:
        for name, meta in UNIVERSE.items():
            row, _ = run_one(name, meta, tf, direction="long")
            if row.get("error"):
                print(f"  {tf:4s} {name:10s} skipped: {row['error']}", flush=True)
                continue
            rows.append({"asset": name, "asset_class": meta["asset_class"], "timeframe": tf,
                         "sharpe": row["sharpe"], "cagr_pct": row["cagr_pct"],
                         "n_trades": row["n_trades"], "n_bars": row["n_bars"],
                         "win_rate": row["win_rate"], "ann_vol": row["ann_vol"]})
        print(f"  {tf} done ({sum(1 for r in rows if r['timeframe']==tf)} instruments)", flush=True)

    res = pd.DataFrame(rows).merge(
        liq[["asset", "log10_dollar_vol", "median_dollar_vol", "volume_unreliable"]],
        on="asset", how="left")
    rel = res[~res["volume_unreliable"].fillna(True)].copy()

    out_corr, tiers = [], []
    for tf in TIMEFRAMES:
        sub = rel[rel.timeframe == tf]
        entry = {"timeframe": tf,
                 "pooled": corr(sub, "log10_dollar_vol", "sharpe"),
                 "by_class": {}}
        for cls, cs in sub.groupby("asset_class"):
            entry["by_class"][cls] = corr(cs, "log10_dollar_vol", "sharpe")
            med = cs["log10_dollar_vol"].median()
            for label, part in (("more liquid half", cs[cs.log10_dollar_vol >= med]),
                                ("less liquid half", cs[cs.log10_dollar_vol < med])):
                tiers.append({"timeframe": tf, "asset_class": cls, "tier": label,
                              "n": int(len(part)),
                              "mean_sharpe": round(float(part["sharpe"].mean()), 3),
                              "median_sharpe": round(float(part["sharpe"].median()), 3),
                              "mean_trades": int(part["n_trades"].mean()),
                              "assets": part.sort_values("log10_dollar_vol", ascending=False)["asset"].tolist()})
        out_corr.append(entry)

    # Daily reference, for a like-for-like comparison. liquidity_test.py
    # already joined the daily Markov result onto liq, so no re-merge needed
    # (merging mk's real_sharpe again would collide into real_sharpe_x/_y).
    d = liq[~liq.volume_unreliable].copy()
    daily = {"timeframe": "1d", "pooled": corr(d, "log10_dollar_vol", "real_sharpe"),
             "by_class": {cls: corr(cs, "log10_dollar_vol", "real_sharpe")
                          for cls, cs in d.groupby("asset_class")}}

    payload = {"timeframes": TIMEFRAMES, "correlations": out_corr, "daily_reference": daily,
               "tiers": tiers, "per_row": rel.round(4).to_dict(orient="records"),
               "n_excluded_bad_volume": int(res["volume_unreliable"].fillna(True).sum() / len(TIMEFRAMES))}
    with open(os.path.join(OUT_DIR, "intraday_liquidity.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)

    print("\n=== Liquidity vs Sharpe, by timeframe (rank correlation) ===")
    print(f"{'TF':>5}  {'pooled':>16}  {'COMMODITY':>16}  {'CRYPTO':>16}")
    for e in out_corr + [daily]:
        def fmt(c):
            return "n/a" if not c else f"{c['spearman']:+.2f} (n={c['n']})"
        print(f"{e['timeframe']:>5}  {fmt(e['pooled']):>16}  "
              f"{fmt(e['by_class'].get('COMMODITY')):>16}  {fmt(e['by_class'].get('CRYPTO')):>16}")

    print("\n=== More-liquid vs less-liquid half, mean Sharpe ===")
    for tf in TIMEFRAMES:
        for cls in ("COMMODITY", "CRYPTO"):
            t = [x for x in tiers if x["timeframe"] == tf and x["asset_class"] == cls]
            if len(t) == 2:
                hi = [x for x in t if x["tier"] == "more liquid half"][0]
                lo = [x for x in t if x["tier"] == "less liquid half"][0]
                print(f"  {tf:4s} {cls:9s}  more liquid {hi['mean_sharpe']:+7.2f} (n={hi['n']})   "
                      f"less liquid {lo['mean_sharpe']:+7.2f} (n={lo['n']})   "
                      f"gap {hi['mean_sharpe'] - lo['mean_sharpe']:+.2f}")
    print(f"\nwrote {os.path.join(OUT_DIR, 'intraday_liquidity.json')}")


if __name__ == "__main__":
    main()
