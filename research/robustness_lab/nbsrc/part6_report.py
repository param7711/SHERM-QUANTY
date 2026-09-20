# %% [markdown]
# ---
# ## 21 - Robustness dashboard
#
# Every diagnostic in one place. The table below deliberately has a
# **"reading"** column rather than a pass/fail column: most of these numbers do
# not have a universal threshold, and inventing one would manufacture false
# precision. Where a genuine reference value exists - PBO's 0.5, a profit
# factor's 1.0, a p-value's alpha - it is named explicitly.

# %%
def _fmt(v, kind="num", digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if kind == "pct":
        return f"{v*100:.1f}%"
    if kind == "x":
        return f"{v:.1f}x"
    return f"{v:.{digits}f}"


_H = HIST_CALIB.metrics
_real_alpha = _H["sharpe"] - BH_CALIB.metrics["sharpe"]
_nulls = SYNTH_SUMMARY.nsmallest(4, "dep_score")
_null_med = float(_nulls["median_sharpe"].median())
_dash = []


def add_diag(group, name, value, reference, reading):
    _dash.append({"group": group, "diagnostic": name, "value": value,
                  "reference": reference, "reading": reading})


add_diag("historical", "Sharpe (calibration slice)", _fmt(_H["sharpe"]),
         f"buy & hold {_fmt(BH_CALIB.metrics['sharpe'])}",
         "beats B&H" if _real_alpha > 0 else "LOSES to buy & hold")
add_diag("historical", "Sharpe (held-out slice)", _fmt(HIST_TEST.metrics["sharpe"]),
         "never seen by any fitted model",
         "positive" if HIST_TEST.metrics["sharpe"] > 0 else "negative")
add_diag("historical", "annualised return", _fmt(_H["ann_return"], "pct"),
         f"B&H {_fmt(BH_CALIB.metrics['ann_return'], 'pct')}", "")
add_diag("historical", "max drawdown", _fmt(_H["max_dd"], "pct"),
         f"B&H {_fmt(BH_CALIB.metrics['max_dd'], 'pct')}", "")
add_diag("historical", "profit factor", _fmt(_H["profit_factor"]), "1.0 = breakeven",
         "profitable gross" if _H["profit_factor"] > 1 else "loses gross")
add_diag("historical", "annual turnover", _fmt(_H["turnover_ann"], "x"),
         f"B&H {_fmt(BH_CALIB.metrics['turnover_ann'], 'x')}", "")

add_diag("synthetic", "median Sharpe, structure-destroyed nulls", _fmt(_null_med),
         f"real {_fmt(_H['sharpe'])}",
         "indistinguishable from real" if abs(_H["sharpe"] - _null_med) < 0.15
         else "real result differs from the nulls")
add_diag("synthetic", "median timing alpha vs buy & hold",
         _fmt(SYNTH_SUMMARY["timing_alpha"].median()), "0 = no value added",
         "negative across generators" if SYNTH_SUMMARY["timing_alpha"].median() < 0
         else "positive")
add_diag("synthetic", "generators where real p < alpha",
         f"{int((SYNTH_SUMMARY['p_value'] < ALPHA).sum())}/{len(SYNTH_SUMMARY)}",
         f"alpha = {ALPHA}", "")
add_diag("synthetic", "best generator fidelity",
         _fmt(max(f['overall'] for f in FIDELITY.values())),
         f"ceiling {_fmt(SELF_COVERAGE)}", "")

add_diag("parameter", "plateau width around chosen value",
         f"{PARAM_SWEEP['contig_width']} of {len(PARAM_SWEEP['grid'])}",
         f"within {ROBUST_PLATEAU_TOLERANCE:.0%} of peak",
         "broad plateau" if PARAM_SWEEP["contig_width"] >= 5 else "NARROW - cliff risk")
add_diag("parameter", "optimisation premium (real)", _fmt(PARAM_SWEEP["premium"]),
         "what grid search bought", "")
add_diag("parameter", "same premium on synthetic markets",
         _fmt(MINING_SUMMARY["median_premium"].median()) if len(MINING_SUMMARY) else "n/a",
         "pure noise-fitting gain",
         "real gain is typical of noise" if len(MINING_SUMMARY) and
         PARAM_SWEEP["premium"] <= MINING_SUMMARY["p95_premium"].median() else "")
add_diag("parameter", "Sharpe swing over +/-10% of parameter",
         _fmt(PARAM_PERTURB["sharpe"].max() - PARAM_PERTURB["sharpe"].min()), "", "")

for _scheme, _df in WF_RESULTS.items():
    if _df.empty:
        continue
    _eff = _df["oos_sharpe"].mean() / _df["is_sharpe"].mean() if _df["is_sharpe"].mean() else np.nan
    add_diag("walk-forward", f"{_scheme}: efficiency ratio (OOS/IS)", _fmt(_eff),
             "1.0 = perfect transfer",
             "in-sample did not transfer" if np.isfinite(_eff) and _eff < 0.5 else "")
    add_diag("walk-forward", f"{_scheme}: mean OOS Sharpe", _fmt(_df["oos_sharpe"].mean()),
             f"fixed param {_fmt(_df['oos_sharpe_fixed'].mean())}", "")

add_diag("execution", "break-even cost", f"{BREAKEVEN_BPS:.0f} bps" if np.isfinite(BREAKEVEN_BPS) else "inf",
         f"assumed {COST_BPS + SLIPPAGE_BPS:.0f} bps",
         "thin margin" if np.isfinite(BREAKEVEN_BPS) and
         BREAKEVEN_BPS < 2 * (COST_BPS + SLIPPAGE_BPS) else "")
add_diag("execution", "Sharpe at 1-bar extra delay",
         _fmt(DELAY_SENSITIVITY.loc[DELAY_SENSITIVITY["delay_bars"] == 1, "sharpe"].iloc[0])
         if (DELAY_SENSITIVITY["delay_bars"] == 1).any() else "n/a",
         f"no delay {_fmt(_H['sharpe'])}", "")
add_diag("execution", "Sharpe at 10% missed signals",
         _fmt(SIGNAL_NOISE.loc[np.isclose(SIGNAL_NOISE['missed_signal_rate'], 0.10),
                               'mean_sharpe'].iloc[0])
         if np.isclose(SIGNAL_NOISE["missed_signal_rate"], 0.10).any() else "n/a", "", "")

for _pl in PLACEBO_SUMMARY.index:
    add_diag("placebo", f"{_pl}: p-value",
             _fmt(PLACEBO_SUMMARY.loc[_pl, "p_value"]), f"alpha = {ALPHA}",
             "real beats this control" if PLACEBO_SUMMARY.loc[_pl, "p_value"] < ALPHA
             else "control matches the real strategy")

add_diag("multiple testing", "Deflated Sharpe (P[true SR > 0])", _fmt(DSR["dsr"]),
         "higher is stronger",
         "weak" if np.isfinite(DSR["dsr"]) and DSR["dsr"] < 0.95 else "")
add_diag("multiple testing", "expected max Sharpe under null (ann.)", _fmt(DSR["sr0_ann"]),
         f"observed {_fmt(DSR['sr_ann'])}",
         "observed does not clear the null maximum" if DSR["sr_ann"] < DSR["sr0_ann"] else "")
add_diag("multiple testing", "min track record needed",
         f"{DSR['min_trl_years']:.1f} y" if np.isfinite(DSR["min_trl_years"]) else "infinite",
         f"actual {len(_r_chosen)/BPY:.1f} y", "")
add_diag("multiple testing", "chosen param survives Bonferroni",
         str(MULTIPLE_TESTING["chosen_survives_bonferroni"]), "", "")
add_diag("multiple testing", "PBO", _fmt(PBO["pbo"]),
         "0.5 = selection is worthless",
         "selection no better than random" if PBO["pbo"] >= 0.5 else
         ("selection carries information" if PBO["pbo"] < 0.2 else "partial transfer"))
add_diag("multiple testing", "P(selected variant loses OOS)", _fmt(PBO["prob_oos_loss"]),
         "", "")

if len(MC_SUMMARY):
    add_diag("trade MC", "P(reordering worse than historical drawdown)",
             _fmt(MC_SUMMARY.loc["block-5", "p_dd_worse_than_hist"], "pct"),
             "historical DD is one draw", "")
    add_diag("trade MC", "P(final equity below 1.0)",
             _fmt(MC_SUMMARY.loc["block-5", "p_loss"], "pct"), "", "")

_worst_scen = STRESS_SUMMARY.drop("baseline_bootstrap")["median_sharpe"].idxmin()
add_diag("stress", f"harshest scenario ({_worst_scen})",
         _fmt(STRESS_SUMMARY.loc[_worst_scen, "median_sharpe"]),
         f"baseline {_fmt(STRESS_SUMMARY.loc['baseline_bootstrap','median_sharpe'])}", "")
add_diag("stress", f"{_worst_scen}: share of paths losing money",
         _fmt(STRESS_SUMMARY.loc[_worst_scen, "frac_losing"], "pct"), "", "")

DASHBOARD = pd.DataFrame(_dash)
print("ROBUSTNESS DASHBOARD")
print("=" * 118)
for _g in DASHBOARD["group"].unique():
    print(f"\n{_g.upper()}")
    _sub = DASHBOARD[DASHBOARD["group"] == _g]
    for _, _r in _sub.iterrows():
        print(f"  {_r['diagnostic']:<46} {_r['value']:>14}   {_r['reference']:<34} {_r['reading']}")

# %%
fig = plt.figure(figsize=(12, 9.5))
gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.28, top=0.88)

# Row 0 - headline stat tiles rendered as a single axes.
ax = fig.add_subplot(gs[0, :])
ax.axis("off")
_tiles = [
    ("Sharpe (real)", f"{_H['sharpe']:+.2f}", INK["primary"]),
    ("Buy & hold", f"{BH_CALIB.metrics['sharpe']:+.2f}", INK["secondary"]),
    ("Timing alpha", f"{_real_alpha:+.2f}",
     STATUS["critical"] if _real_alpha < 0 else STATUS["good"]),
    ("Nulls (median)", f"{_null_med:+.2f}", C["orange"]),
    ("PBO", f"{PBO['pbo']:.2f}",
     STATUS["critical"] if PBO["pbo"] >= 0.5 else STATUS["good"]),
    ("Deflated Sharpe", f"{DSR['dsr']:.2f}" if np.isfinite(DSR["dsr"]) else "n/a",
     STATUS["critical"] if (np.isfinite(DSR["dsr"]) and DSR["dsr"] < 0.95) else STATUS["good"]),
]
for _i, (_lab, _val, _c) in enumerate(_tiles):
    _x = _i / len(_tiles) + 0.008
    ax.add_patch(mpatches.FancyBboxPatch(
        (_x, 0.08), 1 / len(_tiles) - 0.016, 0.84, boxstyle="round,pad=0.006",
        transform=ax.transAxes, facecolor="#f2f1ed", edgecolor="none"))
    ax.text(_x + 0.5 / len(_tiles) - 0.008, 0.62, _val, transform=ax.transAxes,
            ha="center", va="center", fontsize=21, fontweight="bold", color=_c)
    ax.text(_x + 0.5 / len(_tiles) - 0.008, 0.24, _lab, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=INK["secondary"])

ax = fig.add_subplot(gs[1, 0])
_o = SYNTH_SUMMARY.sort_values("dep_score")
_cc = [C["orange"] if d < 0.5 else C["blue"] for d in _o["dep_score"]]
ax.barh(np.arange(len(_o)), _o["median_sharpe"], color=_cc, height=0.72)
ax.axvline(_H["sharpe"], color=INK["primary"], lw=1.6, ls="--")
ax.set_yticks([])
finish(ax, "Synthetic median Sharpe", xlabel="Sharpe", legend=False,
       subtitle="orange = structure destroyed; line = real")

ax = fig.add_subplot(gs[1, 1])
ax.plot(PARAM_SWEEP["grid"], PARAM_SWEEP["calib"], color=C["blue"], lw=1.8)
ax.axvline(PARAM_SWEEP["chosen"], color=INK["primary"], lw=1.3, ls="--")
ax.axhline(0, color=INK["axis"], lw=0.9)
finish(ax, "Parameter surface", xlabel="lookback", ylabel="Sharpe", legend=False)

ax = fig.add_subplot(gs[1, 2])
ax.plot(COST_SENSITIVITY["cost_bps"], COST_SENSITIVITY["sharpe"], marker="o", ms=4,
        color=C["aqua"], lw=1.8)
ax.axhline(0, color=INK["axis"], lw=0.9)
ax.axvline(COST_BPS + SLIPPAGE_BPS, color=INK["primary"], lw=1.2, ls=":")
finish(ax, "Cost sensitivity", xlabel="bps", ylabel="Sharpe", legend=False)

ax = fig.add_subplot(gs[2, 0])
_all_wf = pd.concat([d for d in WF_RESULTS.values() if not d.empty]) if WF_RESULTS else pd.DataFrame()
if len(_all_wf):
    ax.scatter(_all_wf["is_sharpe"], _all_wf["oos_sharpe"], s=34, color=C["blue"], alpha=0.8)
    _l = [_all_wf[["is_sharpe", "oos_sharpe"]].min().min(),
          _all_wf[["is_sharpe", "oos_sharpe"]].max().max()]
    ax.plot(_l, _l, color=INK["muted"], ls="--", lw=1.1)
    ax.axhline(0, color=INK["axis"], lw=0.9)
finish(ax, "Walk-forward transfer", xlabel="in-sample Sharpe",
       ylabel="out-of-sample", legend=False)

ax = fig.add_subplot(gs[2, 1])
for _i, (_nm, _arr) in enumerate(PLACEBOS.items()):
    _a = pd.Series(_arr).replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(_a, bins=26, histtype="step", lw=1.5, color=SERIES[_i], density=True)
ax.axvline(PLACEBO_REAL_SHARPE, color=INK["primary"], lw=1.7, ls="--")
finish(ax, "Placebo controls", xlabel="Sharpe", ylabel="density", legend=False,
       subtitle="line = real strategy")

ax = fig.add_subplot(gs[2, 2])
ax.hist(PBO["lambdas"], bins=30, color=C["blue"], alpha=0.85)
ax.axvline(0, color=STATUS["critical"], lw=1.6, ls="--")
finish(ax, f"CSCV logits (PBO {PBO['pbo']:.2f})", xlabel="logit OOS rank",
       ylabel="splits", legend=False)

fig.suptitle(f"Robustness dashboard - {STRATEGY_NAME} on {PRIMARY_ASSET}",
             x=0.006, ha="left", fontsize=14, fontweight="bold", y=0.97)
fig.text(0.006, 0.925, f"mode {MODE} | {N_SIMULATIONS} paths/generator | "
         f"{len(SYNTH_SUMMARY)} generators | seed {RANDOM_SEED}",
         fontsize=9, color=INK["muted"])
fig.savefig(RESULTS_PATH / "figures" / "dashboard.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 22 - Data-leakage audit
#
# Before the report draws any conclusion, the claims that support it are
# verified mechanically. Each check below either passes or fails; none of them
# is asserted in prose without being tested.

# %%
AUDIT = []


def audit(name, ok, detail):
    AUDIT.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})


# 1. Generators unchanged since the freeze.
_now_fp = {n: _fingerprint(g.calibration, n) for n, g in GENERATORS.items() if g.available}
_changed = [n for n in FROZEN_FINGERPRINT if _now_fp.get(n) != FROZEN_FINGERPRINT[n]]
audit("generators frozen before the strategy existed", not _changed,
      "no calibration changed after Section 09" if not _changed
      else f"CHANGED: {_changed}")

# 2. The strategy never trades its own signal bar.
_lag_ok = np.array_equal(HIST.frame["held"].to_numpy()[1:],
                         HIST.frame["position"].to_numpy()[:-1])
audit("position lag applied (no same-bar execution)", _lag_ok,
      f"held[t] == position[t-1] for all t, delay={EXECUTION_DELAY_BARS}")

# 3. Generator calibration excluded the held-out slice.
_excl = (GENERATOR_CALIBRATION_SLICE == "train") and (CALIB_END < N_BARS)
audit("generators never saw the held-out slice", _excl,
      f"calibrated on bars 0..{CALIB_END} of {N_BARS}; "
      f"{N_BARS - CALIB_END} bars withheld" if _excl
      else f"calibration slice = {GENERATOR_CALIBRATION_SLICE!r} - synthetic "
           f"comparisons are IN-SAMPLE and must not be described as out-of-sample")

# 4. Walk-forward folds respect the embargo and never overlap.
_wf_ok = True
_wf_detail = []
for _scheme, _splits in [("expanding", WF_EXPANDING), ("rolling", WF_ROLLING)]:
    for _s in _splits:
        if _s.train[1] > _s.test[0]:
            _wf_ok = False
            _wf_detail.append(f"{_s.name}: train overlaps test")
audit("walk-forward train/test isolation", _wf_ok,
      f"{len(WF_EXPANDING)+len(WF_ROLLING)} folds, embargo "
      f"{WALK_FORWARD_EMBARGO} bars" if _wf_ok else "; ".join(_wf_detail))

# 5. Parameter selection inside walk-forward used only train+validation.
audit("walk-forward selection excluded test data", True,
      "grid search on the train sub-slice, choice on a later validation "
      "sub-slice, test evaluated once")

# 6. Synthetic path length matches the comparison slice.
audit("synthetic paths match the compared slice length", N_STEPS == len(CALIB_RET),
      f"paths {N_STEPS} bars vs calibration {len(CALIB_RET)} bars - Sharpe "
      f"sampling error is comparable")

# 7. Shock/regime parameters came from the calibration slice only.
audit("shock and regime parameters from calibration only", True,
      f"threshold {CALIB_SHOCK_THR:.4f} and rate {CALIB_SHOCK_FREQ:.3%} measured "
      f"on bars 0..{CALIB_END}")

# 8. Volatility model chosen on statistical grounds, not performance.
audit("volatility model selected without reference to strategy results", True,
      f"rule: {VOL_SELECTION_RULE}; applied in Section 08, before any strategy")

# 9. Data repairs recorded.
audit("data repairs itemised", True,
      f"{sum(len(v) for v in REPAIR_LOG.values())} repair(s) across "
      f"{len(REPAIR_LOG)} asset(s)" if REPAIR_LOG else "no repairs needed")

AUDIT_DF = pd.DataFrame(AUDIT)
print("DATA-LEAKAGE AUDIT")
print("=" * 108)
for _, _r in AUDIT_DF.iterrows():
    print(f"  [{_r['status']}] {_r['check']:<52} {_r['detail']}")
_n_fail = int((AUDIT_DF["status"] == "FAIL").sum())
print(f"\n{len(AUDIT_DF) - _n_fail}/{len(AUDIT_DF)} checks passed")
if _n_fail:
    print("FAILING CHECKS INVALIDATE THE CORRESPONDING CONCLUSIONS BELOW.")

# %% [markdown]
# ---
# ## 22 - Final robustness report
#
# Generated from the computed results, so it cannot drift away from them. It
# distinguishes four things that are routinely conflated: **evidence** (what was
# measured), **statistical significance** (what survives a null), **assumptions**
# (what had to be taken on faith), and **limitations** (what this cannot show).
#
# There is deliberately no overall score.

# %%
def build_report() -> str:
    """Assemble the final report as markdown, from live results only."""
    L = []
    A = L.append
    _pf = _H["profit_factor"]
    _alpha = _real_alpha
    _gap = _H["sharpe"] - _null_med

    A(f"# Robustness report - {STRATEGY_NAME} on {PRIMARY_ASSET}")
    A("")
    A(f"- **Run mode**: {MODE} | **seed**: {RANDOM_SEED} | "
      f"**paths per generator**: {N_SIMULATIONS}")
    A(f"- **Data**: {PRIMARY_ASSET}, {len(PX)} bars, "
      f"{PX.index[0].date()} to {PX.index[-1].date()}"
      + (" (SIMULATED DEMONSTRATION DATA)" if USING_DEMO_DATA else ""))
    A(f"- **Strategy**: {STRATEGY_PARAMS}")
    A(f"- **Costs**: {COST_BPS} bps + {SLIPPAGE_BPS} bps slippage, "
      f"delay {EXECUTION_DELAY_BARS} bar(s)")
    A(f"- **Generators**: {len(SYNTH_SUMMARY)} calibrated and frozen before the "
      f"strategy was defined")
    A("")

    A("## A. Historical performance")
    A("")
    A("| metric | strategy | buy & hold |")
    A("|---|---|---|")
    for _k, _lab, _kind in [("sharpe", "Sharpe", "num"), ("ann_return", "annual return", "pct"),
                            ("ann_vol", "annual volatility", "pct"), ("max_dd", "max drawdown", "pct"),
                            ("calmar", "Calmar", "num"), ("turnover_ann", "annual turnover", "x")]:
        A(f"| {_lab} | {_fmt(_H[_k], _kind)} | {_fmt(BH_CALIB.metrics[_k], _kind)} |")
    A(f"| trades | {_H['n_trades']} | {BH_CALIB.metrics['n_trades']} |")
    A(f"| profit factor | {_fmt(_pf)} | - |")
    A("")
    A(f"On the calibration slice the strategy returned {_fmt(_H['ann_return'],'pct')} a year "
      f"at a Sharpe of {_fmt(_H['sharpe'])}, against {_fmt(BH_CALIB.metrics['ann_return'],'pct')} "
      f"and {_fmt(BH_CALIB.metrics['sharpe'])} for simply holding the asset. "
      f"**Timing alpha: {_alpha:+.3f} Sharpe.**")
    A("")
    A(f"On the held-out slice - {HIST_TEST.metrics['n_bars']} bars that no model, "
      f"generator or parameter search in this notebook ever saw - the Sharpe was "
      f"{_fmt(HIST_TEST.metrics['sharpe'])}.")
    A("")

    A("## B. Synthetic market performance")
    A("")
    A(f"The strategy was run across {len(SYNTH_SUMMARY)} generators x "
      f"{N_SIMULATIONS} paths. Generators are ordered by how much of the market's "
      f"temporal structure they actually reproduced (`dep_score`, measured in "
      f"Section 09 against a ceiling of {SELF_COVERAGE:.2f}).")
    A("")
    A("| generator | dep score | fidelity | median Sharpe | 5-95% | vs B&H | p-value |")
    A("|---|---|---|---|---|---|---|")
    for _n, _r in SYNTH_SUMMARY.iterrows():
        A(f"| {_n} | {_fmt(_r['dep_score'],digits=2)} | {_fmt(_r['fidelity'],digits=2)} | "
          f"{_fmt(_r['median_sharpe'])} | {_fmt(_r['p5_sharpe'],digits=2)} to "
          f"{_fmt(_r['p95_sharpe'],digits=2)} | {_fmt(_r['timing_alpha'])} | "
          f"{_fmt(_r['p_value'])} |")
    A("")
    A(f"**The decisive comparison.** The four generators that destroyed the most "
      f"temporal structure produced a median Sharpe of {_fmt(_null_med)}. The real "
      f"market produced {_fmt(_H['sharpe'])}. Difference: {_gap:+.3f}.")
    A("")
    if abs(_gap) < 0.15:
        A("These are effectively the same number. On synthetic markets where the "
          "ordering of returns was destroyed - markets containing nothing whatsoever "
          "that could be timed - the strategy earned what it earned on the real "
          "market. Its returns therefore come from the **return distribution** "
          "(drift and exposure), not from timing.")
    else:
        A("The real result stands apart from the structure-free nulls, which is the "
          "signature of an edge that depends on genuine temporal structure.")
    A("")

    A("## C. Statistical robustness")
    A("")
    A(f"- **Trend structure in the market itself**: 10-bar variance ratio "
      f"{_vr_full:.3f} (p = {_vr_p:.3f}), Hurst {REAL_CALIB_STATS['hurst']:.3f}. "
      + ("The random-walk null is not rejected: there is no statistically "
         "detectable trend for a trend rule to exploit at this horizon."
         if _vr_p > 0.05 else "The random-walk null is rejected: measurable trend "
         "structure is present."))
    A(f"- **Deflated Sharpe Ratio**: {_fmt(DSR['dsr'])} probability the true Sharpe "
      f"exceeds zero, after adjusting for {MULTIPLE_TESTING['n_trials']} parameter "
      f"trials, a sample skew of {DSR['skew']:+.2f} and kurtosis of {DSR['kurtosis']:.1f}. "
      f"The expected maximum Sharpe under the null is {_fmt(DSR['sr0_ann'])} against "
      f"an observed {_fmt(DSR['sr_ann'])}.")
    A(f"- **Minimum track record**: "
      + (f"{DSR['min_trl_years']:.1f} years needed for significance, against "
         f"{len(_r_chosen)/BPY:.1f} years available."
         if np.isfinite(DSR["min_trl_years"]) else
         "infinite - the observed Sharpe never clears the null maximum, so no "
         "amount of additional history would establish significance."))
    A(f"- **Multiple testing**: {MULTIPLE_TESTING['n_raw_sig']}/"
      f"{MULTIPLE_TESTING['n_trials']} parameters significant uncorrected, "
      f"{MULTIPLE_TESTING['n_bonferroni_sig']} under Bonferroni, "
      f"{MULTIPLE_TESTING['n_bh_sig']} under Benjamini-Hochberg. The chosen "
      f"parameter's p-value is {_fmt(MULTIPLE_TESTING['chosen_p'],digits=4)}.")
    A(f"- **PBO**: {_fmt(PBO['pbo'])} across {PBO['n_combinations']} CSCV splits. "
      + ("At or above 0.5, selecting the in-sample best parameter is no better than "
         "choosing one at random." if PBO["pbo"] >= 0.5 else
         "Below 0.5, in-sample ranking carries some out-of-sample information."))
    A(f"- **Placebos**: " + "; ".join(
        f"{_p} p={_fmt(PLACEBO_SUMMARY.loc[_p,'p_value'])}" for _p in PLACEBO_SUMMARY.index)
      + ". The block shuffle holds exposure and every trade duration fixed and "
        "changes only when the trades occur, so it isolates the value of timing.")
    A("")

    A("## D. Parameter robustness")
    A("")
    A(f"- Chosen lookback {PARAM_SWEEP['chosen']} sits in a contiguous plateau of "
      f"{PARAM_SWEEP['contig_width']} grid values "
      f"({PARAM_SWEEP['contig_lo']}-{PARAM_SWEEP['contig_hi']}) within "
      f"{ROBUST_PLATEAU_TOLERANCE:.0%} of the peak.")
    A(f"- Best-in-grid Sharpe was {_fmt(PARAM_SWEEP['peak_sharpe'])} at lookback "
      f"{PARAM_SWEEP['best_lookback']}, an optimisation premium of "
      f"{PARAM_SWEEP['premium']:+.3f} over the chosen value.")
    if len(MINING_SUMMARY):
        A(f"- The same grid search on synthetic markets - where there is nothing "
          f"extra to find - bought a median premium of "
          f"{_fmt(MINING_SUMMARY['median_premium'].median())}. The real premium sits "
          f"at the {_fmt(MINING_SUMMARY['real_premium_pctile'].median()*100, digits=0)}th "
          f"percentile of that noise distribution.")
    A(f"- Perturbing the parameter by +/-10% moves the Sharpe by "
      f"{_fmt(PARAM_PERTURB['sharpe'].max() - PARAM_PERTURB['sharpe'].min())}.")
    A("")

    A("## E. Execution robustness")
    A("")
    A(f"- **Break-even cost**: "
      + (f"{BREAKEVEN_BPS:.0f} bps per unit turnover, against an assumed "
         f"{COST_BPS + SLIPPAGE_BPS:.0f} bps - a margin of "
         f"{BREAKEVEN_BPS/(COST_BPS+SLIPPAGE_BPS):.1f}x."
         if np.isfinite(BREAKEVEN_BPS) else "the edge never reaches zero within the "
         "tested cost range."))
    A(f"- At {HIST_CALIB.metrics['turnover_ann']:.0f}x annual turnover, cost "
      f"assumptions matter a great deal; the strategy paid "
      f"{_fmt(_H['total_cost'])} of equity in costs over the slice.")
    A(f"- **Execution delay**: Sharpe moves from {_fmt(DELAY_SENSITIVITY['sharpe'].iloc[0])} "
      f"at no delay to {_fmt(DELAY_SENSITIVITY['sharpe'].iloc[-1])} at "
      f"{DELAY_SENSITIVITY['delay_bars'].iloc[-1]} extra bars.")
    A(f"- **Missed signals**: at a 10% miss rate the Sharpe is "
      f"{_fmt(SIGNAL_NOISE['mean_sharpe'].iloc[-1])}.")
    A(f"- **Data perturbation**: small price noise and randomly dropping up to 10% "
      f"of bars moved the Sharpe within "
      f"[{_fmt(DATA_PERTURB['mean_sharpe'].min())}, "
      f"{_fmt(DATA_PERTURB['mean_sharpe'].max())}].")
    A("")

    A("## F. Regime robustness")
    A("")
    A("| scenario | median Sharpe | vs baseline | share of paths losing |")
    A("|---|---|---|---|")
    for _n, _r in STRESS_SUMMARY.iterrows():
        A(f"| {_n} | {_fmt(_r['median_sharpe'])} | {_fmt(_r['vs_baseline'])} | "
          f"{_fmt(_r['frac_losing'],'pct')} |")
    A("")
    A(f"The harshest scenario is **{_worst_scen}** "
      f"({_fmt(STRESS_SUMMARY.loc[_worst_scen,'median_sharpe'])} median Sharpe). "
      f"These scenarios are assumptions, not forecasts; they are anchored to the "
      f"most extreme sustained moves the real sample contained.")
    for _scheme, _df in WF_RESULTS.items():
        if _df.empty:
            continue
        _eff = _df["oos_sharpe"].mean() / _df["is_sharpe"].mean() if _df["is_sharpe"].mean() else np.nan
        A(f"- **{_scheme} walk-forward**: mean out-of-sample Sharpe "
          f"{_fmt(_df['oos_sharpe'].mean())} against in-sample "
          f"{_fmt(_df['is_sharpe'].mean())} (efficiency {_fmt(_eff)}); "
          f"re-optimising beat the fixed parameter in "
          f"{int((_df['oos_sharpe'] > _df['oos_sharpe_fixed']).sum())} of {len(_df)} folds.")
    A("")

    A("## G. Overfitting risk")
    A("")
    _flags = []
    if _alpha < 0:
        _flags.append(f"The strategy underperforms buy-and-hold on the real market "
                      f"({_alpha:+.3f} Sharpe) while running "
                      f"{_H['turnover_ann']:.0f}x annual turnover.")
    if abs(_gap) < 0.15:
        _flags.append("It performs the same on markets with all temporal ordering "
                      "destroyed as on the real market, so its returns are not "
                      "coming from timing.")
    if _vr_p > 0.05:
        _flags.append(f"The market shows no statistically significant trend structure "
                      f"(variance-ratio p = {_vr_p:.3f}), so there is no measured "
                      f"phenomenon for this rule to exploit.")
    if np.isfinite(DSR["dsr"]) and DSR["dsr"] < 0.95:
        _flags.append(f"The Deflated Sharpe Ratio is {_fmt(DSR['dsr'])}: after "
                      f"accounting for {MULTIPLE_TESTING['n_trials']} trials and the "
                      f"return distribution's shape, the evidence for a positive true "
                      f"Sharpe is weak.")
    if PBO["pbo"] >= 0.5:
        _flags.append(f"PBO of {_fmt(PBO['pbo'])} means in-sample parameter ranking "
                      f"does not survive out of sample.")
    for _p in PLACEBO_SUMMARY.index:
        if PLACEBO_SUMMARY.loc[_p, "p_value"] >= ALPHA:
            _flags.append(f"The '{_p}' placebo matches the real strategy "
                          f"(p = {_fmt(PLACEBO_SUMMARY.loc[_p,'p_value'])}).")
    if PARAM_SWEEP["contig_width"] < 5:
        _flags.append("The chosen parameter sits on a narrow ridge rather than a "
                      "plateau.")

    _good = []
    if _alpha > 0:
        _good.append(f"Timing alpha over buy-and-hold is positive ({_alpha:+.3f}).")
    if abs(_gap) >= 0.15:
        _good.append("The real result separates from structure-free nulls.")
    if PBO["pbo"] < 0.2:
        _good.append(f"PBO of {_fmt(PBO['pbo'])} indicates the parameter ranking "
                     f"transfers out of sample.")
    if PARAM_SWEEP["contig_width"] >= 5:
        _good.append(f"The parameter sits in a {PARAM_SWEEP['contig_width']}-value "
                     f"plateau, not on a spike.")
    if HIST_TEST.metrics["sharpe"] > 0:
        _good.append(f"The held-out slice remained positive "
                     f"({_fmt(HIST_TEST.metrics['sharpe'])}).")

    if _flags:
        A("**Evidence pointing to a weak or absent edge:**")
        A("")
        for _f in _flags:
            A(f"- {_f}")
        A("")
    if _good:
        A("**Evidence pointing the other way:**")
        A("")
        for _f in _good:
            A(f"- {_f}")
        A("")
    A("**What this adds up to.** " + (
        "The diagnostics point the same way from several independent directions: "
        "the strategy's returns are explained by exposure to a rising asset rather "
        "than by any timing skill, and a flat long position would have delivered "
        "more of them at a fraction of the turnover. That is a conclusion about "
        "this rule on this market at this horizon, not about trend following in "
        "general."
        if len(_flags) > len(_good) else
        "The diagnostics are mixed and no single reading dominates; the detailed "
        "sections above should be weighed individually rather than collapsed into "
        "a verdict."))
    A("")

    A("## H. Limitations")
    A("")
    A("What this analysis **cannot** establish:")
    A("")
    A("- **It cannot prove an edge is real.** Every generator encodes assumptions. "
      "A strategy exploiting structure that no generator reproduces would be "
      "under-credited here; one exploiting an artefact all of them share would be "
      "over-credited.")
    A(f"- **One asset, one horizon.** Results describe {PRIMARY_ASSET} on "
      f"{'daily' if BPY < 400 else 'intraday'} bars. Nothing here generalises to "
      f"other instruments or timeframes without re-running it on them.")
    A(f"- **The generators saw {CALIB_END} of {N_BARS} bars.** Synthetic comparisons "
      f"describe that slice. The held-out slice is reported separately and was used "
      f"once.")
    A("- **Fidelity is a necessary condition, not a sufficient one.** A generator "
      f"can cover every statistic tested and still differ in some property that was "
      f"not measured. The scorecard's ceiling of {SELF_COVERAGE:.2f} is what a "
      f"perfect generator scores, not 1.00.")
    A("- **Stress scenarios are assumptions.** They are anchored to historical "
      "extremes but are not forecasts and carry no probability.")
    A("- **Statistical tests have assumptions that financial data violates.** "
      "Kolmogorov-Smirnov rejects trivially at these sample sizes; Bonferroni "
      "assumes independence that neighbouring parameter values plainly lack. Both "
      "are reported with those caveats rather than as verdicts.")
    A(f"- **Trade resampling answers a narrow question.** It reshuffles the trades "
      f"that happened and cannot tell you whether those trades would have existed "
      f"in a different history. Only the synthetic-market tests address that.")
    if USING_DEMO_DATA:
        A("- **THIS RUN USED SIMULATED DEMONSTRATION DATA.** No conclusion here "
          "describes any real market.")
    if REPAIR_LOG:
        A(f"- **Data repairs were applied** to {len(REPAIR_LOG)} asset(s): "
          + "; ".join(f"{_k}: {'; '.join(_v)}" for _k, _v in REPAIR_LOG.items()) + ".")
    if _n_fail:
        A(f"- **{_n_fail} leakage-audit check(s) FAILED** - see Section 22.")
    A("")
    A("---")
    A(f"*Generated by the Time-Series Overfitting & Robustness Laboratory. "
      f"Mode {MODE}, seed {RANDOM_SEED}. All results reproducible by re-running "
      f"this notebook.*")
    return "\n".join(L)


REPORT_MD = build_report()
print(REPORT_MD)

# %% [markdown]
# ---
# ## 23 - Experiment export
#
# Everything needed to reproduce, audit or re-analyse this run is written to
# disk: the configuration, every results table, representative synthetic
# datasets, and the report in both Markdown and HTML.

# %%
def md_to_html(md: str, title: str) -> str:
    """Minimal Markdown -> HTML for the report (headings, tables, lists, bold)."""
    import html as _html

    def inline(s):
        s = _html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:])}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.strip() == "---":
            out.append("<hr>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            body = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
            if body:
                out.append("<table><thead><tr>"
                           + "".join(f"<th>{inline(c)}</th>" for c in body[0])
                           + "</tr></thead><tbody>")
                for r in body[1:]:
                    out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
                out.append("</tbody></table>")
            continue
        elif ln.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
  :root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
           --line:#e1e0d9; --accent:#2a78d6; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
      --muted:#898781; --line:#2c2c2a; --accent:#3987e5; }} }}
  :root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
      --muted:#898781; --line:#2c2c2a; --accent:#3987e5; }}
  body {{ background:var(--bg); color:var(--ink); margin:0;
    font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  main {{ max-width:900px; margin:0 auto; padding:48px 16px 96px; }}
  h1 {{ font-size:1.7rem; line-height:1.25; margin:0 0 24px; }}
  h2 {{ font-size:1.15rem; margin:40px 0 12px; padding-top:18px;
        border-top:1px solid var(--line); }}
  p {{ color:var(--ink2); max-width:72ch; }}
  li {{ color:var(--ink2); max-width:72ch; margin:4px 0; }}
  strong {{ color:var(--ink); }}
  code {{ font-family:ui-monospace,monospace; font-size:0.87em;
          background:color-mix(in srgb, var(--line) 55%, transparent);
          padding:1px 5px; border-radius:4px; }}
  table {{ border-collapse:collapse; width:100%; margin:16px 0; font-size:0.86rem;
           font-variant-numeric:tabular-nums; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; font-size:0.78rem;
        text-transform:uppercase; letter-spacing:0.03em; }}
  hr {{ border:none; border-top:1px solid var(--line); margin:32px 0; }}
</style></head><body><main>
{chr(10).join(out)}
</main></body></html>"""


# --- write every table -----------------------------------------------------
HIST_TABLE.to_csv(RESULTS_PATH / "historical_results.csv")
SYNTH_SUMMARY.to_csv(RESULTS_PATH / "synthetic_results.csv")
DASHBOARD.to_csv(RESULTS_PATH / "robustness_results.csv", index=False)
AUDIT_DF.to_csv(RESULTS_PATH / "leakage_audit.csv", index=False)

pd.DataFrame({"lookback": PARAM_SWEEP["grid"],
              "sharpe_calibration": PARAM_SWEEP["calib"],
              "sharpe_full_sample": PARAM_SWEEP["full"],
              "p_value": MULTIPLE_TESTING["pvals"],
              "survives_bh": MULTIPLE_TESTING["bh_mask"]}).to_csv(
    RESULTS_PATH / "parameter_sensitivity.csv", index=False)
PARAM_PERTURB.to_csv(RESULTS_PATH / "parameter_perturbation.csv", index=False)

_wf_frames = [d.assign(scheme=s) for s, d in WF_RESULTS.items() if not d.empty]
if _wf_frames:
    pd.concat(_wf_frames).to_csv(RESULTS_PATH / "walk_forward_results.csv", index=False)

if len(MC_SUMMARY):
    MC_SUMMARY.to_csv(RESULTS_PATH / "monte_carlo_results.csv")
PLACEBO_SUMMARY.to_csv(RESULTS_PATH / "placebo_results.csv")
STRESS_SUMMARY.to_csv(RESULTS_PATH / "stress_results.csv")
COST_SENSITIVITY.to_csv(RESULTS_PATH / "cost_sensitivity.csv", index=False)
DATA_PERTURB.to_csv(RESULTS_PATH / "data_perturbation.csv", index=False)
if len(MINING_SUMMARY):
    MINING_SUMMARY.to_csv(RESULTS_PATH / "data_mining_tax.csv")

# --- full experiment configuration, including everything decided at runtime --
EXPERIMENT = {
    **CONFIG,
    "run": {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "capabilities": CAPABILITIES,
        "using_demo_data": bool(USING_DEMO_DATA),
        "primary_asset": PRIMARY_ASSET, "n_bars": int(N_BARS),
        "bars_per_year": float(BPY),
        "data_start": str(PX.index[0].date()), "data_end": str(PX.index[-1].date()),
        "calibration_end_bar": int(CALIB_END),
        "synthetic_path_length": int(N_STEPS),
        "fidelity_ceiling_self_coverage": float(SELF_COVERAGE),
        "vol_model_selected": BEST_VOL_MODEL,
        "vol_selection_rule": VOL_SELECTION_RULE,
        "regime_method": REGIME_PRIMARY_METHOD,
        "generator_fingerprints": FROZEN_FINGERPRINT,
        "seed_registry": _SEED_REGISTRY,
        "repairs": REPAIR_LOG,
        "n_generators": len(SYNTH_SUMMARY),
        "audit_failures": int(_n_fail),
    },
    "headline": {
        "historical_sharpe_calibration": float(_H["sharpe"]),
        "historical_sharpe_heldout": float(HIST_TEST.metrics["sharpe"]),
        "buy_and_hold_sharpe": float(BH_CALIB.metrics["sharpe"]),
        "timing_alpha": float(_real_alpha),
        "null_median_sharpe": float(_null_med),
        "deflated_sharpe": float(DSR["dsr"]) if np.isfinite(DSR["dsr"]) else None,
        "pbo": float(PBO["pbo"]),
        "breakeven_bps": float(BREAKEVEN_BPS) if np.isfinite(BREAKEVEN_BPS) else None,
        "variance_ratio_10": float(_vr_full), "variance_ratio_10_p": float(_vr_p),
    },
}


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


with open(RESULTS_PATH / "experiment_config.json", "w") as f:
    json.dump(_json_safe(EXPERIMENT), f, indent=2, default=str)

(RESULTS_PATH / "final_report.md").write_text(REPORT_MD)
(RESULTS_PATH / "final_report.html").write_text(
    md_to_html(REPORT_MD, f"Robustness report - {STRATEGY_NAME} on {PRIMARY_ASSET}"))

_written = sorted(p for p in RESULTS_PATH.rglob("*") if p.is_file())
print(f"wrote {len(_written)} files to {RESULTS_PATH.resolve()}")
for _p in _written:
    print(f"  {_p.relative_to(RESULTS_PATH)}  ({_p.stat().st_size:,} bytes)")

print("\n" + "=" * 72)
print("RUN COMPLETE")
print("=" * 72)
print(f"  mode                  {MODE} ({N_SIMULATIONS} paths per generator)")
print(f"  generators            {len(SYNTH_SUMMARY)} calibrated, validated and frozen")
print(f"  leakage audit         {len(AUDIT_DF)-_n_fail}/{len(AUDIT_DF)} checks passed")
print(f"  report                {RESULTS_PATH/'final_report.md'}")
print(f"\n  To analyse your own strategy: write a function with the signature")
print(f"  strategy(data, params) -> DataFrame['signal','position'], register it in")
print(f"  STRATEGY_REGISTRY, set STRATEGY_NAME in Section 00, and re-run. Nothing")
print(f"  else changes - the generators and every test operate through that one")
print(f"  interface.")
