# %% [markdown]
# ---
# ## 09 - Synthetic data validation
#
# An overfitting test run against unfaithful synthetic data is worthless, and
# worse than worthless if it produces a confident verdict. So before any
# strategy is defined, every generator is scored on how well its output
# reproduces the statistics measured in Section 05.
#
# ### How the scoring works
#
# For each generator, many paths are produced and each is passed through the
# same `characterise()` used on the real data. That gives a *distribution* for
# every statistic. The real value is then located inside that distribution, and
# the statistic counts as **covered** if the real value falls within the central
# 90% of the synthetic spread.
#
# Two things about this metric are worth being explicit about, because they
# change how the numbers should be read:
#
# 1. **A perfect generator scores about 0.90, not 1.00.** With a 5th-95th
#    percentile band, one statistic in ten falls outside by construction. The
#    calibration check below measures this directly by scoring a generator's own
#    path against its siblings, and the number it returns is the realistic
#    ceiling for every other score in the table.
# 2. **Coverage is a necessary condition, not a sufficient one.** A generator
#    with enormous variance covers everything by being uselessly vague. The
#    median column is printed alongside so that a wide-and-empty generator is
#    visible as such.
#
# The comparison target is the **calibration slice**, not the full sample: the
# generators only ever saw that slice, and the synthetic paths are the same
# length as it.

# %%
N_FIDELITY_PATHS = int(min(N_SIMULATIONS, 250))
REF_SHOCK_THR = float(np.quantile(np.abs(CALIB_RET), SHOCK_PRIMARY_QUANTILE))

REAL_CALIB_STATS = characterise(CALIB_RET, BPY, f"{PRIMARY_ASSET} (calibration)",
                                ref_threshold=REF_SHOCK_THR)

FIDELITY_GROUPS = {
    "distribution": ["ann_vol", "skew", "kurtosis", "q01", "q99"],
    "tails": ["hill_left", "hill_right", "frac_beyond_3sd"],
    "dependence": ["acf1", "acf_abs1", "acf_abs10", "acf_abs20", "acf_sq1"],
    "volatility": ["vol_of_vol", "vol_persistence", "high_vol_frac"],
    "trend": ["vr2", "vr5", "vr10", "vr20", "hurst", "mean_up_run", "frac_time_up"],
    "drawdown": ["max_dd", "mean_dd_duration", "frac_time_underwater"],
    "shocks": ["abs_shock_freq", "abs_shock_mean_mag", "abs_shock_clustering_ratio",
               "post_shock_vol_ratio"],
}
FIDELITY_STATS = [s for group in FIDELITY_GROUPS.values() for s in group]

print(f"scoring {len(FIDELITY_STATS)} statistics in {len(FIDELITY_GROUPS)} groups")
print(f"{N_FIDELITY_PATHS} paths per generator, {N_STEPS} bars each")
print(f"reference shock threshold |r| >= {REF_SHOCK_THR:.4f} "
      f"(fixed, from the calibration slice)")


def characterise_paths(logret_paths, bpy, ref_threshold):
    """Characterise a set of synthetic log-return paths -> one row per path."""
    simple = np.expm1(logret_paths)
    return pd.DataFrame([
        characterise(simple[i], bpy, f"path{i}", ref_threshold=ref_threshold)
        for i in range(simple.shape[0])
    ])


def coverage_table(real_stats: dict, synth: pd.DataFrame,
                   stat_names: list[str], lo=5.0, hi=95.0) -> pd.DataFrame:
    """Locate each real statistic inside the synthetic distribution of it."""
    rows = []
    for s in stat_names:
        rv = real_stats.get(s, np.nan)
        col = synth[s].replace([np.inf, -np.inf], np.nan).dropna() if s in synth else pd.Series(dtype=float)
        if not np.isfinite(rv) or len(col) < 10:
            rows.append({"stat": s, "real": rv, "synth_median": np.nan,
                         "synth_p5": np.nan, "synth_p95": np.nan,
                         "real_pctile": np.nan, "covered": np.nan})
            continue
        p5, p95 = np.percentile(col, [lo, hi])
        rows.append({
            "stat": s, "real": float(rv), "synth_median": float(col.median()),
            "synth_p5": float(p5), "synth_p95": float(p95),
            "real_pctile": float((col <= rv).mean()),
            "covered": bool(p5 <= rv <= p95),
        })
    return pd.DataFrame(rows)


def fidelity_scores(cov: pd.DataFrame) -> dict:
    """Collapse a coverage table into an overall score and per-group scores."""
    out = {}
    valid = cov.dropna(subset=["covered"])
    out["overall"] = float(valid["covered"].mean()) if len(valid) else np.nan
    out["n_scored"] = int(len(valid))
    for g, names in FIDELITY_GROUPS.items():
        sub = valid[valid["stat"].isin(names)]
        out[g] = float(sub["covered"].mean()) if len(sub) else np.nan
    return out


print("fidelity machinery defined")

# %%
# The expensive cell: generate and characterise paths for every generator.
# Each generator is handled and discarded in turn so that peak memory stays at
# one generator's worth of paths rather than twenty.
import time as _time

SYNTH_STATS: dict[str, pd.DataFrame] = {}
SYNTH_SAMPLES: dict[str, np.ndarray] = {}
COVERAGE: dict[str, pd.DataFrame] = {}
FIDELITY: dict[str, dict] = {}
GEN_ERRORS: dict[str, str] = {}

_avail_names = [n for n, g in GENERATORS.items() if g.available]
print(f"running fidelity pass over {len(_avail_names)} generators "
      f"({N_FIDELITY_PATHS} paths each)\n")

for _name in _avail_names:
    _g = GENERATORS[_name]
    _t0 = _time.time()
    try:
        _rng_g = get_rng(f"fidelity::{_name}")
        _paths = _g(_rng_g, N_FIDELITY_PATHS, N_STEPS)
        _stats = characterise_paths(_paths, BPY, REF_SHOCK_THR)
        _cov = coverage_table(REAL_CALIB_STATS, _stats, FIDELITY_STATS)
        _sc = fidelity_scores(_cov)

        SYNTH_STATS[_name] = _stats
        COVERAGE[_name] = _cov
        FIDELITY[_name] = _sc
        SYNTH_SAMPLES[_name] = _paths[:SAVE_REPRESENTATIVE_PATHS].copy()
        print(f"  {_name:<28} fidelity {_sc['overall']:.2f}  "
              f"({_sc['n_scored']} stats)  {_time.time()-_t0:5.1f}s")
    except Exception as exc:
        GEN_ERRORS[_name] = f"{type(exc).__name__}: {exc}"
        _g.available, _g.error = False, GEN_ERRORS[_name]
        print(f"  {_name:<28} FAILED: {GEN_ERRORS[_name]}")

print(f"\ncompleted: {len(FIDELITY)} generators scored, {len(GEN_ERRORS)} failed")

# %% [markdown]
# ### Calibrating the scorecard against itself
#
# Before reading any score, it needs a reference point. One path from a
# generator is held out and treated as if it were the real data, then scored
# against its own siblings. Since that path genuinely came from the generator,
# the resulting coverage is what a **perfect** generator achieves - it isolates
# the irreducible part of the score that comes from the 90% band and from
# sampling noise, rather than from any mismatch.
#
# Every score in the table that follows should be read against this ceiling, not
# against 1.00.

# %%
_calib_gen = "stationary_L20" if "stationary_L20" in SYNTH_STATS else (
    next(iter(SYNTH_STATS)) if SYNTH_STATS else None)

if _calib_gen and len(SYNTH_STATS[_calib_gen]) >= 30:
    _df = SYNTH_STATS[_calib_gen]
    _self_scores = []
    for _i in range(min(25, len(_df))):
        _pseudo_real = _df.iloc[_i].to_dict()
        _rest = _df.drop(_df.index[_i])
        _self_scores.append(fidelity_scores(
            coverage_table(_pseudo_real, _rest, FIDELITY_STATS))["overall"])
    SELF_COVERAGE = float(np.mean(_self_scores))
    print(f"self-coverage calibration using {_calib_gen}:")
    print(f"  held out {len(_self_scores)} of its own paths, scored each against the rest")
    print(f"  mean coverage = {SELF_COVERAGE:.3f}  (sd {np.std(_self_scores):.3f})")
    print(f"  nominal expectation from a 5th-95th band = 0.900")
    print(f"\n=> treat {SELF_COVERAGE:.2f} as the practical ceiling. A generator scoring")
    print("   near it is as faithful as this metric can detect; one scoring well below")
    print("   it is measurably failing to reproduce the real market's statistics.")
else:
    SELF_COVERAGE = 0.90
    print("insufficient paths for self-coverage calibration; using the nominal 0.90")

# %%
_fid = pd.DataFrame(FIDELITY).T
_fid.index.name = "generator"
_fid = _fid.sort_values("overall", ascending=False)
_fid["family"] = [GENERATORS[n].family for n in _fid.index]
_cols = ["family", "overall"] + list(FIDELITY_GROUPS)
print(f"Fidelity scorecard - fraction of statistics whose real value falls inside")
print(f"the generator's own 5th-95th percentile band (ceiling {SELF_COVERAGE:.2f})")
print("=" * 100)
print(_fid[_cols].round(3).to_string())

_best = _fid.index[0]
_worst = _fid.index[-1]
print(f"\nmost faithful : {_best}  ({_fid.loc[_best,'overall']:.2f})")
print(f"least faithful: {_worst}  ({_fid.loc[_worst,'overall']:.2f})")
print("\nThe least faithful generators are NOT failures - they are controls. The IID")
print("bootstrap is SUPPOSED to score badly on the dependence and trend groups,")
print("because destroying that structure is its entire purpose. What matters is")
print("that each generator fails where its design says it should, which the")
print("per-group columns make checkable.")

# %%
# Where does each generator fail? Per-group, versus the calibrated ceiling.
_gap = _fid[list(FIDELITY_GROUPS)].astype(float)
print("Groups where each generator falls furthest short of the ceiling")
print("=" * 78)
for _n in _fid.index:
    _row = _gap.loc[_n].dropna()
    if _row.empty:
        continue
    _bad = _row[_row < SELF_COVERAGE - 0.25].sort_values()
    _desc = ", ".join(f"{g} {v:.2f}" for g, v in _bad.items()) if len(_bad) else "none"
    print(f"  {_n:<28} {_desc}")
print("\nCompare these against the 'destroys' column of the Section 08 table. A")
print("generator whose weak groups match what it was designed to destroy is")
print("behaving correctly. One that fails a group it claims to preserve is a bug.")

# %%
fig, ax = plt.subplots(figsize=(9.5, 6.4))
_mat = _fid[list(FIDELITY_GROUPS)].astype(float).to_numpy()
_cmap = mpl.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
_im = ax.imshow(_mat, cmap=_cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(FIDELITY_GROUPS)))
ax.set_xticklabels(list(FIDELITY_GROUPS), rotation=35, ha="right", fontsize=8.5)
ax.set_yticks(range(len(_fid)))
ax.set_yticklabels(_fid.index, fontsize=8)
for i in range(_mat.shape[0]):
    for j in range(_mat.shape[1]):
        v = _mat[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v > 0.55 else INK["primary"])
ax.set_title("Fidelity by statistic group", loc="left")
ax.text(0.0, 1.03, "1.00 = the real value sits inside the synthetic 5th-95th band "
        "for every statistic in the group", transform=ax.transAxes,
        fontsize=8.5, color=INK["muted"], va="bottom")
ax.grid(False)
fig.colorbar(_im, ax=ax, shrink=0.55, label="coverage")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Distributional distance tests
#
# Coverage asks whether summary statistics line up. These tests compare the full
# return distributions directly. Each answers a different question and each has
# a limitation that matters here:
#
# | test | measures | limitation in this setting |
# |---|---|---|
# | **Kolmogorov-Smirnov** | largest gap between the two empirical CDFs | with thousands of observations it rejects on differences far too small to matter; read the *statistic*, not the p-value |
# | **Anderson-Darling** | CDF gap, weighted toward the tails | same large-sample sensitivity; more informative than KS for fat tails because it weights them |
# | **Wasserstein** | the "work" to move one distribution onto the other | an effect size in return units, not a test - reported relative to the real standard deviation so it is unit-free |
# | **Jensen-Shannon** | information-theoretic divergence of the binned densities | depends on the binning; bounded in [0, 1], so it is comparable across generators |
#
# **Do not treat a small p-value as proof the generator is broken, or a large one
# as proof it is correct.** Two series can share a marginal distribution exactly
# and still behave completely differently in time - the IID bootstrap is the
# proof of that, and it will score near-perfectly on every test in this table
# while destroying every temporal property a strategy might use. These tests
# check one necessary property in isolation. The coverage scorecard above, and
# the dependence plots below, are what cover the rest.

# %%
_rng_dist = get_rng("distributional_tests")
_real_r = CALIB_RET.astype(float)
_rows = []
for _name in _fid.index:
    if _name not in SYNTH_SAMPLES:
        continue
    _syn = np.expm1(SYNTH_SAMPLES[_name]).ravel()
    if len(_syn) > 20000:
        _syn = _rng_dist.choice(_syn, 20000, replace=False)
    try:
        _ks = stats.ks_2samp(_real_r, _syn)
        _w = stats.wasserstein_distance(_real_r, _syn) / _real_r.std()
        _lo = min(_real_r.min(), _syn.min())
        _hi = max(_real_r.max(), _syn.max())
        _bins = np.linspace(_lo, _hi, 120)
        _p = np.histogram(_real_r, bins=_bins, density=True)[0] + 1e-12
        _q = np.histogram(_syn, bins=_bins, density=True)[0] + 1e-12
        # scipy returns NaN when the two histograms coincide, because the
        # quantity under its square root lands on a tiny negative. That case is
        # a divergence of exactly zero, not a failure - IAAFT hits it by design,
        # since it reproduces the marginal distribution exactly.
        _jsv = jensenshannon(_p / _p.sum(), _q / _q.sum(), base=2)
        _js = float(_jsv) if np.isfinite(_jsv) else 0.0
        try:
            _ad = float(stats.anderson_ksamp([_real_r, _syn]).statistic)
        except Exception:
            _ad = np.nan
        _rows.append({"generator": _name, "ks_stat": float(_ks.statistic),
                      "ks_p": float(_ks.pvalue), "ad_stat": _ad,
                      "wasserstein_rel": float(_w), "js_div": _js})
    except Exception as exc:
        _rows.append({"generator": _name, "ks_stat": np.nan, "ks_p": np.nan,
                      "ad_stat": np.nan, "wasserstein_rel": np.nan, "js_div": np.nan})

DIST_TESTS = pd.DataFrame(_rows).set_index("generator").sort_values("wasserstein_rel")
print("Distributional distance from the real calibration returns (smaller = closer)")
print(DIST_TESTS.round(4).to_string())
print(f"\nKS rejects at p<0.05 for {int((DIST_TESTS['ks_p'] < 0.05).sum())} of "
      f"{len(DIST_TESTS)} generators.")
print("Read that number with the caveat above in mind: with thousands of")
print("observations, KS detects differences far smaller than anything that would")
print("change a backtest. The Wasserstein column - an effect size expressed in")
print("units of the real standard deviation - is the more useful ranking.")

# %%
# Visual validation. A handful of representative paths, never all of them.
_show = [n for n in ["iid_bootstrap", "stationary_L20", "filtered_historical_sim",
                     "iaaft_surrogate"] if n in SYNTH_SAMPLES]
_show += [n for n in _fid.index if n not in _show][:max(0, 4 - len(_show))]
_show = _show[:4]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex=True)
_real_px = np.exp(np.cumsum(CALIB_LOGRET))
for ax, _name in zip(axes.ravel(), _show):
    for _i, _p in enumerate(SYNTH_SAMPLES[_name][:4]):
        ax.plot(np.exp(np.cumsum(_p)), color=C["blue"], lw=0.85, alpha=0.5,
                label="synthetic paths" if _i == 0 else None)
    ax.plot(_real_px, color=INK["primary"], lw=1.8, label="real", zorder=5)
    ax.set_yscale("log")
    finish(ax, f"{_name}  (fidelity {_fid.loc[_name,'overall']:.2f})",
           ylabel="index (log)", xlabel="bar")
fig.suptitle("Real vs. synthetic price paths", x=0.005, ha="left", fontsize=12,
             fontweight="bold", y=1.0)
fig.tight_layout()
plt.show()
print("Synthetic paths should look like plausible alternative histories of the same")
print("market - similar scale of movement, similar character - without tracing the")
print("real path. If any synthetic path shadows the real line closely, the")
print("generator is replaying history rather than resampling it.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))

ax = axes[0]
_bins = np.linspace(np.quantile(_real_r, 0.001), np.quantile(_real_r, 0.999), 90)
ax.hist(_real_r, bins=_bins, density=True, color=INK["muted"], alpha=0.55, label="real")
for _i, _name in enumerate(_show[:3]):
    _s = np.expm1(SYNTH_SAMPLES[_name]).ravel()
    ax.hist(_s, bins=_bins, density=True, histtype="step", lw=1.6,
            color=SERIES[_i], label=_name)
ax.set_yscale("log")
finish(ax, "Return distribution", xlabel="simple return", ylabel="density (log)")

ax = axes[1]
_lags = 25
ax.plot(range(1, _lags + 1), acf(np.abs(_real_r), _lags)[1:], color=INK["primary"],
        lw=2.2, label="real", zorder=5)
for _i, _name in enumerate(_show[:3]):
    _band = np.array([acf(np.abs(np.expm1(p)), _lags)[1:] for p in SYNTH_SAMPLES[_name]])
    ax.plot(range(1, _lags + 1), np.median(_band, axis=0), color=SERIES[_i],
            lw=1.5, label=_name)
ax.axhline(0, color=INK["axis"], lw=0.9)
finish(ax, "Volatility clustering: ACF of |returns|", xlabel="lag (bars)",
       ylabel="autocorrelation",
       subtitle="the property the IID bootstrap destroys by design")

ax = axes[2]
_real_dd = drawdown_episodes(np.cumprod(1 + _real_r))["depth"] * 100
ax.hist(_real_dd, bins=40, density=True, color=INK["muted"], alpha=0.55, label="real")
for _i, _name in enumerate(_show[:3]):
    _dds = np.concatenate([drawdown_episodes(np.cumprod(1 + np.expm1(p)))["depth"].to_numpy()
                           for p in SYNTH_SAMPLES[_name][:3]]) * 100
    ax.hist(_dds, bins=40, density=True, histtype="step", lw=1.6,
            color=SERIES[_i], label=_name)
ax.set_yscale("log")
finish(ax, "Drawdown depth distribution", xlabel="episode depth (%)", ylabel="density (log)")

fig.tight_layout()
plt.show()

print("The middle panel is the decisive one. The real line decays slowly and stays")
print("positive - volatility clustering. Any generator whose line sits flat at zero")
print("has produced a market where calm and turbulent periods do not exist, however")
print("well it matched the return histogram in the left panel.")

# %% [markdown]
# ### Freezing the generators
#
# This is the commitment point of the honest workflow. Every generator has now
# been calibrated from market data and scored against market statistics, and no
# strategy has been defined, backtested, or even named. Nothing after this cell
# re-fits a generator.
#
# That ordering is not a stylistic preference. The failure mode it prevents is
# the one that makes a robustness study worthless: adjusting the synthetic-data
# model until the strategy clears it, which turns the whole exercise into an
# elaborate way of confirming what you hoped. A fingerprint of every
# generator's calibration is recorded here and re-verified in Section 22, so
# that a violation would be detected rather than assumed away.

# %%
import hashlib


def _fingerprint(obj, name: str = "") -> str:
    """A stable digest of a generator's calibration, for tamper detection.

    The name is folded in so that two generators sharing a calibration but
    differing in sampling method (a moving-block and a stationary bootstrap at
    the same block length, for instance) get distinct fingerprints.
    """
    h = hashlib.sha256()
    h.update(name.encode())
    for k in sorted(obj):
        v = obj[k]
        h.update(str(k).encode())
        if isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes()
                     if v.dtype.kind in "fiu" else str(v.tolist()).encode())
        elif isinstance(v, (int, float, str, bool, type(None))):
            h.update(str(v).encode())
        elif isinstance(v, (list, tuple)):
            h.update(str([getattr(x, "tolist", lambda: x)() for x in v]).encode())
        else:
            h.update(type(v).__name__.encode())
    return h.hexdigest()[:16]


GENERATORS_FROZEN = True
FROZEN_FINGERPRINT = {n: _fingerprint(g.calibration, n)
                      for n, g in GENERATORS.items() if g.available}

print("=" * 72)
print("GENERATORS FROZEN")
print("=" * 72)
print(f"  {len(FROZEN_FINGERPRINT)} generators, calibrated on bars 0..{CALIB_END} "
      f"(slice = {GENERATOR_CALIBRATION_SLICE!r})")
print(f"  volatility model selected by: {VOL_SELECTION_RULE}")
print(f"  fidelity ceiling (self-coverage): {SELF_COVERAGE:.3f}")
print(f"  no strategy has been defined at this point in the notebook")
print("\n  fingerprints:")
for _n, _fp in list(FROZEN_FINGERPRINT.items())[:6]:
    print(f"    {_n:<28} {_fp}")
if len(FROZEN_FINGERPRINT) > 6:
    print(f"    ... and {len(FROZEN_FINGERPRINT)-6} more")

# Persist the fidelity results before moving on.
_fid.to_csv(RESULTS_PATH / "synthetic_statistics.csv")
DIST_TESTS.to_csv(RESULTS_PATH / "synthetic_distance_tests.csv")
for _name, _paths in SYNTH_SAMPLES.items():
    _px = pd.DataFrame(
        {f"path_{i}": np.exp(np.cumsum(_paths[i])) for i in range(len(_paths))})
    _px.to_csv(RESULTS_PATH / "synthetic_samples" / f"{_name}.csv", index_label="bar")
print(f"\nwrote fidelity tables and {len(SYNTH_SAMPLES)} representative path files "
      f"to {RESULTS_PATH}")
