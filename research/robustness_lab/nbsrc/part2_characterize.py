# %% [markdown]
# ---
# ## 05 - Statistical characterisation
#
# This is the measurement stage of the honest workflow. Everything measured here
# becomes the specification the synthetic generators must hit, and Section 09
# scores them against exactly these numbers using exactly this code.
#
# One function, `characterise()`, computes every statistic, and it is applied
# identically to real and synthetic series. That matters: if real and synthetic
# were measured by different code paths, any difference between them could be an
# artefact of the measurement rather than the data.
#
# The statistics are grouped by what they capture:
#
# | group | captures | why a generator must reproduce it |
# |---|---|---|
# | **distribution** | mean, sd, skew, kurtosis, quantiles | how big a typical and an atypical move is |
# | **tails** | Hill index, exceedance frequencies | how often the extreme happens |
# | **dependence** | ACF of returns, \|returns\|, returns² | whether moves predict later moves |
# | **volatility** | vol of vol, persistence, clustering | whether calm and wild periods bunch |
# | **trend** | variance ratio, Hurst, run lengths | whether price *trends* - the thing a trend rule trades |
# | **drawdown** | depth, duration, recovery | whether the pain pattern is realistic |
# | **shocks** | frequency, magnitude, clustering, aftermath | whether crises look like crises |
#
# The trend group is the one that decides whether this whole exercise is
# meaningful. A generator that destroys trend structure cannot fairly evaluate a
# trend-following strategy, and one that manufactures trend structure the real
# market lacks will flatter it.

# %%
def acf(x: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Sample autocorrelation, lags 0..nlags (biased estimator, the standard one).

    Hand-rolled so that real and synthetic series are measured by identical code
    with no optional dependency in the path. Verified against statsmodels below.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    denom = float(np.dot(x, x))
    if n < 3 or denom <= 0:
        return np.full(nlags + 1, np.nan)
    nlags = min(nlags, n - 2)
    out = np.empty(nlags + 1)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = float(np.dot(x[:-k], x[k:])) / denom
    return out


def pacf_durbin_levinson(r: np.ndarray, nlags: int) -> np.ndarray:
    """Partial autocorrelations from an ACF via the Durbin-Levinson recursion."""
    nlags = min(nlags, len(r) - 1)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    if nlags >= 1:
        phi[1, 1] = r[1]
        out[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out[k] = phi[k, k]
    return out


def variance_ratio(logret: np.ndarray, q: int) -> tuple[float, float, float]:
    """Lo-MacKinlay variance ratio with the heteroskedasticity-robust z-statistic.

    VR(q) = Var(q-bar return) / (q * Var(1-bar return)).
      VR > 1 -> positive serial dependence: moves extend. This is trend.
      VR = 1 -> a random walk.
      VR < 1 -> mean reversion.

    The robust z is used because financial returns are strongly
    heteroskedastic; the homoskedastic version would reject a random walk far
    too often purely because volatility clusters.

    Returns (vr, z_statistic, two_sided_p).
    """
    r = np.asarray(logret, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4 * q or q < 2:
        return np.nan, np.nan, np.nan
    mu = r.mean()
    dev = r - mu
    sig_a = float(np.dot(dev, dev)) / (n - 1)
    if sig_a <= 0:
        return np.nan, np.nan, np.nan

    x = np.cumsum(r)
    diffs = x[q:] - x[:-q] - q * mu           # overlapping q-period deviations
    m = q * (n - q + 1) * (1.0 - q / n)
    sig_c = float(np.dot(diffs, diffs)) / m
    vr = sig_c / sig_a

    # Lo-MacKinlay heteroskedasticity-consistent variance of the VR statistic:
    #   delta_j = sum_t (r_t - mu)^2 (r_{t-j} - mu)^2 / [ sum_t (r_t - mu)^2 ]^2
    #   theta   = sum_{j=1}^{q-1} [2(q-j)/q]^2 * delta_j
    # There is deliberately NO factor of n here - it is already carried by the
    # squared sum in the denominator. Including one inflates theta by n and
    # collapses every z-statistic toward zero, which makes even a strongly
    # autocorrelated series look like a random walk.
    theta = 0.0
    d2 = dev**2
    denom = float(np.sum(d2)) ** 2
    if denom <= 0:
        return vr, np.nan, np.nan
    for j in range(1, q):
        delta = float(np.dot(d2[j:], d2[:-j])) / denom
        theta += (2.0 * (q - j) / q) ** 2 * delta
    if theta <= 0:
        return vr, np.nan, np.nan
    z = (vr - 1.0) / math.sqrt(theta)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(vr), float(z), float(p)


def hurst_rs(x: np.ndarray, min_w: int = 16, max_w: int | None = None) -> float:
    """Hurst exponent by rescaled-range (R/S) analysis over dyadic windows.

    H = 0.5 random walk, H > 0.5 persistent/trending, H < 0.5 mean-reverting.
    A crude estimator - reported alongside the variance ratio rather than
    instead of it, because R/S is known to be biased in short samples.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 128:
        return np.nan
    max_w = max_w or n // 4
    widths, rs_vals = [], []
    w = min_w
    while w <= max_w:
        n_chunks = n // w
        if n_chunks < 1:
            break
        vals = []
        for i in range(n_chunks):
            seg = x[i * w:(i + 1) * w]
            dev = np.cumsum(seg - seg.mean())
            rng = dev.max() - dev.min()
            sd = seg.std(ddof=1)
            if sd > 0 and rng > 0:
                vals.append(rng / sd)
        if vals:
            widths.append(w)
            rs_vals.append(np.mean(vals))
        w *= 2
    if len(widths) < 3:
        return np.nan
    slope = np.polyfit(np.log(widths), np.log(rs_vals), 1)[0]
    return float(slope)


def hill_tail_index(x: np.ndarray, tail_frac: float = 0.05) -> float:
    """Hill estimator of the tail index alpha for |x|.

    Smaller alpha means a fatter tail. alpha <= 4 implies infinite kurtosis in
    the limiting distribution, alpha <= 2 implies infinite variance. Reported
    because 'excess kurtosis = 12' is hard to interpret while 'alpha = 3.1,
    so the fourth moment does not exist' is not.
    """
    a = np.sort(np.abs(np.asarray(x, dtype=float)))[::-1]
    a = a[np.isfinite(a) & (a > 0)]
    k = int(len(a) * tail_frac)
    if k < 10:
        return np.nan
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of maximal consecutive-True runs in a boolean array."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return np.array([], dtype=int)
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return ends - starts


def drawdown_episodes(equity: np.ndarray) -> pd.DataFrame:
    """Every distinct underwater episode in an equity curve.

    Returns one row per episode with depth (negative fraction), total duration,
    bars to trough, bars from trough back to the old high, and whether the
    episode had recovered by the end of the sample. The final episode is
    typically unrecovered and is flagged, because counting an in-progress
    drawdown as if it had ended would understate duration.
    """
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    underwater = eq < peak * (1 - 1e-12)
    rows, i, n = [], 0, len(eq)
    while i < n:
        if not underwater[i]:
            i += 1
            continue
        j = i
        while j < n and underwater[j]:
            j += 1
        seg = eq[i:j]
        pk = peak[i]
        t_rel = int(np.argmin(seg))
        rows.append({
            "start": i, "end": j - 1, "duration": j - i,
            "depth": float(seg.min() / pk - 1.0),
            "bars_to_trough": t_rel + 1,
            "bars_to_recover": (j - i) - (t_rel + 1),
            "recovered": bool(j < n),
        })
        i = j
    return pd.DataFrame(rows, columns=["start", "end", "duration", "depth",
                                       "bars_to_trough", "bars_to_recover", "recovered"])


def shock_stats(r: np.ndarray, method: str, threshold_sd: float,
                quantile: float, window: int) -> dict:
    """Identify unusually large moves and describe their behaviour.

    `method='sd'` thresholds at k standard deviations, which assumes the sd is a
    meaningful scale - questionable under fat tails. `method='quantile'`
    thresholds at an empirical quantile of |r| and makes no distributional
    assumption; it is the default for that reason.

    clustering_ratio compares the mean gap between shocks with the gap a
    memoryless (Poisson) process would produce at the same rate. Above 1 means
    shocks arrive closer together than chance - they cluster.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    a = np.abs(r)
    n = len(r)
    if n < 50:
        return {k: np.nan for k in ("shock_threshold", "shock_freq", "shock_mean_mag",
                                    "shock_max_mag", "shock_clustering_ratio",
                                    "shock_followed_frac", "post_shock_vol_ratio",
                                    "pre_shock_vol_ratio", "shock_neg_frac")}
    thr = threshold_sd * r.std() if method == "sd" else float(np.quantile(a, quantile))
    idx = np.where(a >= thr)[0]
    k = len(idx)
    if k < 2:
        return {"shock_threshold": thr, "shock_freq": k / n, "shock_mean_mag": np.nan,
                "shock_max_mag": np.nan, "shock_clustering_ratio": np.nan,
                "shock_followed_frac": np.nan, "post_shock_vol_ratio": np.nan,
                "pre_shock_vol_ratio": np.nan, "shock_neg_frac": np.nan}

    freq = k / n
    gaps = np.diff(idx)
    expected_gap = 1.0 / freq
    clustering = expected_gap / gaps.mean() if gaps.mean() > 0 else np.nan
    followed = float(np.mean(gaps <= window))

    base_vol = a.mean()
    post = [a[i + 1:i + 1 + window] for i in idx if i + 1 + window <= n]
    pre = [a[max(0, i - window):i] for i in idx if i - window >= 0]
    post_ratio = float(np.mean(np.concatenate(post)) / base_vol) if post else np.nan
    pre_ratio = float(np.mean(np.concatenate(pre)) / base_vol) if pre else np.nan

    return {
        "shock_threshold": float(thr), "shock_freq": float(freq),
        "shock_mean_mag": float(a[idx].mean()), "shock_max_mag": float(a[idx].max()),
        "shock_clustering_ratio": float(clustering), "shock_followed_frac": followed,
        "post_shock_vol_ratio": post_ratio, "pre_shock_vol_ratio": pre_ratio,
        "shock_neg_frac": float(np.mean(r[idx] < 0)),
    }


print("statistical primitives defined")

# %% [markdown]
# ### Cross-check the hand-rolled primitives
#
# `acf` and the variance ratio are load-bearing: the fidelity scores in Section
# 09 and the trend conclusions both depend on them. Before trusting them, they
# are checked against an independent implementation and against series whose
# right answer is known analytically.

# %%
_rng_chk = get_rng("primitive_checks")

# ACF against statsmodels on real data.
if HAVE_STATSMODELS:
    _mine = acf(RET.to_numpy(), 20)
    _theirs = sm.tsa.acf(RET.to_numpy(), nlags=20, fft=False)
    _err = float(np.max(np.abs(_mine - _theirs)))
    print(f"[{'PASS' if _err < 1e-10 else 'FAIL'}] acf matches statsmodels "
          f"(max abs diff {_err:.2e})")
else:
    print("[SKIP] acf cross-check - statsmodels not installed")

# Variance ratio on a known random walk: VR should sit at 1 and not reject.
_wn = _rng_chk.standard_normal(8000) * 0.01
_vr, _z, _p = variance_ratio(_wn, 10)
print(f"[{'PASS' if abs(_vr - 1) < 0.12 and _p > 0.01 else 'FAIL'}] "
      f"variance ratio on white noise: VR={_vr:.3f}, p={_p:.3f} (expect ~1, not significant)")

# Variance ratio on a deliberately trending (positively autocorrelated) series.
_tr = np.zeros(8000)
for _t in range(1, 8000):
    _tr[_t] = 0.35 * _tr[_t - 1] + _rng_chk.standard_normal() * 0.01
_vr2, _z2, _p2 = variance_ratio(_tr, 10)
print(f"[{'PASS' if _vr2 > 1.3 and _p2 < 0.01 else 'FAIL'}] "
      f"variance ratio on AR(1) phi=+0.35: VR={_vr2:.3f}, p={_p2:.2e} (expect >1, significant)")

# Variance ratio on a mean-reverting series.
_mr = np.zeros(8000)
for _t in range(1, 8000):
    _mr[_t] = -0.35 * _mr[_t - 1] + _rng_chk.standard_normal() * 0.01
_vr3, _z3, _p3 = variance_ratio(_mr, 10)
print(f"[{'PASS' if _vr3 < 0.8 and _p3 < 0.01 else 'FAIL'}] "
      f"variance ratio on AR(1) phi=-0.35: VR={_vr3:.3f}, p={_p3:.2e} (expect <1, significant)")

# Hurst on white noise should land near 0.5.
_h = hurst_rs(_wn)
print(f"[{'PASS' if 0.40 < _h < 0.62 else 'FAIL'}] Hurst on white noise = {_h:.3f} (expect ~0.5)")

# run_lengths on a hand-checkable pattern.
_rl = run_lengths(np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool))
print(f"[{'PASS' if list(_rl) == [2, 3, 1] else 'FAIL'}] run_lengths -> {list(_rl)} (expect [2, 3, 1])")

# drawdown_episodes on a hand-checkable curve: up, -20%, recover, up.
_eq = np.array([1.0, 1.2, 1.0, 0.96, 1.1, 1.25, 1.30])
_ep = drawdown_episodes(_eq)
print(f"[{'PASS' if len(_ep) == 1 and abs(_ep.depth.iloc[0] + 0.20) < 1e-9 else 'FAIL'}] "
      f"drawdown_episodes -> {len(_ep)} episode, depth {_ep.depth.iloc[0]:.1%} (expect 1, -20.0%)")

# %%
def characterise(returns, bpy: float, label: str = "",
                 ref_threshold: float | None = None) -> dict:
    """Compute the full statistical fingerprint of one return series.

    This is THE measurement function. Real data and every synthetic path are
    passed through it unchanged, which is what makes the Section 09 comparison
    an apples-to-apples one.

    Parameters
    ----------
    returns : array-like of SIMPLE (arithmetic) returns, one per bar.
    bpy : bars per year, for annualisation.
    label : optional tag carried through into the output dict.
    ref_threshold : an ABSOLUTE |return| level defining a shock. The `shock_*`
        statistics threshold each series at its own empirical quantile, which
        makes `shock_freq` self-normalising - it comes out at roughly
        1 - SHOCK_PRIMARY_QUANTILE for every series, real or synthetic, and so
        carries no information for comparing them. Passing a fixed reference
        level (measured once on the real data) additionally populates the
        `abs_shock_*` statistics, which ARE comparable across series: a
        generator producing too few large moves shows a lower `abs_shock_freq`.

    Returns
    -------
    dict of scalar statistics. Any statistic that cannot be computed on the
    given sample is NaN rather than absent, so results stack into a DataFrame
    with stable columns.

    Assumptions: `returns` are ordered in time, > -1 (so equity stays positive),
    and long enough (>= 100 bars) for the dependence statistics to mean
    anything. Shorter series return the NaN-filled skeleton.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    out: dict[str, float | str] = {"label": label, "n_obs": len(r)}

    keys = ["mean", "std", "ann_return", "ann_vol", "skew", "kurtosis",
            "q001", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q999",
            "min", "max", "hill_left", "hill_right", "frac_beyond_3sd", "frac_beyond_5sd",
            "acf1", "acf2", "acf5", "acf10", "acf_abs1", "acf_abs5", "acf_abs10", "acf_abs20",
            "acf_sq1", "acf_sq5", "acf_sq10", "lb_p_ret", "lb_p_abs",
            "vol_of_vol", "vol_persistence", "high_vol_frac",
            "vr2", "vr5", "vr10", "vr20", "vr10_p", "hurst",
            "mean_up_run", "mean_down_run", "max_up_run", "frac_time_up",
            "mean_pos_run", "mean_neg_run",
            "max_dd", "mean_dd_depth", "median_dd_depth", "mean_dd_duration",
            "max_dd_duration", "mean_recovery_bars", "n_dd_episodes", "frac_time_underwater",
            "shock_threshold", "shock_freq", "shock_mean_mag", "shock_max_mag",
            "shock_clustering_ratio", "shock_followed_frac",
            "post_shock_vol_ratio", "pre_shock_vol_ratio", "shock_neg_frac",
            "abs_shock_freq", "abs_shock_mean_mag", "abs_shock_clustering_ratio"]
    for k in keys:
        out[k] = np.nan
    if len(r) < 100:
        return out

    lr = np.log1p(np.clip(r, -0.999999, None))

    # --- distribution ----------------------------------------------------
    out["mean"] = float(r.mean())
    out["std"] = float(r.std(ddof=1))
    out["ann_vol"] = float(r.std(ddof=1) * math.sqrt(bpy))
    out["ann_return"] = float(np.expm1(lr.mean() * bpy))
    out["skew"] = float(stats.skew(r))
    out["kurtosis"] = float(stats.kurtosis(r, fisher=True))
    for q, name in [(0.001, "q001"), (0.01, "q01"), (0.05, "q05"), (0.25, "q25"),
                    (0.50, "q50"), (0.75, "q75"), (0.95, "q95"), (0.99, "q99"),
                    (0.999, "q999")]:
        out[name] = float(np.quantile(r, q))
    out["min"], out["max"] = float(r.min()), float(r.max())

    # --- tails -----------------------------------------------------------
    out["hill_left"] = hill_tail_index(r[r < 0])
    out["hill_right"] = hill_tail_index(r[r > 0])
    sd = r.std(ddof=1)
    if sd > 0:
        out["frac_beyond_3sd"] = float(np.mean(np.abs(r - r.mean()) > 3 * sd))
        out["frac_beyond_5sd"] = float(np.mean(np.abs(r - r.mean()) > 5 * sd))

    # --- dependence ------------------------------------------------------
    a_r = acf(r, 20)
    a_abs = acf(np.abs(r), 20)
    a_sq = acf(r**2, 20)
    for lag, name in [(1, "acf1"), (2, "acf2"), (5, "acf5"), (10, "acf10")]:
        out[name] = float(a_r[lag]) if len(a_r) > lag else np.nan
    for lag, name in [(1, "acf_abs1"), (5, "acf_abs5"), (10, "acf_abs10"), (20, "acf_abs20")]:
        out[name] = float(a_abs[lag]) if len(a_abs) > lag else np.nan
    for lag, name in [(1, "acf_sq1"), (5, "acf_sq5"), (10, "acf_sq10")]:
        out[name] = float(a_sq[lag]) if len(a_sq) > lag else np.nan
    if HAVE_STATSMODELS and len(r) > 60:
        try:
            out["lb_p_ret"] = float(acorr_ljungbox(r, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
            out["lb_p_abs"] = float(acorr_ljungbox(np.abs(r), lags=[10], return_df=True)["lb_pvalue"].iloc[0])
        except Exception:
            pass

    # --- volatility ------------------------------------------------------
    w = max(5, int(REGIME_FEATURE_WINDOW))
    if len(r) > 4 * w:
        rv = pd.Series(r).rolling(w).std().dropna().to_numpy()
        if len(rv) > 30 and rv.mean() > 0:
            out["vol_of_vol"] = float(rv.std(ddof=1) / rv.mean())
            lv = np.log(np.clip(rv, 1e-12, None))
            out["vol_persistence"] = float(acf(lv, 1)[1])
            out["high_vol_frac"] = float(np.mean(rv > np.median(rv) * 1.5))

    # --- trend -----------------------------------------------------------
    for q, name in [(2, "vr2"), (5, "vr5"), (10, "vr10"), (20, "vr20")]:
        v, _z, p = variance_ratio(lr, q)
        out[name] = v
        if name == "vr10":
            out["vr10_p"] = p
    out["hurst"] = hurst_rs(lr)

    price = np.exp(np.cumsum(lr))
    ma = pd.Series(price).rolling(w).mean().to_numpy()
    valid = np.isfinite(ma)
    above = (price > ma) & valid
    below = (price < ma) & valid
    up_runs, down_runs = run_lengths(above), run_lengths(below)
    out["mean_up_run"] = float(up_runs.mean()) if len(up_runs) else np.nan
    out["mean_down_run"] = float(down_runs.mean()) if len(down_runs) else np.nan
    out["max_up_run"] = float(up_runs.max()) if len(up_runs) else np.nan
    out["frac_time_up"] = float(above[valid].mean()) if valid.any() else np.nan
    pos_runs, neg_runs = run_lengths(r > 0), run_lengths(r < 0)
    out["mean_pos_run"] = float(pos_runs.mean()) if len(pos_runs) else np.nan
    out["mean_neg_run"] = float(neg_runs.mean()) if len(neg_runs) else np.nan

    # --- drawdowns -------------------------------------------------------
    eq = np.cumprod(1.0 + r)
    dd = drawdown_series(eq)
    out["max_dd"] = float(dd.min())
    out["frac_time_underwater"] = float(np.mean(dd < -1e-12))
    ep = drawdown_episodes(eq)
    if len(ep):
        out["n_dd_episodes"] = float(len(ep))
        out["mean_dd_depth"] = float(ep["depth"].mean())
        out["median_dd_depth"] = float(ep["depth"].median())
        out["mean_dd_duration"] = float(ep["duration"].mean())
        out["max_dd_duration"] = float(ep["duration"].max())
        rec = ep.loc[ep["recovered"], "bars_to_recover"]
        out["mean_recovery_bars"] = float(rec.mean()) if len(rec) else np.nan

    # --- shocks ----------------------------------------------------------
    out.update(shock_stats(r, SHOCK_METHOD, SHOCK_THRESHOLD_SD,
                           SHOCK_PRIMARY_QUANTILE, SHOCK_RECOVERY_WINDOW))

    # Shocks against a fixed external threshold, so counts are comparable
    # between series rather than each being defined relative to itself.
    if ref_threshold is not None and ref_threshold > 0:
        a = np.abs(r)
        idx = np.where(a >= ref_threshold)[0]
        out["abs_shock_freq"] = float(len(idx) / len(r))
        if len(idx) >= 2:
            out["abs_shock_mean_mag"] = float(a[idx].mean())
            gaps = np.diff(idx)
            if gaps.mean() > 0:
                out["abs_shock_clustering_ratio"] = float(
                    (len(r) / len(idx)) / gaps.mean())
    return out


REAL_STATS = characterise(RET.to_numpy(), BPY, PRIMARY_ASSET)

_groups = {
    "distribution": ["ann_return", "ann_vol", "skew", "kurtosis", "q01", "q99", "min", "max"],
    "tails": ["hill_left", "hill_right", "frac_beyond_3sd", "frac_beyond_5sd"],
    "dependence": ["acf1", "acf5", "acf_abs1", "acf_abs10", "acf_abs20", "acf_sq1", "lb_p_ret", "lb_p_abs"],
    "volatility": ["vol_of_vol", "vol_persistence", "high_vol_frac"],
    "trend": ["vr2", "vr5", "vr10", "vr20", "vr10_p", "hurst", "mean_up_run", "mean_down_run", "frac_time_up"],
    "drawdown": ["max_dd", "mean_dd_depth", "mean_dd_duration", "max_dd_duration",
                 "mean_recovery_bars", "n_dd_episodes", "frac_time_underwater"],
    "shocks": ["shock_threshold", "shock_freq", "shock_mean_mag", "shock_clustering_ratio",
               "shock_followed_frac", "post_shock_vol_ratio", "shock_neg_frac"],
}
print(f"Statistical fingerprint - {PRIMARY_ASSET} ({REAL_STATS['n_obs']} bars)")
print("=" * 62)
for g, ks in _groups.items():
    print(f"\n{g.upper()}")
    for k in ks:
        print(f"  {k:<24} {num(REAL_STATS[k], 4)}")

# %% [markdown]
# **Reading the trend block.** `vr10` is the variance ratio at a 10-bar horizon
# with `vr10_p` its heteroskedasticity-robust p-value. Above 1 means 10-bar
# moves are larger than 10 independent 1-bar moves would be - price extends
# rather than reverting. That is the statistical footprint of a trend, and it is
# precisely the property a trend-following rule needs in order to work. If a
# generator reproduces everything else but flattens `vr10` to 1, it has built a
# market in which no trend strategy could ever work, and comparing against it
# would be rigged.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
_lags = 30
for ax, (series, name, col) in zip(axes, [
        (RET.to_numpy(), "returns", C["blue"]),
        (np.abs(RET.to_numpy()), "|returns|", C["orange"]),
        (RET.to_numpy() ** 2, "returns squared", C["aqua"])]):
    a = acf(series, _lags)
    ax.bar(range(1, _lags + 1), a[1:], color=col, width=0.72)
    ci = 1.96 / math.sqrt(len(series))
    ax.axhspan(-ci, ci, color=INK["grid"], alpha=0.9, zorder=0)
    ax.axhline(0, color=INK["axis"], lw=0.9)
    finish(ax, f"ACF of {name}", xlabel="lag (bars)",
           ylabel="autocorrelation" if name == "returns" else None, legend=False)
axes[0].text(0.98, 0.94, "shaded band = 95% CI under independence",
             transform=axes[0].transAxes, ha="right", va="top",
             fontsize=8, color=INK["muted"])
fig.tight_layout()
plt.show()

print("The contrast between the first panel and the other two is the single most")
print("important fact about financial returns:")
print(f"  ACF(1) of returns      = {REAL_STATS['acf1']:+.4f}  (near zero - direction is ~unpredictable)")
print(f"  ACF(1) of |returns|    = {REAL_STATS['acf_abs1']:+.4f}  (large and positive)")
print(f"  ACF(20) of |returns|   = {REAL_STATS['acf_abs20']:+.4f}  (still positive 20 bars out)")
print("Direction barely autocorrelates, but MAGNITUDE strongly does, and decays")
print("slowly. That is volatility clustering, and it is the property an IID")
print("bootstrap destroys completely - which is exactly why we test against one.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

ax = axes[0]
_up = run_lengths((PX["close"] > PX["close"].rolling(REGIME_FEATURE_WINDOW).mean()).to_numpy())
_dn = run_lengths((PX["close"] < PX["close"].rolling(REGIME_FEATURE_WINDOW).mean()).to_numpy())
_bins = np.arange(0.5, 61.5, 2)
ax.hist(_up, bins=_bins, color=C["blue"], alpha=0.8, label=f"above MA (mean {_up.mean():.1f})")
ax.hist(_dn, bins=_bins, color=C["orange"], alpha=0.65, label=f"below MA (mean {_dn.mean():.1f})")
ax.set_yscale("log")
finish(ax, "Trend run lengths", xlabel=f"consecutive bars vs {REGIME_FEATURE_WINDOW}-bar MA",
       ylabel="count (log)")

ax = axes[1]
_ep = drawdown_episodes(np.cumprod(1 + RET.to_numpy()))
ax.scatter(_ep["duration"], _ep["depth"] * 100, s=14, alpha=0.5, color=C["blue"],
           label="drawdown episode")
ax.set_xscale("log")
finish(ax, "Drawdown depth vs duration", xlabel="duration (bars, log)", ylabel="depth (%)",
       legend=False, subtitle=f"{len(_ep)} episodes; worst {_ep['depth'].min():.1%}")

ax = axes[2]
_a = np.abs(RET.to_numpy())
_thr = REAL_STATS["shock_threshold"]
_sh = np.where(_a >= _thr)[0]
ax.plot(np.arange(len(_a)), _a * 100, color=INK["grid"], lw=0.6)
ax.scatter(_sh, _a[_sh] * 100, s=9, color=STATUS["critical"], zorder=3,
           label=f"shock (|r| >= {_thr:.2%})")
finish(ax, "Shock timing", xlabel="bar index", ylabel="|return| (%)")

fig.tight_layout()
plt.show()

print(f"shock frequency        {REAL_STATS['shock_freq']:.3%} of bars")
print(f"clustering ratio       {REAL_STATS['shock_clustering_ratio']:.2f}  "
      f"(>1 means shocks arrive closer together than a memoryless process)")
print(f"post-shock vol ratio   {REAL_STATS['post_shock_vol_ratio']:.2f}  "
      f"(volatility in the {SHOCK_RECOVERY_WINDOW} bars after a shock, vs average)")
print("Shocks are not isolated accidents - they bunch, and they leave elevated")
print("volatility behind them. A generator that sprinkles jumps uniformly at")
print("random will miss both effects.")

# %%
# Characterise every loaded asset - the cross-sectional view.
ALL_STATS = pd.DataFrame([
    characterise(simple_returns(df).to_numpy(), infer_bars_per_year(df.index), name)
    for name, df in sorted(DATA.items())
]).set_index("label")

_show = ["n_obs", "ann_return", "ann_vol", "skew", "kurtosis", "hill_left",
         "acf1", "acf_abs1", "vr10", "vr10_p", "hurst", "max_dd", "shock_clustering_ratio"]
print("Cross-asset fingerprint")
print(ALL_STATS[_show].round(4).to_string())

_trending = ALL_STATS[(ALL_STATS["vr10"] > 1) & (ALL_STATS["vr10_p"] < 0.05)]
print(f"\n{len(_trending)} of {len(ALL_STATS)} assets show statistically significant "
      f"trend structure at a 10-bar horizon (VR>1, p<0.05): {list(_trending.index)}")
print("This is measured on the price data alone - no strategy involved. It is the")
print("prior question: is there anything here for a trend rule to find?")

# %% [markdown]
# ---
# ## 06 - Train / validation / test structure
#
# Financial series are never shuffled. A random split would put bar *t+1* in
# train and bar *t* in test, and since volatility clusters and trends persist,
# the model would effectively be told the answer. Every split here is
# chronological.
#
# Three things are defined:
#
# 1. **A single chronological holdout** - train / validation / test in time
#    order, used for the honest split of labour: fit on train, choose on
#    validation, look at test exactly once.
# 2. **Walk-forward splits**, expanding and rolling. Expanding keeps all history
#    (more data, but the early regime never leaves). Rolling keeps a fixed
#    window (adapts to regime change, but discards history).
# 3. **An embargo** - a gap of `WALK_FORWARD_EMBARGO` bars dropped between the
#    end of train and the start of test.
#
# The embargo matters more than it looks. Any feature computed over a trailing
# window (a moving average, a rolling volatility) makes the first few test bars
# depend on the last few train bars. Without a gap, that overlap is a real, if
# small, leak. Dropping a handful of bars costs almost nothing and removes it.

# %%
@dataclass(frozen=True)
class Split:
    """One train/test (optionally validation) index range over a series.

    Indices are positional (iloc), half-open on the right, and always satisfy
    train < validation < test in time with no overlap.
    """
    name: str
    train: tuple[int, int]
    test: tuple[int, int]
    validation: tuple[int, int] | None = None

    def slice_of(self, which: str) -> slice:
        rng = getattr(self, which)
        if rng is None:
            raise ValueError(f"split {self.name!r} has no {which} range")
        return slice(rng[0], rng[1])

    def describe(self, index: pd.DatetimeIndex) -> str:
        def fmt(rng):
            if rng is None:
                return "-"
            return f"{index[rng[0]].date()}..{index[rng[1]-1].date()} ({rng[1]-rng[0]})"
        return (f"{self.name:<18} train {fmt(self.train)}  "
                f"val {fmt(self.validation)}  test {fmt(self.test)}")


def chronological_split(n: int, train_frac: float, val_frac: float,
                        test_frac: float, embargo: int = 0) -> Split:
    """A single in-time-order train/validation/test partition."""
    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    if n < 100:
        raise ValueError(f"need at least 100 observations to split, got {n}")
    i_tr = int(n * train_frac)
    i_va = i_tr + int(n * val_frac)
    if embargo:
        i_tr_end = max(1, i_tr - embargo)
        i_va_end = max(i_va - embargo, i_va - embargo)
    else:
        i_tr_end, i_va_end = i_tr, i_va
    return Split("holdout", train=(0, i_tr_end), validation=(i_tr, i_va_end), test=(i_va, n))


def walk_forward_splits(n: int, n_folds: int, min_train: int, embargo: int,
                        scheme: str = "expanding") -> list[Split]:
    """Sequential train->test folds marching forward through time.

    scheme='expanding' - train always starts at bar 0 and grows.
    scheme='rolling'   - train is a fixed-length window that slides.

    Each fold's test period is disjoint from every other fold's test period, so
    concatenating the fold test results gives one continuous out-of-sample
    track record with no bar counted twice.
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError(f"scheme must be 'expanding' or 'rolling', got {scheme!r}")
    usable = n - min_train
    if usable < n_folds * 20:
        raise ValueError(
            f"cannot build {n_folds} folds: only {usable} bars after reserving "
            f"min_train={min_train}; need >= {n_folds*20}. Reduce WALK_FORWARD_N_FOLDS "
            f"or WALK_FORWARD_MIN_TRAIN.")
    test_len = usable // n_folds
    splits = []
    for k in range(n_folds):
        test_lo = min_train + k * test_len
        test_hi = n if k == n_folds - 1 else test_lo + test_len
        train_hi = max(1, test_lo - embargo)
        train_lo = 0 if scheme == "expanding" else max(0, train_hi - min_train)
        splits.append(Split(f"{scheme}_{k+1}", train=(train_lo, train_hi), test=(test_lo, test_hi)))
    return splits


N_BARS = len(RET)
HOLDOUT = chronological_split(N_BARS, TRAIN_FRAC, VALIDATION_FRAC, TEST_FRAC, WALK_FORWARD_EMBARGO)
WF_EXPANDING = walk_forward_splits(N_BARS, WALK_FORWARD_N_FOLDS, WALK_FORWARD_MIN_TRAIN,
                                   WALK_FORWARD_EMBARGO, "expanding")
WF_ROLLING = walk_forward_splits(N_BARS, WALK_FORWARD_N_FOLDS, WALK_FORWARD_MIN_TRAIN,
                                 WALK_FORWARD_EMBARGO, "rolling")

_ridx = RET.index
print(f"{PRIMARY_ASSET}: {N_BARS} return observations, embargo = {WALK_FORWARD_EMBARGO} bars\n")
print(HOLDOUT.describe(_ridx))
print("\nexpanding walk-forward:")
for s in WF_EXPANDING:
    print("  " + s.describe(_ridx))
print("\nrolling walk-forward:")
for s in WF_ROLLING:
    print("  " + s.describe(_ridx))

# Assert the no-overlap property rather than trusting the arithmetic.
for _scheme, _folds in [("expanding", WF_EXPANDING), ("rolling", WF_ROLLING)]:
    for a, b in zip(_folds, _folds[1:]):
        assert a.test[1] <= b.test[0], f"{_scheme}: test windows overlap"
    for s in _folds:
        assert s.train[1] <= s.test[0], f"{_scheme}: train runs into test in {s.name}"
        assert s.test[0] - s.train[1] >= min(WALK_FORWARD_EMBARGO, s.test[0]), \
            f"{_scheme}: embargo not respected in {s.name}"
assert HOLDOUT.train[1] <= HOLDOUT.validation[0] <= HOLDOUT.validation[1] <= HOLDOUT.test[0]
print("\n[PASS] all splits are chronological, non-overlapping, and respect the embargo")

# %%
fig, ax = plt.subplots(figsize=(11, 3.6))
_rows = [("holdout", HOLDOUT)] + [(s.name, s) for s in WF_EXPANDING] + \
        [(s.name, s) for s in WF_ROLLING]
_role_color = {"train": C["blue"], "validation": C["yellow"], "test": C["orange"]}
for y, (nm, s) in enumerate(_rows):
    for role in ("train", "validation", "test"):
        rng = getattr(s, role)
        if rng is None:
            continue
        ax.barh(y, rng[1] - rng[0], left=rng[0], height=0.62,
                color=_role_color[role], edgecolor=INK["surface"], linewidth=1.2)
ax.set_yticks(range(len(_rows)))
ax.set_yticklabels([nm for nm, _ in _rows], fontsize=8)
ax.invert_yaxis()
for role, col in _role_color.items():
    ax.barh([], [], color=col, label=role)
finish(ax, "Chronological split structure", xlabel="bar index (time ->)",
       subtitle="gaps between train and test are the embargo")
ax.grid(axis="y", visible=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 07 - Regime detection
#
# Markets alternate between states - quiet uptrends, violent selloffs, directionless
# chop - and a generator that ignores this produces a homogeneous market that
# never has a crisis and never has a calm decade.
#
# Three methods are fitted and compared, because no single regime methodology is
# obviously right and pretending otherwise would be the kind of unexamined
# assumption this notebook is supposed to avoid:
#
# | method | how it decides | strength | weakness |
# |---|---|---|---|
# | `quantile` | buckets trailing vol, splits on trailing trend sign | transparent, always converges, no fitting | boundaries are arbitrary; no persistence model |
# | `gmm` | Gaussian mixture on (trend, vol) features | learns the clusters from data | treats bars as independent - no persistence |
# | `hmm` | Gaussian hidden Markov model on returns | models persistence explicitly; gives a transition matrix | EM can hit local optima; assumes Gaussian emissions |
#
# The HMM is the one the generators consume, because it is the only one of the
# three that produces a **transition matrix** - and a transition matrix is what
# lets Section 08 generate *new* regime sequences rather than replaying the
# historical one.
#
# **On leakage.** Regime features are strictly trailing (a rolling window that
# ends at the current bar), so labelling bar *t* uses no information from after
# *t*. Separately, when `GENERATOR_CALIBRATION_SLICE="train"` the regime model
# that feeds the generators is fitted on the training slice only.

# %%
def regime_features(returns: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing trend and volatility features, one row per usable bar.

    Both features use a window ENDING at the current bar, so no future
    information enters the label for that bar.

    Returns (features [n_valid, 2], valid_positions) where the columns are
    (mean return over the window, sd of return over the window), each
    standardised to zero mean and unit variance.
    """
    r = pd.Series(np.asarray(returns, dtype=float))
    trend = r.rolling(window).mean()
    vol = r.rolling(window).std()
    valid = trend.notna() & vol.notna() & np.isfinite(trend) & np.isfinite(vol)
    pos = np.where(valid.to_numpy())[0]
    X = np.column_stack([trend.to_numpy()[pos], vol.to_numpy()[pos]])
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    return X, pos


def fit_regimes_quantile(returns, window, n_regimes) -> dict:
    """Split on trailing trend sign x trailing volatility buckets.

    Requires an even `n_regimes`: the states are (n_regimes/2) volatility
    buckets crossed with {downtrend, uptrend}. Fails loudly rather than
    silently reinterpreting an odd request.
    """
    if n_regimes % 2 != 0:
        raise ValueError(
            f"the 'quantile' method builds vol-buckets x trend-sign and so needs an "
            f"even N_REGIMES; got {n_regimes}. Use 'gmm' or 'hmm' for odd counts.")
    n_vol = n_regimes // 2
    X, pos = regime_features(returns, window)
    trend, vol = X[:, 0], X[:, 1]
    edges = np.quantile(vol, np.linspace(0, 1, n_vol + 1)[1:-1])
    vol_bucket = np.searchsorted(edges, vol)
    labels = vol_bucket * 2 + (trend > 0).astype(int)
    return {"method": "quantile", "labels": labels, "positions": pos,
            "n_regimes": n_regimes, "converged": True, "loglik": np.nan, "bic": np.nan}


def fit_regimes_gmm(returns, window, n_regimes, seed) -> dict:
    """Gaussian mixture over the (trend, vol) feature pairs.

    Note the structural limitation: a mixture model has no notion of time, so it
    can say 'this bar looks like a crisis bar' but cannot say 'crises last about
    three weeks'. That is why it is compared, not used, for generation.
    """
    if not HAVE_SKLEARN:
        return {"method": "gmm", "converged": False, "error": "scikit-learn not installed"}
    X, pos = regime_features(returns, window)
    g = GaussianMixture(n_components=n_regimes, covariance_type="full",
                        n_init=8, random_state=seed, max_iter=500)
    labels = g.fit_predict(X)
    return {"method": "gmm", "labels": labels, "positions": pos, "n_regimes": n_regimes,
            "converged": bool(g.converged_), "loglik": float(g.score(X) * len(X)),
            "bic": float(g.bic(X)), "model": g}


def _gaussian_logpdf(x, mu, sigma):
    """Log N(x | mu, sigma) evaluated for every (observation, state) pair."""
    sigma = np.maximum(sigma, 1e-8)
    z = (x[:, None] - mu[None, :]) / sigma[None, :]
    return -0.5 * z**2 - np.log(sigma)[None, :] - 0.5 * math.log(2 * math.pi)


def fit_gaussian_hmm(x, n_states, seed, n_restarts=8, max_iter=200, tol=1e-6):
    """Baum-Welch (EM) fit of a univariate Gaussian hidden Markov model.

    Hand-rolled rather than taken from a library for three reasons: it removes a
    dependency from a load-bearing path, it makes the model's assumptions
    visible and auditable, and it can be verified against data simulated from a
    known model (which is done immediately below).

    The forward-backward pass uses per-timestep scaling rather than logs, which
    is numerically stable here and materially faster.

    Parameters
    ----------
    x : 1-D observations (returns, already scaled to a convenient magnitude).
    n_states : number of hidden regimes.
    n_restarts : random initialisations; the best log-likelihood wins, which
        mitigates but does not eliminate EM's local-optimum problem.

    Returns dict with pi, A (transition matrix), mu, sigma, loglik, bic,
    gamma (posterior state probabilities) and `converged`.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T, K = len(x), n_states
    if T < 50 * K:
        raise ValueError(f"need >= {50*K} observations to fit a {K}-state HMM, got {T}")
    rng = np.random.default_rng(seed)
    best = None

    for restart in range(n_restarts):
        # Initialise from quantile slices so states start separated.
        qs = np.quantile(x, np.linspace(0, 1, K + 1))
        mu = np.array([x[(x >= qs[k]) & (x <= qs[k + 1])].mean() if
                       np.any((x >= qs[k]) & (x <= qs[k + 1])) else x.mean()
                       for k in range(K)])
        mu = mu + rng.normal(0, x.std() * 0.1, K) if restart else mu
        sigma = np.full(K, x.std()) * rng.uniform(0.5, 1.5, K)
        A = np.full((K, K), 0.1 / max(1, K - 1))
        np.fill_diagonal(A, 0.9)
        A = A / A.sum(axis=1, keepdims=True)
        pi = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        converged = False
        for _ in range(max_iter):
            logB = _gaussian_logpdf(x, mu, sigma)
            # Shift per timestep before exponentiating: keeps B in range without
            # changing the posteriors, and the shift cancels out of the scaling.
            shift = logB.max(axis=1, keepdims=True)
            B = np.exp(logB - shift)

            alpha = np.empty((T, K))
            c = np.empty(T)
            alpha[0] = pi * B[0]
            c[0] = alpha[0].sum()
            if c[0] <= 0:
                break
            alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[t]
                c[t] = alpha[t].sum()
                if c[t] <= 0:
                    c[t] = 1e-300
                alpha[t] /= c[t]

            beta = np.empty((T, K))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

            gamma = alpha * beta
            gsum = gamma.sum(axis=1, keepdims=True)
            gamma = gamma / np.where(gsum > 0, gsum, 1.0)

            xi_num = np.zeros((K, K))
            for t in range(T - 1):
                xi_num += (alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]) / c[t + 1]

            loglik = float(np.sum(np.log(c)) + np.sum(shift))
            pi = gamma[0] / gamma[0].sum()
            A = xi_num / np.where(xi_num.sum(axis=1, keepdims=True) > 0,
                                  xi_num.sum(axis=1, keepdims=True), 1.0)
            w = gamma.sum(axis=0)
            w = np.where(w > 1e-12, w, 1e-12)
            mu = (gamma * x[:, None]).sum(axis=0) / w
            var = (gamma * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / w
            sigma = np.sqrt(np.maximum(var, 1e-12))

            if abs(loglik - prev_ll) < tol * max(1.0, abs(prev_ll)):
                converged = True
                break
            prev_ll = loglik

        if best is None or loglik > best["loglik"]:
            # States are unidentifiable up to permutation; sort by mean so the
            # labelling is stable across restarts and across calls.
            order = np.argsort(mu)
            best = {"pi": pi[order], "A": A[np.ix_(order, order)], "mu": mu[order],
                    "sigma": sigma[order], "loglik": loglik, "gamma": gamma[:, order],
                    "converged": converged, "n_states": K, "n_obs": T}

    n_params = K - 1 + K * (K - 1) + 2 * K
    best["bic"] = float(n_params * math.log(best["n_obs"]) - 2 * best["loglik"])
    return best


def hmm_viterbi(x, model) -> np.ndarray:
    """Most likely hidden state sequence (Viterbi), computed in log space."""
    x = np.asarray(x, dtype=float)
    K = model["n_states"]
    logB = _gaussian_logpdf(x, model["mu"], model["sigma"])
    logA = np.log(np.maximum(model["A"], 1e-300))
    T = len(x)
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = np.log(np.maximum(model["pi"], 1e-300)) + logB[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + logA
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(K)] + logB[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


print("regime methods defined")

# %% [markdown]
# ### Verifying the HMM against a known model
#
# An EM fit that silently converges to nonsense is the most dangerous kind of
# bug, because it produces plausible-looking numbers. So before the HMM is used
# on real data it is asked to recover parameters from data generated by a model
# whose true parameters are known. If it cannot recover those, nothing
# downstream of it can be trusted.

# %%
def _simulate_hmm(pi, A, mu, sigma, n, rng):
    """Draw a sequence from a known Gaussian HMM (the ground truth for the test)."""
    K = len(pi)
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, n):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    return mu[states] + sigma[states] * rng.standard_normal(n), states


_rng_hmm = get_rng("hmm_recovery_test")
_true_A = np.array([[0.97, 0.03], [0.06, 0.94]])
_true_mu = np.array([-0.15, 0.08])
_true_sigma = np.array([1.60, 0.55])
_true_pi = np.array([0.4, 0.6])

_x_sim, _s_sim = _simulate_hmm(_true_pi, _true_A, _true_mu, _true_sigma, 6000, _rng_hmm)
_fit = fit_gaussian_hmm(_x_sim, 2, seed=11, n_restarts=6)

_mu_err = np.max(np.abs(_fit["mu"] - _true_mu))
_sd_err = np.max(np.abs(_fit["sigma"] - _true_sigma))
_A_err = np.max(np.abs(_fit["A"] - _true_A))
_path = hmm_viterbi(_x_sim, _fit)
_acc = max((_path == _s_sim).mean(), (_path == (1 - _s_sim)).mean())

print("HMM parameter recovery (2 states, 6000 simulated bars)")
print(f"  true  mu    {np.round(_true_mu,3)}   sigma {np.round(_true_sigma,3)}")
print(f"  fitted mu   {np.round(_fit['mu'],3)}   sigma {np.round(_fit['sigma'],3)}")
print(f"  true  A     {np.round(_true_A,3).tolist()}")
print(f"  fitted A    {np.round(_fit['A'],3).tolist()}")
print(f"  converged={_fit['converged']}  loglik={_fit['loglik']:.1f}")
print(f"\n  [{'PASS' if _mu_err < 0.06 else 'FAIL'}] mean recovery      max err {_mu_err:.4f} (tol 0.06)")
print(f"  [{'PASS' if _sd_err < 0.08 else 'FAIL'}] sigma recovery     max err {_sd_err:.4f} (tol 0.08)")
print(f"  [{'PASS' if _A_err  < 0.05 else 'FAIL'}] transition recovery max err {_A_err:.4f} (tol 0.05)")
print(f"  [{'PASS' if _acc   > 0.90 else 'FAIL'}] Viterbi state accuracy {_acc:.3f} (tol 0.90)")

if _mu_err > 0.06 or _A_err > 0.05 or _acc < 0.90:
    raise AssertionError("HMM failed parameter recovery - do not trust regime-based generators")

# %%
# Fit all three methods on the calibration slice and compare them.
CALIB_END = HOLDOUT.train[1] if GENERATOR_CALIBRATION_SLICE == "train" else N_BARS
CALIB_RET = RET.to_numpy()[:CALIB_END]
print(f"calibration slice = {GENERATOR_CALIBRATION_SLICE}: bars 0..{CALIB_END} "
      f"({RET.index[0].date()} .. {RET.index[CALIB_END-1].date()})")
print(f"the remaining {N_BARS - CALIB_END} bars are NOT seen by any generator\n")


def regime_summary(labels, positions, returns, n_regimes, bpy) -> pd.DataFrame:
    """Per-regime descriptive statistics plus persistence, from a label array."""
    r = np.asarray(returns)[positions]
    rows = []
    for k in range(n_regimes):
        m = labels == k
        if not m.any():
            rows.append({"regime": k, "share": 0.0, "mean_ret_ann": np.nan,
                         "vol_ann": np.nan, "persistence": np.nan, "mean_duration": np.nan})
            continue
        runs = run_lengths(m)
        rows.append({
            "regime": k, "share": float(m.mean()),
            "mean_ret_ann": float(r[m].mean() * bpy),
            "vol_ann": float(r[m].std(ddof=1) * math.sqrt(bpy)) if m.sum() > 2 else np.nan,
            "persistence": float(np.mean(labels[1:][m[:-1]] == labels[:-1][m[:-1]]))
                            if m[:-1].any() else np.nan,
            "mean_duration": float(runs.mean()) if len(runs) else np.nan,
        })
    return pd.DataFrame(rows)


REGIME_FITS = {}
for _method in REGIME_METHODS:
    try:
        if _method == "quantile":
            _fit_r = fit_regimes_quantile(CALIB_RET, REGIME_FEATURE_WINDOW, N_REGIMES)
        elif _method == "gmm":
            _fit_r = fit_regimes_gmm(CALIB_RET, REGIME_FEATURE_WINDOW, N_REGIMES, seed=13)
        elif _method == "hmm":
            _m = fit_gaussian_hmm(CALIB_RET * 100.0, N_REGIMES, seed=17, n_restarts=8)
            _lab = hmm_viterbi(CALIB_RET * 100.0, _m)
            _fit_r = {"method": "hmm", "labels": _lab,
                      "positions": np.arange(len(CALIB_RET)), "n_regimes": N_REGIMES,
                      "converged": _m["converged"], "loglik": _m["loglik"],
                      "bic": _m["bic"], "model": _m}
        else:
            raise ValueError(f"unknown regime method {_method!r}")

        if not _fit_r.get("converged", False) and "error" not in _fit_r:
            print(f"  WARNING: {_method} did not converge; results reported but treat with caution")
        if "error" in _fit_r:
            print(f"  SKIPPED {_method}: {_fit_r['error']}")
            continue
        REGIME_FITS[_method] = _fit_r
    except Exception as exc:
        print(f"  FAILED {_method}: {type(exc).__name__}: {exc}")

print(f"\nfitted {len(REGIME_FITS)} regime model(s): {list(REGIME_FITS)}\n")
for _method, _f in REGIME_FITS.items():
    _summ = regime_summary(_f["labels"], _f["positions"], CALIB_RET, N_REGIMES, BPY)
    print(f"--- {_method}  (BIC {num(_f.get('bic'), 1)}, "
          f"mean persistence {_summ['persistence'].mean():.3f})")
    print(_summ.round(4).to_string(index=False))
    print()

if REGIME_PRIMARY_METHOD not in REGIME_FITS:
    _alt = next(iter(REGIME_FITS))
    print(f"REGIME_PRIMARY_METHOD={REGIME_PRIMARY_METHOD!r} unavailable; "
          f"generators will use {_alt!r} instead.")
    REGIME_PRIMARY_METHOD = _alt
    CONFIG["regimes"]["primary_method"] = _alt

PRIMARY_REGIMES = REGIME_FITS[REGIME_PRIMARY_METHOD]
print(f"generators will consume the {REGIME_PRIMARY_METHOD!r} regimes")

# %% [markdown]
# **Why persistence is the column that matters.** A regime model whose states
# last a bar or two has not found regimes - it has found noise and given it
# names. `mean_duration` and `persistence` are the check.
#
# Read the HMM's row for state 0 in the table above with that in mind. On
# fat-tailed data a Gaussian HMM will typically spend one of its states
# absorbing outliers: it comes out with a huge variance, an extreme mean, and an
# expected duration of about one bar. That state is not a regime. It is a
# **jump component** - the model's way of admitting that Gaussian emissions
# cannot cover the tails on their own. The diagnostic printed below the
# transition matrix flags any such state explicitly, because reading it as
# "the market spends 2% of its time in a -500%-annualised regime" would be
# nonsense.
#
# This is also why the HMM is the method the generators consume, and it is *not*
# because it wins on average persistence - as the table shows, it does not. It
# is because it is the only one of the three that estimates a **transition
# matrix**, and a transition matrix is what lets Section 08 generate genuinely
# new regime sequences instead of replaying the historical one. The remaining
# long-lived HMM states do carry realistic persistence (see the expected
# durations), and the short-lived one contributes the shocks.

# %%
_f = PRIMARY_REGIMES
_lab_full = np.full(len(CALIB_RET), -1)
_lab_full[_f["positions"]] = _f["labels"]

fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
_dates = RET.index[:CALIB_END]
_price = PX["close"].to_numpy()[1:CALIB_END + 1]

ax = axes[0]
ax.plot(_dates, _price, color=INK["secondary"], lw=1.0, zorder=3)
ax.set_yscale("log")
_summ = regime_summary(_f["labels"], _f["positions"], CALIB_RET, N_REGIMES, BPY)
for k in range(N_REGIMES):
    m = _lab_full == k
    if not m.any():
        continue
    ax.fill_between(_dates, _price.min(), _price.max(), where=m,
                    color=SERIES[k % len(SERIES)], alpha=0.20, lw=0,
                    label=f"regime {k}: vol {_summ.vol_ann.iloc[k]:.0%}, "
                          f"ret {_summ.mean_ret_ann.iloc[k]:+.0%} ann")
finish(ax, f"{PRIMARY_ASSET} regimes ({REGIME_PRIMARY_METHOD}, calibration slice)",
       ylabel="close (log)", subtitle="shading = most likely regime for that bar")

ax = axes[1]
ax.plot(_dates, pd.Series(CALIB_RET, index=_dates).rolling(REGIME_FEATURE_WINDOW).std()
        * np.sqrt(BPY), color=C["orange"], lw=1.1)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
finish(ax, None, xlabel="date", ylabel=f"{REGIME_FEATURE_WINDOW}-bar vol (ann.)", legend=False)
fig.tight_layout()
plt.show()

if "model" in _f and "A" in _f.get("model", {}):
    print("Estimated transition matrix P(row -> column):")
    _A = _f["model"]["A"]
    print("        " + "".join(f"   to {j}  " for j in range(N_REGIMES)))
    for i in range(N_REGIMES):
        print(f"  from {i}  " + "".join(f"  {_A[i,j]:.3f} " for j in range(N_REGIMES)))
    print(f"\nExpected duration of each state (1/(1-P_ii)), in bars:")
    _jump_states = []
    for i in range(N_REGIMES):
        _d = 1.0 / max(1e-9, 1.0 - _A[i, i])
        _kind = "JUMP component" if _d < 3.0 else "regime"
        if _d < 3.0:
            _jump_states.append(i)
        print(f"  state {i}: {_d:7.1f} bars  ({_d/BPY*12:6.1f} months)   <- {_kind}")

    if _jump_states:
        print(f"\nDIAGNOSTIC: state(s) {_jump_states} have an expected duration under 3")
        print("bars. These are not regimes. A Gaussian HMM cannot represent fat tails")
        print("with its emission densities alone, so EM allocates a high-variance,")
        print("low-persistence state to absorb the outliers. Read them as the model's")
        print("jump component - which is useful here, because it is exactly what gives")
        print("the regime generator its shocks. Do NOT read their annualised means as")
        print("a description of any sustained market state.")
        _persist = [i for i in range(N_REGIMES) if i not in _jump_states]
        print(f"The persistent regimes are {_persist}, with durations "
              f"{[round(1.0/max(1e-9,1.0-_A[i,i]),1) for i in _persist]} bars.")

    print("\nThis matrix is what Section 08 samples to build NEW regime sequences.")
    print("The synthetic market will have crises and calm stretches of realistic")
    print("length, in a different order than history happened to deliver them.")
