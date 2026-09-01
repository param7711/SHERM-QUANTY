"""
Rolling linear-regression channel — the regime filter under test.

At every bar t the channel is fitted on the trailing window
close[t-N+1 .. t] only, and every quantity is read at the *endpoint* of
that fit. Nothing from bar t+1 onward touches bar t's numbers, so the
signal series is causal by construction.

    center_t : value of the fitted line at t
    sigma_t  : standard deviation of the fit residuals in the window
    z_t      : (close_t - center_t) / sigma_t      -> where price sits
    smaz_t   : (sma_t   - center_t) / sigma_t      -> where the SMA sits
    slope_t  : per-bar drift of the fitted line

The band at +/- k sigma is therefore |z| = k, and "the 20-SMA broke above
the channel" is smaz > k.
"""

import numpy as np
import pandas as pd


def regression_channel(close: pd.Series, n: int) -> pd.DataFrame:
    """Rolling OLS of close on bar index over a trailing window of n bars.

    Computed on explicitly centred windows rather than from raw rolling
    sums: on price levels near 1e3-1e4 the sum-of-squares shortcut loses
    most of its significant digits to cancellation, and sigma is exactly
    the quantity that cancellation destroys.
    """
    y = close.to_numpy(dtype=float)
    t = y.shape[0]
    if t < n:
        raise ValueError(f'need at least {n} bars, got {t}')

    win = np.lib.stride_tricks.sliding_window_view(y, n)     # (t-n+1, n)
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = (xc * xc).sum()

    y_bar = win.mean(axis=1)
    yc = win - y_bar[:, None]
    slope = (yc @ xc) / sxx
    resid = yc - slope[:, None] * xc[None, :]
    sigma_v = np.sqrt((resid * resid).sum(axis=1) / (n - 2))
    center_v = y_bar + slope * xc[-1]

    center = np.full(t, np.nan)
    sigma = np.full(t, np.nan)
    slope_full = np.full(t, np.nan)
    center[n - 1:] = center_v
    sigma[n - 1:] = np.where(sigma_v > 0, sigma_v, np.nan)
    slope_full[n - 1:] = slope

    return pd.DataFrame({
        'center': center,
        'sigma':  sigma,
        'slope':  slope_full,
        'z':      (y - center) / sigma,
    }, index=close.index)


def channel_features(close: pd.Series, n: int, sma_len: int) -> pd.DataFrame:
    """Channel plus the SMA's position inside it."""
    ch = regression_channel(close, n)
    sma = close.rolling(sma_len).mean()
    ch['sma'] = sma
    ch['smaz'] = (sma - ch['center']) / ch['sigma']
    return ch


def _self_check():
    """A pure straight line must give ~zero residual and exact slope."""
    idx = pd.date_range('2020-01-01', periods=300, freq='B')
    line = pd.Series(100 + 0.5 * np.arange(300), index=idx)
    ch = regression_channel(line, 100)
    assert np.allclose(ch['slope'].dropna(), 0.5), ch['slope'].dropna().head()
    # Zero residual is degenerate and is deliberately mapped to NaN so that
    # z never divides by zero.
    assert ch['sigma'].iloc[99:].isna().all()

    # A known noisy case: compare against numpy.polyfit on the last window.
    rng = np.random.default_rng(0)
    noisy = pd.Series(100 + 0.2 * np.arange(300) + rng.normal(0, 2, 300), index=idx)
    n = 120
    ch = regression_channel(noisy, n)
    w = noisy.to_numpy()[-n:]
    b, a = np.polyfit(np.arange(n), w, 1)
    fit = a + b * np.arange(n)
    ref_sigma = np.sqrt(((w - fit) ** 2).sum() / (n - 2))
    assert abs(ch['slope'].iloc[-1] - b) < 1e-9
    assert abs(ch['center'].iloc[-1] - fit[-1]) < 1e-8
    assert abs(ch['sigma'].iloc[-1] - ref_sigma) < 1e-8

    # Causality: truncating the future must not change past values.
    full = regression_channel(noisy, n)
    trunc = regression_channel(noisy.iloc[:250], n)
    assert np.allclose(full['z'].iloc[:250].dropna(),
                       trunc['z'].dropna(), equal_nan=True)
    print('channel.py self-check passed (slope, center, sigma, causality)')


if __name__ == '__main__':
    _self_check()
