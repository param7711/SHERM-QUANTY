"""
Builds stability_lag.ipynb -- a standalone notebook that MEASURES and ATTACKS the
two defects named in STABILITY_LAG_PROTOCOL.md:

    S  fit stability   = mean pairwise label agreement across R=4 DISJOINT seed sets
    L  transition lag  = bars from a ZigZag swing start to the first correctly
                         directed emitted label (median and p75)
    W  whipsaw         = switches per 100 bars  (the anti-gaming guard on L)

Design rules (from the frozen protocol + the coordinator's scope addition):
  * S, L and W are reported TOGETHER on every row. Baseline is always row 1.
  * BAR_DIR_WEIGHT is FIXED at 0.0 everywhere. Not swept, asserted.
  * The ZigZag deliberately uses future prices -- it is a RETROSPECTIVE metric.
    A three-part tripwire asserts no ZigZag output can reach `label_bars`.
  * Guards G1..G4 per candidate, printed pass/fail. G4 is DRIVEN by a degenerate
    all-SIDEWAYS stub and asserted to VOID it.
  * NO composite score. A winner is declared ONLY on strict dominance.
  * Config/capacity arms are first-class (coordinator scope addition), carrying
    the SAME S/L/W triple, the SAME guards, plus rows-per-parameter.

The engine is INLINED VERBATIM out of build_master_notebook_v2.py, the same
practice build_hmm_share_charts.py uses. Nothing here imports or modifies the
master, its generator, regime_scorecard.py or build_hmm_share_charts.py.
"""
import json
import os

HERE = '/tmp/claude-0/-home-user-SHERM-QUANTY/f1a44349-dfe0-5e0f-9b1f-b0b39e930e1b/scratchpad'
OUT = f'{HERE}/stability_lag.ipynb'
MASTER_GEN = f'{HERE}/build_master_notebook_v2.py'

# ---------------------------------------------------------------------------
# Harvest the engine cells from the master generator, VERBATIM.
# Same technique as build_hmm_share_charts.py: the generator's cell bodies are
# Python string literals with runtime concatenation, so they are obtained by
# EXECUTING the generator in a private namespace (output path redirected to a
# throwaway file) and reading its assembled `cells` list.
# ---------------------------------------------------------------------------
_gsrc = open(MASTER_GEN).read()
assert "regdet_v11_master.ipynb'" in _gsrc
_gsrc = _gsrc.replace("regdet_v11_master.ipynb'", "_stability_lag_engine_harvest.ipynb'", 1)
_ns = {'__name__': '_master_gen_harvest'}
exec(compile(_gsrc, MASTER_GEN, 'exec'), _ns)
_mcells = _ns['cells']


def _src(i):
    return ''.join(_mcells[i]['source'])


IDX_CONST, IDX_LOAD, IDX_ENGINE, IDX_PLOT = 3, 5, 7, 9
SRC_CONST, SRC_LOAD = _src(IDX_CONST), _src(IDX_LOAD)
SRC_ENGINE, SRC_PLOT = _src(IDX_ENGINE), _src(IDX_PLOT)

# Cell identity is asserted by CONTENT, never trusted by number.
assert 'REGIME_COLORS' in SRC_CONST and 'BAR_DIR_WEIGHT   = 0.0' in SRC_CONST
assert 'import numpy as np' in SRC_CONST and 'CONFIGS = [' in SRC_CONST
assert 'def load_2h' in SRC_LOAD and 'def _synth' in SRC_LOAD
assert 'def build_features' in SRC_ENGINE and 'def label_bars' in SRC_ENGINE \
    and 'def fit_hmm_ensemble' in SRC_ENGINE and 'def confirm_delay' in SRC_ENGINE \
    and 'def n_params' in SRC_ENGINE \
    and 'def ensemble_direction_masses_by_mode' in SRC_ENGINE
assert 'def shade_bands' in SRC_PLOT and 'def regime_legend' in SRC_PLOT \
    and 'def set_price_ylim' in SRC_PLOT and 'def shade_regimes' in SRC_PLOT

PROVENANCE = (
    f'build_master_notebook_v2.py cell {IDX_CONST} (constants/imports/CONFIGS), '
    f'cell {IDX_LOAD} (load_2h/_synth), cell {IDX_ENGINE} (feature + labeling engine), '
    f'cell {IDX_PLOT} (regime-background plot helpers)'
)
try:
    os.remove(f'{HERE}/_stability_lag_engine_harvest.ipynb')
except OSError:
    pass

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip('\n').splitlines(keepends=True)})


def co(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip('\n').splitlines(keepends=True)})


BANNER = '# ' + '=' * 74 + '\n'

# ===========================================================================
md(r"""
# RegDet V1.1 — FIT STABILITY (S) vs TRANSITION LAG (L)

Implements `STABILITY_LAG_PROTOCOL.md` **exactly**. The protocol was frozen before
any number in this notebook existed.

## The two defects

**D1 — fit instability.** Refitting the same arm with a *disjoint seed set* flips
~6% of all bars. That is larger than the entire 25%↔100% HMM-share difference, so
it contaminates every comparison in the project.

**D2 — transition lag.** Labels describe the regime that *just ended*. Reference
case: the May-2025 V-bottom, labelled `L_BEAR` for the whole +2.7% advance and
turning `BULL` only near the top.

## The central tension — stated up front, not hidden

> Lower lag = react sooner = react to smaller moves = **more** switches and **more**
> sensitivity to the underlying fit. **These objectives oppose each other.** The
> honest deliverable may be a FRONTIER, not a fix. If nothing improves one without
> degrading the other, this notebook prints `NO DOMINATING CONFIGURATION` and shows
> the frontier. It does not manufacture a winner.

## The three metrics (frozen)

| | definition | direction |
|---|---|---|
| **S** | mean pairwise 5-label agreement across **R = 4 disjoint seed sets** | higher better |
| **L** | bars from a label-blind 2.0% ZigZag swing start to the first correctly-directed emitted label; unmatched swings score the **full swing length** | lower better |
| **W** | label switches per 100 bars | guard — must not get worse |

## Guards — a candidate is VOID if any fails

`G1` every label occupancy in [3%, 50%] · `G2` no label collapses below 3% ·
`G3` W ≤ 25 / 100 bars · `G4` SIDEWAYS ≤ 50%.

**G4 exists because S is trivially maximised by labelling everything SIDEWAYS.**
Section 3 drives that exact degenerate case through the real guard code and asserts
it is VOIDED — the guard is tested, not trusted.

## Fixed by decree

`BAR_DIR_WEIGHT = 0.0` throughout. It is settled and is **not** swept here; the
notebook asserts it at every labelling call.
""")

co(r"""
%pip install -q hmmlearn yfinance
""")

# --- engine ----------------------------------------------------------------
md(f"""
## 1. Engine, inlined verbatim

Inlined out of `build_master_notebook_v2.py` rather than imported, because a
standalone `.ipynb` on Kaggle cannot import a local `.py` sitting next to it.
Source: `{PROVENANCE}`.

Not one line is modified — same constants, same features, same gates, same
hysteresis, same causality proofs. The master notebook, its generator,
`regime_scorecard.py` and `build_hmm_share_charts.py` are untouched by this
notebook.
""")

co(BANNER
   + f'# CONSTANTS + IMPORTS -- INLINED VERBATIM from {MASTER_GEN.split("/")[-1]}\n'
   + f'# (master notebook code cell {IDX_CONST}). Unmodified.\n'
   + BANNER + SRC_CONST)

co(BANNER
   + f'# DATA LOADER -- INLINED VERBATIM (master code cell {IDX_LOAD}).\n'
   + '# yfinance 60m -> 2h resample, with a synthetic fallback. Unmodified.\n'
   + BANNER + SRC_LOAD)

co(BANNER
   + f'# FEATURE + LABELING ENGINE -- INLINED VERBATIM (master code cell {IDX_ENGINE}).\n'
   + '# build_features / fit_hmm_ensemble / confirm_delay / label_bars / n_params.\n'
   + BANNER + SRC_ENGINE)

co(BANNER
   + f'# PLOT HELPERS -- INLINED VERBATIM (master code cell {IDX_PLOT}).\n'
   + '# regime_blocks / shade_bands / shade_regimes / set_price_ylim / regime_legend.\n'
   + '# These are the MASTER NOTEBOOK regime-background style used by figure 4.\n'
   + BANNER + SRC_PLOT)

co(r"""
# ---------------------------------------------------------------------------
# PROVENANCE BANNER + the two standing decrees, asserted rather than promised.
# ---------------------------------------------------------------------------
import itertools, time, contextlib
import matplotlib.dates as mdates
from scipy.stats import spearmanr

T_START = time.time()

if TAC_SYNTH:
    print('#' * 100)
    for _ in range(3):
        print('#   SYNTHETIC - ILLUSTRATIVE ONLY   ' * 2)
    print('#' * 100)
    print('#  The yfinance fetch FAILED (it is firewalled in some sandboxes).')
    print('#  Every chart and number below is generated from a synthetic GBM series.')
    print('#  It is a PLUMBING TEST ONLY. It says NOTHING about real Nifty regimes,')
    print('#  about real fit stability, and NOTHING about real transition lag.')
    print('#  Re-run on Kaggle, where yfinance works, before believing any verdict.')
    print('#' * 100)
else:
    print('=' * 100)
    print('REAL yfinance 2h data. Verdicts below are decision-grade.')
    print('=' * 100)

# DECREE 1 -- BAR_DIR_WEIGHT is settled at 0.0 and is not a free variable here.
assert BAR_DIR_WEIGHT == 0.0, 'BAR_DIR_WEIGHT must be 0.0 for this notebook'
W_FIXED = 0.0
print(f'\nDECREE: BAR_DIR_WEIGHT is FIXED at {W_FIXED} (direction is 100% HMM). '
      f'Not swept. Asserted at every labelling call.')

print(f'bars={len(nifty)}   span {nifty.index[0]:%Y-%m-%d} -> {nifty.index[-1]:%Y-%m-%d}')
print(f'shipped labeling knobs: CONFIRM_BARS={CONFIRM_BARS}  ENSEMBLE_K={ENSEMBLE_K}  '
      f'ESCALATION_DURING_HOLD={ESCALATION_DURING_HOLD!r}  '
      f'INTENSITY_MODE={INTENSITY_MODE!r}  DIRECTION_MODE={DIRECTION_MODE!r}')
print(f'shipped config       : {ADOPTED_CONFIG_NAME}')
""")

# ===========================================================================
md(r"""
## 2. The metrics — S, L, W — and the ZigZag leakage tripwire

### The look-ahead question, answered before the code

The ZigZag **uses future prices on purpose**. It is a *retrospective evaluation*
device, exactly like a forward return: it grades a decision that was already made.
The protocol permits that and requires one thing in exchange — **it must never
touch a label**.

That is enforced three ways, all of them runtime asserts, not comments:

1. **Phase ordering.** All labelling happens in the LABEL PHASE; the ZigZag is not
   computed until the EVAL PHASE. `label_bars` is wrapped by a guard that asserts
   `ZZ_CALLS == 0` — i.e. *no label was ever produced after a ZigZag existed*.
2. **Object identity.** The guard scans every positional and keyword argument of
   every `label_bars` call and refuses any object that is a ZigZag output.
3. **Memory aliasing.** For array arguments it additionally asserts
   `np.shares_memory(arg, zz) is False` against every ZigZag array, catching a view
   or a slice that identity alone would miss.

The same wrapper asserts `bar_dir_weight == 0.0` on every single call, so
`BAR_DIR_WEIGHT` cannot drift inside a sweep.
""")

co(r"""
# ===========================================================================
# METRIC PARAMETERS -- ALL DECLARED HERE, BEFORE ANY NUMBER EXISTS.
# ===========================================================================
R_SEED_SETS   = 4       # protocol: R = 4 DISJOINT seed sets. NOT reducible.
ZZ_PCT        = 2.0     # protocol: ZigZag reversal threshold, 2.0%
W_GUARD_MAX   = 25.0    # G3
OCC_MIN_PCT   = 3.0     # G1 / G2
OCC_MAX_PCT   = 50.0    # G1
SIDEWAYS_MAX  = 50.0    # G4

# S2 -- log-likelihood pruning threshold, expressed as a FRACTION OF THE LL SPREAD
# of the ensemble. Declared here, not tuned to a result. A member is kept when
#     (best_ll - ll) <= S2_LL_FRAC * (best_ll - worst_ll)
S2_LL_FRAC    = 0.25

# L2 -- adaptive confirmation. The "strong signal" threshold is declared as a RATE
# over the FIT WINDOW (the same device H_TARGET_RATE already uses), not as a raw
# number tuned after seeing lag. Strictly causal: it is a fit-window constant.
L2_FAST_RATE  = 0.25    # the top 25% of fit-window bars by |bull_mass - bear_mass|
                        # get 1-bar confirmation; everything else keeps 2.

print(f'R = {R_SEED_SETS} disjoint seed sets   ZigZag = {ZZ_PCT}%   '
      f'G3 W<={W_GUARD_MAX}   G1 occ in [{OCC_MIN_PCT},{OCC_MAX_PCT}]%   '
      f'G4 SIDEWAYS<={SIDEWAYS_MAX}%')
print(f'S2_LL_FRAC={S2_LL_FRAC}   L2_FAST_RATE={L2_FAST_RATE}   '
      '(both declared before any result)')
""")

co(r"""
# ===========================================================================
# THE ZIGZAG -- LABEL-BLIND, RETROSPECTIVE. Sees prices only; never a label.
# ===========================================================================
ZZ_CALLS = 0            # how many times a ZigZag has been computed
ZZ_IDS = set()          # id() of every object a ZigZag has ever returned
ZZ_ARRAYS = []          # the arrays themselves, for np.shares_memory checks


def zigzag_pivots(px, pct=ZZ_PCT):
    '''Indices of ZigZag pivots on a close series, `pct`% reversal threshold.

    RETROSPECTIVE BY CONSTRUCTION: a pivot at bar i is only confirmed once price
    has moved pct% away from it, which happens at some bar > i. That is exactly
    the look-ahead this function is allowed to have and `label_bars` is not.

    Takes a bare float array in and returns a bare int array out -- it has no
    access to a label, a mass, a model or a feature frame.
    '''
    global ZZ_CALLS
    px = np.asarray(px, dtype=float)
    n = len(px)
    if n < 3:
        out = np.array([], dtype=int)
    else:
        hi_i = lo_i = 0
        d, start, piv, ext_i = 0, None, [], 0
        for i in range(1, n):
            if px[i] > px[hi_i]:
                hi_i = i
            if px[i] < px[lo_i]:
                lo_i = i
            if (px[hi_i] - px[i]) / px[hi_i] * 100.0 >= pct:
                d, piv, ext_i, start = -1, [hi_i], i, i
                break
            if (px[i] - px[lo_i]) / px[lo_i] * 100.0 >= pct:
                d, piv, ext_i, start = +1, [lo_i], i, i
                break
        if d == 0:
            out = np.array([], dtype=int)
        else:
            for i in range(start + 1, n):
                if d > 0:
                    if px[i] >= px[ext_i]:
                        ext_i = i
                    elif (px[ext_i] - px[i]) / px[ext_i] * 100.0 >= pct:
                        piv.append(ext_i); d, ext_i = -1, i
                else:
                    if px[i] <= px[ext_i]:
                        ext_i = i
                    elif (px[i] - px[ext_i]) / px[ext_i] * 100.0 >= pct:
                        piv.append(ext_i); d, ext_i = +1, i
            if ext_i != piv[-1]:
                piv.append(ext_i)          # final, PROVISIONAL pivot
            out = np.asarray(piv, dtype=int)

    ZZ_CALLS += 1
    ZZ_IDS.add(id(out))
    ZZ_ARRAYS.append(out)
    return out


def zigzag_swings(px, pct=ZZ_PCT):
    '''[(i0, i1, +1/-1), ...] -- consecutive pivot pairs that clear `pct`.

    The last (provisional) leg is kept only if it actually cleared the threshold,
    so a half-formed leg at the right edge cannot inflate or deflate lag.
    '''
    px = np.asarray(px, dtype=float)
    piv = zigzag_pivots(px, pct)
    sw = []
    for a, b in zip(piv[:-1], piv[1:]):
        move = (px[b] - px[a]) / px[a] * 100.0
        if abs(move) >= pct:
            sw.append((int(a), int(b), 1 if move > 0 else -1))
    out = sw
    ZZ_IDS.add(id(out))
    return out


# ---------------------------------------------------------------------------
# THE LEAKAGE TRIPWIRE. `label_bars` is SHADOWED by a guard; every later call in
# this notebook goes through it. Three independent checks, plus the w=0 decree.
# ---------------------------------------------------------------------------
_label_bars_raw = label_bars
LABEL_CALLS = 0
LABEL_PHASE_OPEN = True     # flipped shut once the EVAL phase starts


@contextlib.contextmanager
def reopen_label_phase(why):
    '''EXPLICITLY and LOUDLY reopen the label phase after the ZigZag exists.

    Used once, for the combination arms, whose identity cannot be known until the
    single interventions have been measured. Only the PHASE-ORDERING check is
    relaxed; checks 2, 3 and 4 (object identity, memory aliasing, w == 0.0) stay
    armed throughout, and the reopening prints itself so it can never be a silent
    hole in the proof.
    '''
    global LABEL_PHASE_OPEN
    print('!' * 78)
    print(f'!! LABEL PHASE EXPLICITLY REOPENED: {why}')
    print('!! ordering check relaxed; identity / aliasing / w==0 checks STAY ARMED')
    print('!' * 78)
    LABEL_PHASE_OPEN = True
    try:
        yield
    finally:
        LABEL_PHASE_OPEN = False
        print('!! LABEL PHASE CLOSED AGAIN -- ordering check re-armed.')


def label_bars(*args, **kwargs):
    '''Guarded shadow of the engine's label_bars. Adds asserts, changes nothing.'''
    global LABEL_CALLS
    assert ZZ_CALLS == 0 or LABEL_PHASE_OPEN, (
        'LEAKAGE: a label was produced AFTER a ZigZag had been computed. All '
        'labelling must complete in the LABEL PHASE, before any ZigZag exists.')
    _bw = kwargs.get('bar_dir_weight', BAR_DIR_WEIGHT)
    assert _bw == 0.0, f'BAR_DIR_WEIGHT drifted to {_bw} inside a labelling call'
    for v in list(args) + list(kwargs.values()):
        assert id(v) not in ZZ_IDS, 'LEAKAGE: a ZigZag output was passed to label_bars'
        if isinstance(v, np.ndarray):
            for z in ZZ_ARRAYS:
                assert not np.shares_memory(v, z), \
                    'LEAKAGE: a label_bars argument aliases ZigZag memory'
    LABEL_CALLS += 1
    return _label_bars_raw(*args, **kwargs)


print('ZigZag + leakage tripwire ready.')
print('  check 1  phase ordering  : label_bars asserts ZZ_CALLS == 0')
print('  check 2  object identity : every argument checked against ZZ_IDS')
print('  check 3  memory aliasing : np.shares_memory vs every ZigZag array')
print('  check 4  decree          : bar_dir_weight == 0.0 on every call')
""")

co(r"""
# ===========================================================================
# S / L / W  -- the three metrics, one function each. No composite anywhere.
# ===========================================================================
DIR_OF_LABEL = {'H_BULL': 1, 'L_BULL': 1, 'SIDEWAYS': 0, 'L_BEAR': -1, 'H_BEAR': -1}


def emitted_direction(labels):
    '''5-label string array -> +1 bull / 0 sideways / -1 bear.'''
    return np.array([DIR_OF_LABEL[x] for x in np.asarray(labels)], dtype=int)


def metric_S(label_sets):
    '''S = mean pairwise 5-LABEL agreement (%) across R seed sets.

    Returns (S_mean, pairwise_matrix RxR, list_of_pairwise_values).
    The FULL matrix is returned because the protocol requires it reported, not
    just the mean.
    '''
    R = len(label_sets)
    M = np.full((R, R), np.nan)
    vals = []
    for i in range(R):
        M[i, i] = 100.0
        for j in range(i + 1, R):
            a = np.asarray(label_sets[i]); b = np.asarray(label_sets[j])
            assert len(a) == len(b), 'label sets must be the same length'
            v = 100.0 * float(np.mean(a == b))
            M[i, j] = M[j, i] = v
            vals.append(v)
    return float(np.mean(vals)), M, vals


def metric_L(labels, swings):
    '''L per ZigZag swing = bars from swing START to the first correctly
    directed EMITTED label inside the swing.

    An UNMATCHED swing (never labelled correctly anywhere inside it) scores the
    FULL swing length -- the worst case, per protocol. It is not dropped.

    Returns (median, p75, per_swing_array, n_unmatched).
    '''
    d = emitted_direction(labels)
    lags, unmatched = [], 0
    for i0, i1, sgn in swings:
        seg = d[i0:i1 + 1]
        hit = np.flatnonzero(seg == sgn)
        if hit.size:
            lags.append(int(hit[0]))
        else:
            lags.append(int(i1 - i0))
            unmatched += 1
    if not lags:
        return np.nan, np.nan, np.array([]), 0
    a = np.asarray(lags, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 75)), a, unmatched


def metric_W(labels):
    '''W = 5-label switches per 100 bars.'''
    a = np.asarray(labels)
    if len(a) < 2:
        return 0.0
    return 100.0 * float(np.sum(a[1:] != a[:-1])) / (len(a) - 1)


def occupancy_pct(labels):
    a = np.asarray(labels)
    return {lb: 100.0 * float(np.mean(a == lb)) for lb in REGIME_LABELS}


# ---------------------------------------------------------------------------
# GUARDS G1..G4. One implementation, used by real arms AND by the degeneracy
# stub in section 3, so the test exercises the shipping code path.
# ---------------------------------------------------------------------------
def evaluate_guards(occ, W):
    '''{name: (bool_pass, reason)} for the four protocol guards.'''
    g = {}
    bad_hi = [l for l in REGIME_LABELS if occ.get(l, 0.0) > OCC_MAX_PCT]
    bad_lo = [l for l in REGIME_LABELS if occ.get(l, 0.0) < OCC_MIN_PCT]
    g['G1'] = (not bad_hi and not bad_lo,
               'ok' if not (bad_hi or bad_lo)
               else f'outside [{OCC_MIN_PCT},{OCC_MAX_PCT}]%: '
                    + ','.join(sorted(set(bad_hi + bad_lo))))
    g['G2'] = (not bad_lo, 'ok' if not bad_lo else 'collapsed: ' + ','.join(bad_lo))
    g['G3'] = (W <= W_GUARD_MAX, f'W={W:.2f} vs max {W_GUARD_MAX}')
    g['G4'] = (occ.get('SIDEWAYS', 0.0) <= SIDEWAYS_MAX,
               f"SIDEWAYS={occ.get('SIDEWAYS', 0.0):.1f}% vs max {SIDEWAYS_MAX}%")
    return g


def guards_pass(g):
    return all(v[0] for v in g.values())


print('metrics ready: metric_S (pairwise matrix), metric_L (median/p75, unmatched '
      '= full swing length), metric_W, evaluate_guards (G1-G4).')
print('NO composite score is defined anywhere in this notebook -- by design.')
""")

# ===========================================================================
md(r"""
## 3. Machinery self-tests — including **driving G4 with a degenerate stub**

Four tests, all of which must pass before a single arm is fitted.

1. **ZigZag correctness** on a hand-built sawtooth with known pivots.
2. **`confirm_delay_dyn` equivalence** — the generalised (per-bar) confirmation
   scanner must reproduce the engine's frozen `confirm_delay` bit-for-bit when its
   requirement is held constant. This is the "every new switch has an OFF value
   asserted against the frozen original" rule.
3. **G4 bites.** A stub that forces an all-`SIDEWAYS` labelling is pushed through
   the *same* summarising and guard code the real arms use. It scores a perfect
   `S = 100.00%` — which is precisely how S is gamed — and must be **VOIDED**.
4. **G3 bites** on a synthetic high-whipsaw labelling.
""")

co(r"""
# ---------------------------------------------------------------------------
# The generalised confirmation scanner. Structurally identical to the engine's
# confirm_delay; the ONLY change is that the required run length is asked for
# per bar via need_fn(t, prev_emitted, candidate) instead of being a constant.
# STILL STRICTLY CAUSAL: out[t] depends on raw[0..t] and out[t-1] only.
# ---------------------------------------------------------------------------
def confirm_delay_dyn(raw, need_fn):
    raw = np.asarray(raw)
    n = len(raw)
    out = np.empty(n, dtype=raw.dtype)
    if n == 0:
        return out
    out[0] = raw[0]
    cand, run = raw[0], 0
    for t in range(1, n):
        if raw[t] == out[t - 1]:
            out[t] = raw[t]
            cand, run = raw[t], 0
        else:
            if raw[t] == cand:
                run += 1
            else:
                cand, run = raw[t], 1
            if run >= int(need_fn(t, out[t - 1], cand)):
                out[t] = cand
                run = 0
            else:
                out[t] = out[t - 1]
    return out


@contextlib.contextmanager
def confirm_policy(need_fn):
    '''Temporarily route label_bars' confirmation through `need_fn`.

    label_bars resolves `confirm_delay` from the notebook globals at call time,
    so rebinding that name here changes the confirmation rule WITHOUT editing,
    copying or forking a single line of the inlined engine. Restored on exit.
    '''
    g = globals()
    old = g['confirm_delay']

    def _patched(raw, confirm_bars=None):
        return confirm_delay_dyn(raw, need_fn)

    g['confirm_delay'] = _patched
    try:
        yield
    finally:
        g['confirm_delay'] = old


# ---- TEST 1: ZigZag on a known sawtooth -----------------------------------
_saw = np.array([100, 101, 100.5, 105, 104, 101, 100, 103, 108, 107, 102, 106.])
_sw = zigzag_swings(_saw, pct=2.0)
assert len(_sw) >= 3, f'zigzag found too few swings on the sawtooth: {_sw}'
for _a, _b, _s in _sw:
    _mv = (_saw[_b] - _saw[_a]) / _saw[_a] * 100
    assert _a < _b and np.sign(_mv) == _s and abs(_mv) >= 2.0, f'bad swing {(_a,_b,_s)}'
assert [s for _, _, s in _sw] == [(-1) ** k * _sw[0][2] for k in range(len(_sw))], \
    'zigzag swing directions must alternate'
print(f'TEST 1 PASS  zigzag: {len(_sw)} alternating swings, each >= 2.0%: {_sw}')

# The unit test above ran the ZigZag on a 12-point TOY array that is not the
# price series and never enters the pipeline. Its bookkeeping is cleared here,
# explicitly and in the open, so the leakage tripwire starts the label phase from
# a true zero rather than from a self-test artefact.
ZZ_CALLS = 0
ZZ_IDS.clear()
ZZ_ARRAYS.clear()
print('             (ZigZag bookkeeping reset after the toy-array unit test; the '
      'tripwire now starts from ZZ_CALLS = 0)')

# ---- TEST 2: confirm_delay_dyn == confirm_delay when the need is constant ---
_rng = np.random.default_rng(7)
_raw = _rng.choice(np.array(['BULL', 'BEAR', 'SIDE']), size=4000)
for _cb in (1, 2, 3, 5):
    _ref = confirm_delay(_raw, _cb)
    _got = confirm_delay_dyn(_raw, lambda t, p, c, _n=_cb: _n)
    assert np.array_equal(_ref, _got), f'confirm_delay_dyn diverges at confirm_bars={_cb}'
print('TEST 2 PASS  confirm_delay_dyn reproduces the frozen confirm_delay '
      'bit-for-bit at confirm_bars = 1, 2, 3, 5 (its OFF value)')

# ---- TESTS 3 & 4 are deferred to section 6, where summarise_arm exists ------
print('TESTS 3 & 4 (G4 / G3 bite) run in section 6, against the REAL guard path.')
""")

# ===========================================================================
md(r"""
## 4. Fit harness — R = 4 disjoint seed sets

`R = 4` is the whole reason S means anything, so it is **never** the thing that
gets cut for runtime. Seed set *r* uses seeds `BASE_SEED + r*K + 0..K-1`, which are
disjoint by construction and asserted to be.

Every arm is (config × ensemble policy × labelling policy). Fits are **cached** by
`(features, N, cov, K, R, init)` so that the eight labelling-side interventions and
every combination reuse the baseline's fits instead of refitting — the refits that
remain are exactly the ones that *change the fit*.
""")

co(r"""
# ===========================================================================
# FEATURES + SCALING (per feature subset), and the seed-set fitter.
# ===========================================================================
FEAT_DF = build_features(nifty, vix, LOOKBACK_SCALE)
DATES = FEAT_DF.index
CLOSE = nifty.reindex(DATES).values
TREND_RAW = FEAT_DF[TREND_FEATURE].values
N_BARS = len(DATES)
N_FIT = max(int(N_BARS * TRAIN_FRACTION), 50)
print(f'{N_BARS} feature bars; anchored fit window = {N_FIT} bars '
      f'({TRAIN_FRACTION:.0%})  {DATES[0]:%Y-%m-%d} -> {DATES[-1]:%Y-%m-%d}')

_XS_CACHE = {}


def scaled_X(features):
    key = tuple(features)
    if key not in _XS_CACHE:
        raw = FEAT_DF[list(features)].values
        sc = StandardScaler().fit(raw[:N_FIT])
        _XS_CACHE[key] = sc.transform(raw)
    return _XS_CACHE[key]


def seed_sets(K, R=R_SEED_SETS, base=BASE_SEED):
    ss = [list(range(base + r * K, base + r * K + K)) for r in range(R)]
    flat = [s for st in ss for s in st]
    assert len(set(flat)) == len(flat), 'seed sets must be DISJOINT'
    return ss


def fit_deterministic(X, N, cov, n_iter=None):
    '''S4 -- SEED-FREE initialisation.

    means from a fixed-seed k-means, sorted deterministically; startprob, transmat
    and covars set to fixed values. `init_params=''` means hmmlearn initialises
    NOTHING, so `random_state` cannot enter the fit at all. Consequence, stated
    plainly: every seed produces the IDENTICAL model, so S is 100% BY
    CONSTRUCTION. That is a real property, not an achievement -- see section 7.
    '''
    from sklearn.cluster import KMeans
    n_iter = HMM_ITER if n_iter is None else n_iter
    km = KMeans(n_clusters=N, n_init=10, random_state=0).fit(X)
    means = km.cluster_centers_[np.argsort(km.cluster_centers_[:, 0])]
    m = hmm.GaussianHMM(n_components=N, covariance_type=cov, n_iter=n_iter,
                        init_params='', params='stmc')
    m.startprob_ = np.full(N, 1.0 / N)
    tm = np.full((N, N), 0.1 / max(N - 1, 1))
    np.fill_diagonal(tm, 0.9)
    m.transmat_ = tm
    m.means_ = means
    d = X.shape[1]
    if cov == 'full':
        m.covars_ = np.tile(np.cov(X.T) + 1e-3 * np.eye(d), (N, 1, 1))
    else:
        m.covars_ = np.tile(X.var(axis=0) + 1e-6, (N, 1))
    m.fit(X)
    return m


FIT_CACHE = {}
FIT_COUNT = 0


def get_fits(features, N, cov, K, init='seeded'):
    '''[models_for_seed_set_0, ..., models_for_seed_set_{R-1}]. Cached.'''
    global FIT_COUNT
    key = (tuple(features), N, cov, K, R_SEED_SETS, init)
    if key in FIT_CACHE:
        return FIT_CACHE[key]
    X = scaled_X(features)[:N_FIT]
    sets = seed_sets(K)
    out = []
    for sd in sets:
        if init == 'det':
            m = fit_deterministic(X, N, cov)
            out.append([m] * K)          # seed-free: K identical members
            FIT_COUNT += 1
        else:
            ms, _ = fit_hmm_ensemble(X, N, cov, K=K, base_seed=sd[0])
            out.append(ms)
            FIT_COUNT += K
    FIT_CACHE[key] = out
    return out


print(f'fit harness ready. R={R_SEED_SETS} DISJOINT seed sets, e.g. K=6 -> '
      f'{seed_sets(6)}')
""")

# ===========================================================================
md(r"""
## 5. The arms

### Interventions (protocol section "Interventions to test")

| id | axis | what changes |
|---|---|---|
| `BASELINE` | — | shipped config, `K=6`, `CONFIRM_BARS=2`, `hold='block'` |
| `S1a/S1b` | S | `ENSEMBLE_K` 6 → 12 → 24 |
| `S2` | S | drop ensemble members whose converged LL is > 25% of the LL spread below the best |
| `S3` | S | **majority-vote the per-seed emitted direction** instead of averaging direction mass |
| `S4` | S | deterministic (seed-free) initialisation |
| `L1` | L | `CONFIRM_BARS` 2 → 1 |
| `L2` | L | adaptive confirmation: 1 bar when `\|bull−bear\|` clears the fit-window 75th percentile, else 2 |
| `L3` | L | asymmetric confirmation: 1 bar to LEAVE a directional state, 2 to ENTER one |
| `L4` | L | `ESCALATION_DURING_HOLD` `'block'` → `'allow'` |

> **`L4` is reported but is expected to be a null on L, and this was written down
> before running it.** `ESCALATION_DURING_HOLD` acts only on the H↔L *intensity*
> axis; the L metric grades the *direction* axis, which that switch cannot reach.
> `L4` therefore cannot move L by construction. It is still run — because "cannot"
> deserves evidence, and because it genuinely can move W.

### Config / capacity arms (coordinator scope addition)

The hypothesis under test: **model capacity, not the S1–S4 machinery, is the
dominant driver of fit instability.** More free parameters ⇒ rougher likelihood
surface ⇒ more local optima ⇒ seeds diverge more. `rows/param` is the theoretical
predictor and is reported next to measured S so the hypothesis is *tested* rather
than assumed. The three named `CONFIGS` are taken verbatim from the master
generator; the capacity grid varies `N_STATES ∈ {3,4,5,6}` × `cov ∈ {diag, full}`,
and `B: lean-feat` supplies the feature-count axis.

Guards apply to config arms **exactly** as to intervention arms: a low-capacity
config that wins S by collapsing toward one label is VOID, not a winner.
""")

co(r"""
# ===========================================================================
# ARM DEFINITIONS. Every arm is a dict; nothing here runs a fit yet.
#   fit  : features / N / cov / K / init
#   lab  : kwargs handed to label_bars   (bar_dir_weight is NEVER among them)
#   post : optional ensemble post-processor (S2 pruning)
#   pol  : optional confirmation-policy factory (L2 / L3)
#   mass : optional direction-mass override  (S3)
# ===========================================================================
CFG_BY_NAME = {c['name']: c for c in CONFIGS}
BASE_CFG = CFG_BY_NAME[ADOPTED_CONFIG_NAME]
BASE_FEATURES = tuple(BASE_CFG['features'])
BASE_N, BASE_COV = BASE_CFG['N'], BASE_CFG['cov']


def arm(name, group, note, features=None, N=None, cov=None, K=ENSEMBLE_K,
        init='seeded', lab=None, post=None, pol=None, mass=False):
    return dict(name=name, group=group, note=note,
                features=tuple(features or BASE_FEATURES),
                N=N or BASE_N, cov=cov or BASE_COV, K=K, init=init,
                lab=dict(lab or {}), post=post, pol=pol, mass=mass)


# ---- S2: log-likelihood pruning -------------------------------------------
def prune_by_ll(models, X):
    lls = np.array([float(m.score(X)) for m in models])
    best, worst = lls.max(), lls.min()
    spread = best - worst
    if spread <= 0:
        return list(models), lls, len(models)
    keep = lls >= best - S2_LL_FRAC * spread
    kept = [m for m, k in zip(models, keep) if k]
    assert kept, 'LL pruning must always keep at least the best member'
    return kept, lls, len(kept)


# ---- L2 / L3: confirmation policies ---------------------------------------
def make_pol_L2(strength, thr):
    '''1-bar confirmation on strong bars, 2 otherwise. Uses ONLY bar t.'''
    def need(t, prev, cand):
        return 1 if strength[t] >= thr else CONFIRM_BARS
    return need


def make_pol_L3(_strength=None, _thr=None):
    '''Asymmetric: 1 bar to LEAVE a directional state, CONFIRM_BARS to ENTER one.

    'Extreme regime' is read at the DIRECTION level (BULL/BEAR vs SIDE), because
    that is the axis confirm_delay acts on. Leaving BULL or BEAR is cheap; taking
    a directional position from SIDE still costs the full confirmation.
    '''
    def need(t, prev, cand):
        return 1 if prev in ('BULL', 'BEAR') else CONFIRM_BARS
    return need


ARMS = [
    arm('BASELINE', 'baseline',
        f'shipped: {ADOPTED_CONFIG_NAME}, K={ENSEMBLE_K}, CONFIRM_BARS={CONFIRM_BARS}, '
        f'hold={ESCALATION_DURING_HOLD!r}'),

    # ---- S interventions --------------------------------------------------
    arm('S1a K=12', 'S', 'ENSEMBLE_K 6 -> 12', K=12),
    arm('S1b K=24', 'S', 'ENSEMBLE_K 6 -> 24', K=24),
    arm('S2 LL-prune', 'S', f'drop members > {S2_LL_FRAC:.0%} of the LL spread '
        f'below the best', post='ll_prune'),
    arm('S3 vote', 'S', 'majority-vote the per-seed EMITTED direction', mass=True),
    arm('S4 det-init', 'S', 'deterministic seed-free initialisation', init='det'),

    # ---- L interventions --------------------------------------------------
    arm('L1 confirm=1', 'L', 'CONFIRM_BARS 2 -> 1', lab=dict(confirm_bars=1)),
    arm('L2 adaptive', 'L', f'1 bar when |bull-bear| >= fit-window '
        f'p{100*(1-L2_FAST_RATE):.0f}, else 2', pol='L2'),
    arm('L3 asymmetric', 'L', 'fast to LEAVE a directional state, slow to enter',
        pol='L3'),
    arm('L4 hold=allow', 'L', "ESCALATION_DURING_HOLD 'block' -> 'allow' "
        '(intensity axis only -- expected null on L)',
        lab=dict(escalation_during_hold='allow')),
]

# ---- CONFIG / CAPACITY arms (scope addition) -------------------------------
for _c in CONFIGS:
    if _c['name'] == ADOPTED_CONFIG_NAME:
        continue                       # that IS the baseline; reused, not refitted
    ARMS.append(arm(f"CFG {_c['name']}", 'config',
                    f"named config: N={_c['N']} cov={_c['cov']} "
                    f"{len(_c['features'])} features",
                    features=_c['features'], N=_c['N'], cov=_c['cov']))

_named = {(tuple(c['features']), c['N'], c['cov']) for c in CONFIGS}
for _N in (3, 4, 5, 6):
    for _cov in ('diag', 'full'):
        if (tuple(FEATURE_COLS), _N, _cov) in _named:
            continue                   # already covered by a named config
        ARMS.append(arm(f'CAP N={_N} {_cov}', 'capacity',
                        f'capacity grid: N={_N} cov={_cov} 9 features',
                        features=FEATURE_COLS, N=_N, cov=_cov))

print(f'{len(ARMS)} arms declared before any fit:')
for _a in ARMS:
    print(f"  {_a['name']:<16} [{_a['group']:<8}] {_a['note']}")

_keys, _budget = set(), 0
for _a in ARMS:
    _k = (tuple(_a['features']), _a['N'], _a['cov'], _a['K'], R_SEED_SETS, _a['init'])
    if _k not in _keys:
        _keys.add(_k)
        _budget += (R_SEED_SETS if _a['init'] == 'det' else _a['K'] * R_SEED_SETS)
print(f'\nfit budget: {_budget} HMM fits across {len(_keys)} distinct fit sets '
      f'({len(ARMS)} arms). Labelling-side arms reuse the baseline fits; only arms '
      f'that change the FIT ITSELF refit.')
""")

# ===========================================================================
md(r"""
## 6. LABEL PHASE — every arm labelled, R times, **before any ZigZag exists**

This ordering is the leakage proof. When this section finishes, `ZZ_CALLS` is
still 0 and every label in the notebook has already been written.
""")

co(r"""
# ===========================================================================
# THE LABEL PHASE.
# ===========================================================================
assert ZZ_CALLS == 0, 'a ZigZag was computed before the label phase -- ordering broken'


def _masses_for(models, features):
    '''Direction masses as the engine computes them (used only to build the L2
    strength series -- a fit-window constant, strictly causal).'''
    b, s, r, _ = ensemble_direction_masses_by_mode(
        models, scaled_X(features), list(features), DIRECTION_EXCLUDE,
        DIRECTION_MODE, None, None, None)
    return b, s, r


def _vote_masses_factory():
    '''S3 -- replace mass-averaging with a per-seed majority VOTE.

    For every ensemble member: form that member's own direction call (argmax of
    its bull/bear/side masses, with the CONF_L low-confidence override applied to
    that member). Then the returned 'masses' are the VOTE SHARES across members.

    They are a valid probability simplex, they are permutation-invariant, their
    argmax IS the majority vote, and regime_confidence becomes the winning vote
    share -- so CONF_L keeps its meaning (a bar with no majority is SIDEWAYS).
    Everything downstream -- gates, hysteresis, prob_* columns -- is untouched.
    '''
    def voted(models, Xs, feature_subset, exclude, mode, tau, c, scale):
        models = list(models)
        votes = np.zeros((len(Xs), 3))          # bull, side, bear
        for m in models:
            b, s, r, _ = _emdbm_raw([m], Xs, feature_subset, exclude, mode, tau, c, scale)
            st = np.stack([b, r, s], axis=1)    # bull, bear, side
            conf = st.max(axis=1)
            w = st.argmax(axis=1)
            w = np.where(conf < CONF_L, 2, w)   # 2 == side
            idx = np.where(w == 0, 0, np.where(w == 1, 2, 1))   # -> bull,side,bear
            votes[np.arange(len(w)), idx] += 1.0
        votes /= len(models)
        probs0 = _filtered_posteriors(models[0], Xs)
        return votes[:, 0], votes[:, 1], votes[:, 2], probs0
    return voted


_emdbm_raw = ensemble_direction_masses_by_mode


@contextlib.contextmanager
def mass_override(fn):
    g = globals()
    old = g['ensemble_direction_masses_by_mode']
    g['ensemble_direction_masses_by_mode'] = fn
    try:
        yield
    finally:
        g['ensemble_direction_masses_by_mode'] = old


def label_one(models, a):
    '''Label the full series for ONE seed set of ONE arm.'''
    Xs = scaled_X(a['features'])
    kw = dict(a['lab'])
    kw.setdefault('dir_feats', FEAT_DF)
    ms = list(models)

    if a['post'] == 'll_prune':
        ms, lls, nkept = prune_by_ll(ms, Xs[:N_FIT])
        a.setdefault('_prune_kept', []).append(nkept)

    need_fn = None
    if a['pol'] == 'L2':
        b, s, r = _masses_for(ms, a['features'])
        strength = np.abs(b - r)
        thr = float(np.quantile(strength[:N_FIT], 1.0 - L2_FAST_RATE))
        a.setdefault('_l2_thr', []).append(thr)
        a.setdefault('_l2_rate', []).append(float(np.mean(strength >= thr)))
        need_fn = make_pol_L2(strength, thr)
    elif a['pol'] == 'L3':
        need_fn = make_pol_L3()

    stack = contextlib.ExitStack()
    with stack:
        if need_fn is not None:
            stack.enter_context(confirm_policy(need_fn))
        if a['mass']:
            stack.enter_context(mass_override(_vote_masses_factory()))
        out = label_bars(ms, Xs, DATES, nifty, TREND_RAW, N_FIT,
                         list(a['features']), **kw)
    return out['tactical_regime_state'].values.copy()


t0 = time.time()
LABELS = {}                 # arm name -> [labels_seed_set_0 .. R-1]
for a in ARMS:
    fits = get_fits(a['features'], a['N'], a['cov'], a['K'], a['init'])
    LABELS[a['name']] = [label_one(ms, a) for ms in fits]
    print(f"  labelled {a['name']:<16} K={a['K']:<3} "
          f"N={a['N']} cov={a['cov']:<4} feats={len(a['features'])}  "
          f"({time.time() - t0:5.1f}s cumulative)")

LABEL_PHASE_SECS = time.time() - t0
print(f'\nLABEL PHASE COMPLETE in {LABEL_PHASE_SECS:.1f}s   '
      f'{FIT_COUNT} HMM fits, {LABEL_CALLS} label_bars calls')
assert ZZ_CALLS == 0, 'LEAKAGE: a ZigZag existed during the label phase'
print('ASSERT OK: ZZ_CALLS == 0 -- every label in this notebook was produced '
      'before any ZigZag was computed.')

# S4 sanity: a seed-free fit must be identical across the disjoint seed sets.
_d4 = get_fits(BASE_FEATURES, BASE_N, BASE_COV, ENSEMBLE_K, 'det')
assert all(np.array_equal(LABELS['S4 det-init'][0], x) for x in LABELS['S4 det-init']), \
    'deterministic init produced seed-dependent labels'
print('ASSERT OK: S4 deterministic init gives IDENTICAL labels in all '
      f'{R_SEED_SETS} seed sets -> its S is 100% BY CONSTRUCTION.')
""")

co(r"""
# ===========================================================================
# EVAL PHASE BEGINS. The label phase is closed and the ZigZag may be computed.
# ===========================================================================
LABEL_PHASE_OPEN = False
print('LABEL PHASE CLOSED. Any label_bars call from here on fails the ordering '
      'check unless the phase is EXPLICITLY and loudly reopened.')

SWINGS = zigzag_swings(CLOSE, ZZ_PCT)
_moves = [100 * (CLOSE[b] - CLOSE[a]) / CLOSE[a] for a, b, _ in SWINGS]
print(f'ZigZag ({ZZ_PCT}% reversal) found {len(SWINGS)} swings over {N_BARS} bars.')
print(f'  swing length  : median {np.median([b - a for a, b, _ in SWINGS]):.0f} bars, '
      f'max {max(b - a for a, b, _ in SWINGS)}')
print(f'  swing size    : median |move| {np.median(np.abs(_moves)):.2f}%, '
      f'max {np.max(np.abs(_moves)):.2f}%')
print(f'  direction mix : {sum(1 for *_, s in SWINGS if s > 0)} up / '
      f'{sum(1 for *_, s in SWINGS if s < 0)} down')
print(f'ZZ_CALLS = {ZZ_CALLS}  -- from here on, ANY label_bars call would fail '
      'the tripwire.')


def summarise_arm(name, label_sets, note='', group='', rows_per_param=np.nan):
    '''S, L and W TOGETHER, plus occupancy, guards and the seed-set spreads.

    ONE function, used by every real arm AND by the degeneracy stubs below, so
    the guard test exercises the shipping path rather than a copy of it.
    '''
    S, M, pair_vals = metric_S(label_sets)
    L_med = [metric_L(lb, SWINGS)[0] for lb in label_sets]
    L_p75 = [metric_L(lb, SWINGS)[1] for lb in label_sets]
    unm = [metric_L(lb, SWINGS)[3] for lb in label_sets]
    Ws = [metric_W(lb) for lb in label_sets]
    occs = [occupancy_pct(lb) for lb in label_sets]
    occ = {lb: float(np.mean([o[lb] for o in occs])) for lb in REGIME_LABELS}
    W = float(np.mean(Ws))
    g = evaluate_guards(occ, W)
    return dict(name=name, group=group, note=note,
                S=S, S_min=min(pair_vals), S_max=max(pair_vals), S_matrix=M,
                L=float(np.mean(L_med)), L_lo=float(np.min(L_med)),
                L_hi=float(np.max(L_med)), L75=float(np.mean(L_p75)),
                unmatched=float(np.mean(unm)),
                W=W, W_lo=float(np.min(Ws)), W_hi=float(np.max(Ws)),
                occ=occ, guards=g, void=not guards_pass(g),
                rows_per_param=rows_per_param)


# ---- TEST 3: G4 MUST BITE on a degenerate all-SIDEWAYS labelling ------------
_stub_side = [np.full(N_BARS, 'SIDEWAYS', dtype='<U8') for _ in range(R_SEED_SETS)]
_r4 = summarise_arm('STUB all-SIDEWAYS', _stub_side, 'degeneracy stub', 'test')
print(f"\nTEST 3  degeneracy stub: S={_r4['S']:.2f}%  L={_r4['L']:.1f}  W={_r4['W']:.2f}  "
      f"SIDEWAYS={_r4['occ']['SIDEWAYS']:.1f}%")
for _k, (_p, _why) in _r4['guards'].items():
    print(f"        {_k} {'PASS' if _p else 'FAIL'}  {_why}")
assert _r4['S'] == 100.0, 'the stub should score a PERFECT S -- that is the point'
assert not _r4['guards']['G4'][0], 'G4 FAILED TO BITE on an all-SIDEWAYS labelling'
assert _r4['void'], 'the degenerate stub was not VOIDED'
print('TEST 3 PASS  a perfect S=100.00% is VOIDED by G4 (and G1/G2). '
      'S cannot be gamed by collapsing to SIDEWAYS.')

# ---- TEST 4: G3 MUST BITE on a high-whipsaw labelling -----------------------
_alt = np.array([REGIME_LABELS[i % 5] for i in range(N_BARS)], dtype='<U8')
_r3 = summarise_arm('STUB whipsaw', [_alt] * R_SEED_SETS, 'whipsaw stub', 'test')
assert not _r3['guards']['G3'][0], 'G3 FAILED TO BITE at W=100/100 bars'
print(f"TEST 4 PASS  W={_r3['W']:.1f}/100 bars is VOIDED by G3 "
      f"(max {W_GUARD_MAX}).")
""")

# ===========================================================================
md(r"""
## 7. Results — S, L and W together, on every row. Baseline is row 1.
""")

co(r"""
# ===========================================================================
# SUMMARISE EVERY ARM.
# ===========================================================================
def rpp(a):
    p = n_params(a['N'], len(a['features']), a['cov'])
    return round(N_FIT / p, 2)


RESULTS = [summarise_arm(a['name'], LABELS[a['name']], a['note'], a['group'], rpp(a))
           for a in ARMS]
RES = {r['name']: r for r in RESULTS}
BASE = RES['BASELINE']

rows = []
for r in RESULTS:
    g = r['guards']
    rows.append({
        'arm': r['name'], 'group': r['group'],
        'S %': round(r['S'], 2),
        'L med': round(r['L'], 2), 'L p75': round(r['L75'], 2),
        'W /100': round(r['W'], 2),
        'unmat': round(r['unmatched'], 1),
        'rows/par': r['rows_per_param'],
        'SIDE %': round(r['occ']['SIDEWAYS'], 1),
        'G1': 'P' if g['G1'][0] else 'F', 'G2': 'P' if g['G2'][0] else 'F',
        'G3': 'P' if g['G3'][0] else 'F', 'G4': 'P' if g['G4'][0] else 'F',
        'verdict': 'VOID' if r['void'] else 'ok',
        'dS': round(r['S'] - BASE['S'], 2),
        'dL': round(r['L'] - BASE['L'], 2),
        'dW': round(r['W'] - BASE['W'], 2),
    })
TAB = pd.DataFrame(rows)
pd.set_option('display.width', 200, 'display.max_columns', 40)
print('S = mean pairwise 5-label agreement over R=%d DISJOINT seed sets (higher better)'
      % R_SEED_SETS)
print('L = ZigZag transition lag in bars, mean over seed sets (lower better)')
print('W = label switches per 100 bars (guard: must not get worse)')
print(f'unmat = ZigZag swings NEVER labelled in the right direction (of '
      f'{len(SWINGS)}); each one already scores the FULL swing length inside L')
print('=' * 150)
print(TAB.to_string(index=False))
print('=' * 150)
print('NO COMPOSITE SCORE IS COMPUTED. The three columns are not combined.')

# ---- did the switchable interventions actually FIRE? -----------------------
# An intervention that never engages is a null for a boring reason, and that has
# to be distinguishable from one that engaged and did not help.
_armmap = {a['name']: a for a in ARMS}
_a2 = _armmap['L2 adaptive']
if '_l2_thr' in _a2:
    print(f"L2 diagnostic : threshold |bull-bear| >= "
          f"{np.mean(_a2['_l2_thr']):.4f} (fit-window p{100*(1-L2_FAST_RATE):.0f}); "
          f"the 1-bar fast path was eligible on "
          f"{100*np.mean(_a2['_l2_rate']):.1f}% of all bars")
_a3 = _armmap['S2 LL-prune']
if '_prune_kept' in _a3:
    _kept = float(np.mean(_a3['_prune_kept']))
    print(f"S2 diagnostic : kept {_kept:.2f} of {_a3['K']} ensemble members on "
          f"average (threshold {S2_LL_FRAC:.0%} of the LL spread below the best)")
    if _kept <= _a3['K'] / 2:
        print('              *** S2 IS DE-ENSEMBLING, NOT PRUNING. The across-seed LL')
        print('              spread is wide enough that a "fraction of the spread" rule')
        print(f'              discards most of the ensemble, leaving ~{_kept:.1f} members.')
        print('              That is the OPPOSITE of what S2 was meant to do, and it is')
        print('              why its S is worse, not better. The threshold is NOT being')
        print('              moved to fix this -- it was declared before the run and any')
        print('              retune would be fitting the knob to the answer. The honest')
        print('              read is that S2 as SPECIFIED is mis-specified: an absolute')
        print('              LL-gap rule, or a rule keeping at least K/2 members, would')
        print('              be the thing to pre-declare next time.')
print(f"L4 diagnostic : dL = {RES['L4 hold=allow']['L'] - BASE['L']:+.2f} bars. "
      "ESCALATION_DURING_HOLD acts on the H<->L INTENSITY axis only, so it CANNOT "
      "move a DIRECTION-lag metric. This null was predicted in section 5 before "
      "the run; dL == 0 is the evidence, not a disappointment.")
""")

co(r"""
# ===========================================================================
# THE FULL PAIRWISE AGREEMENT MATRICES (protocol requires the matrix, not just
# the mean) -- printed for the baseline and for every S-axis arm.
# ===========================================================================
for _n in ['BASELINE'] + [a['name'] for a in ARMS if a['group'] == 'S']:
    _r = RES[_n]
    print(f"\n{_n}   S = {_r['S']:.2f}%   pairwise range "
          f"[{_r['S_min']:.2f}, {_r['S_max']:.2f}]")
    _m = pd.DataFrame(_r['S_matrix'],
                      index=[f'set{i}' for i in range(R_SEED_SETS)],
                      columns=[f'set{i}' for i in range(R_SEED_SETS)])
    print(_m.round(2).to_string())
""")

co(r"""
# ===========================================================================
# HEADROOM CHECK -- REQUIRED BEFORE ANY VERDICT.
# The question: can a candidate's improvement even be distinguished from the
# seed-to-seed noise of the measurement itself?
# ===========================================================================
S_SPREAD = BASE['S_max'] - BASE['S_min']         # spread of the 6 pairwise values
L_SPREAD = BASE['L_hi'] - BASE['L_lo']           # spread of L across seed sets
W_SPREAD = BASE['W_hi'] - BASE['W_lo']           # spread of W across seed sets

print('HEADROOM CHECK (baseline, across the R=%d disjoint seed sets)' % R_SEED_SETS)
print('-' * 78)
print(f"  S : {BASE['S']:.2f}%   pairwise values span "
      f"{BASE['S_min']:.2f} -> {BASE['S_max']:.2f}   spread = {S_SPREAD:.2f} pp")
print(f"  L : {BASE['L']:.2f} bars   per-seed-set span "
      f"{BASE['L_lo']:.2f} -> {BASE['L_hi']:.2f}   spread = {L_SPREAD:.2f} bars")
print(f"  W : {BASE['W']:.2f}/100   per-seed-set span "
      f"{BASE['W_lo']:.2f} -> {BASE['W_hi']:.2f}   spread = {W_SPREAD:.2f}")
print('-' * 78)
print('RULE (pre-declared): a candidate whose improvement is SMALLER than the '
      'corresponding spread is NOISE, not a result.')


def is_noise(r):
    return (abs(r['S'] - BASE['S']) <= S_SPREAD) and (abs(r['L'] - BASE['L']) <= L_SPREAD)


_noisy = [r['name'] for r in RESULTS if r['name'] != 'BASELINE' and is_noise(r)]
print(f"\n{len(_noisy)} of {len(RESULTS) - 1} arms move BOTH S and L by less than the "
      f"baseline seed spread -> indistinguishable from noise:")
print('  ' + (', '.join(_noisy) if _noisy else '(none)'))
""")

# ===========================================================================
md(r"""
## 8. Combinations

Measured **after** every intervention has been measured alone, and deliberately
few: the runtime that would go into a full S×L grid is worth more spent on `R=4`
and on the config arms. The combinations tried are the best-on-S intervention
crossed with the best-on-L intervention, plus the best-on-S *config* crossed with
the best-on-L intervention.
""")

co(r"""
# ===========================================================================
# COMBINATIONS -- chosen from the measured single-intervention results.
# All of them REUSE already-cached fits, so this section adds no HMM fits
# unless a config arm brings its own (already-cached) fit set.
# ===========================================================================
assert ZZ_CALLS > 0, 'combination section must run in the EVAL phase'


def _best(group, key, lower_better):
    all_g = [r for r in RESULTS if r['group'] == group]
    cand = [r for r in all_g if not r['void']]
    if not cand:
        cand = all_g
        print(f'  NOTE: every {group} arm is VOID; picking on the metric anyway '
              f'(the combination inherits the VOID and is reported as such).')
    elif len(cand) < len(all_g):
        print(f'  NOTE: {len(all_g) - len(cand)} of {len(all_g)} {group} arms are '
              f'VOID and are NOT selectable, so the pick below can be worse on '
              f'{key} than a VOID arm. That is the guard doing its job, not a bug.')
    return min(cand, key=lambda r: (r[key] if lower_better else -r[key]))


_bS = _best('S', 'S', False)
_bL = _best('L', 'L', True)
_bCap = _best('capacity', 'S', False)
print(f"best S-axis intervention : {_bS['name']}  (S={_bS['S']:.2f})")
print(f"best L-axis intervention : {_bL['name']}  (L={_bL['L']:.2f})")
print(f"best capacity arm on S   : {_bCap['name']}  (S={_bCap['S']:.2f}, "
      f"rows/par={_bCap['rows_per_param']})")

_armof = {a['name']: a for a in ARMS}
COMBOS = []


def combine(fit_name, lab_name, label):
    '''Fit-side parameters come from `fit_name`, labelling-side from BOTH.'''
    A, B = _armof[fit_name], _armof[lab_name]
    c = arm(label, 'combo', f'{fit_name} + {lab_name}',
            features=A['features'], N=A['N'], cov=A['cov'],
            K=A['K'], init=A['init'],
            lab={**A['lab'], **B['lab']},
            post=A['post'] or B['post'],
            pol=A['pol'] or B['pol'],
            mass=A['mass'] or B['mass'])
    COMBOS.append(c)
    return c


combine(_bS['name'], _bL['name'], f"C1 {_bS['name']}+{_bL['name']}")
combine(_bCap['name'], _bL['name'], f"C2 {_bCap['name']}+{_bL['name']}")

# The combination arms cannot be named until the singles have been measured, and
# measuring needs the ZigZag. So the label phase is reopened -- EXPLICITLY, and
# loudly, and only for its ordering check. Object-identity, memory-aliasing and
# the w == 0.0 decree stay armed the whole time, so a ZigZag array still cannot
# reach a label here.
_zz_before = (ZZ_CALLS, len(ZZ_ARRAYS), [id(z) for z in ZZ_ARRAYS])
with reopen_label_phase('combination arms only'):
    for c in COMBOS:
        fits = get_fits(c['features'], c['N'], c['cov'], c['K'], c['init'])
        LABELS[c['name']] = [label_one(ms, c) for ms in fits]
        print(f"  labelled {c['name']}")
assert not LABEL_PHASE_OPEN
assert (ZZ_CALLS, len(ZZ_ARRAYS), [id(z) for z in ZZ_ARRAYS]) == _zz_before, \
    'the ZigZag state changed during the reopened label phase'
print('ASSERT OK: no ZigZag was computed and no ZigZag array was touched while the '
      'label phase was reopened.')

for c in COMBOS:
    RESULTS.append(summarise_arm(c['name'], LABELS[c['name']], c['note'],
                                 'combo', rpp(c)))
    ARMS.append(c)
RES = {r['name']: r for r in RESULTS}
BASE = RES['BASELINE']
_armof = {a['name']: a for a in ARMS}          # rebuilt to include the combos

_cr = pd.DataFrame([{
    'arm': r['name'], 'S %': round(r['S'], 2), 'L med': round(r['L'], 2),
    'W /100': round(r['W'], 2), 'SIDE %': round(r['occ']['SIDEWAYS'], 1),
    'guards': ' '.join(k for k, v in r['guards'].items() if not v[0]) or 'all pass',
    'verdict': 'VOID' if r['void'] else 'ok',
    'dS': round(r['S'] - BASE['S'], 2), 'dL': round(r['L'] - BASE['L'], 2),
    'dW': round(r['W'] - BASE['W'], 2),
} for r in RESULTS if r['group'] in ('baseline', 'combo')])
print()
print(_cr.to_string(index=False))
""")

# ===========================================================================
md(r"""
## 9. Verdict — strict dominance only

A candidate wins **only** if, versus the baseline:

`S` higher **and** `L` lower **and** `W` not worse **and** all four guards pass.

There is no composite score and no tie-break. If nothing clears that bar, the
notebook prints `NO DOMINATING CONFIGURATION` and shows the frontier — which the
protocol says is the *likely* honest outcome, because S and L oppose each other.
""")

co(r"""
# ===========================================================================
# VERDICT.
# ===========================================================================
print('=' * 100)
if BASE['void']:
    print('*** WARNING: THE BASELINE ITSELF IS VOID ON THIS DATA. ***')
    for k, (p, why) in BASE['guards'].items():
        if not p:
            print(f'    {k} FAIL: {why}')
    if TAC_SYNTH:
        print('    This is a SYNTHETIC series whose occupancy profile is not the real')
        print("    market's. It is a plumbing observation, NOT a finding about RegDet.")
    print('    Dominance is still evaluated below, relative to that baseline, but no')
    print('    verdict from this run is decision-grade.')
    print('=' * 100)

DOMINATING = []
for r in RESULTS:
    if r['name'] == 'BASELINE':
        continue
    if r['void']:
        continue
    if r['S'] > BASE['S'] and r['L'] < BASE['L'] and r['W'] <= BASE['W']:
        DOMINATING.append(r)

if DOMINATING:
    print('DOMINATING CONFIGURATION(S) FOUND — S up, L down, W not worse, all guards pass:')
    for r in DOMINATING:
        _n = is_noise(r)
        print(f"  {r['name']:<22} S {BASE['S']:.2f} -> {r['S']:.2f}   "
              f"L {BASE['L']:.2f} -> {r['L']:.2f}   W {BASE['W']:.2f} -> {r['W']:.2f}"
              + ('   *** INSIDE THE SEED SPREAD -> NOISE, NOT A RESULT ***' if _n else ''))
    DOMINATING = [r for r in DOMINATING if not is_noise(r)]
    if not DOMINATING:
        print('\n  ...and every one of them is inside the baseline seed spread.')
        print('  NO DOMINATING CONFIGURATION (after the headroom check).')
else:
    print('NO DOMINATING CONFIGURATION')
    print()
    print('No candidate improved S AND L without making W worse, with all guards')
    print('passing. This is the outcome the frozen protocol named as likely: lower')
    print('lag means reacting to smaller moves, which means more switches and more')
    print('sensitivity to the fit. The deliverable is the FRONTIER below, not a fix.')

# ---- the frontier: arms not dominated by any other arm on (L down, S up) ----
_live = [r for r in RESULTS if not r['void']]
FRONTIER = [r for r in _live
            if not any(o is not r and o['L'] <= r['L'] and o['S'] >= r['S']
                       and (o['L'] < r['L'] or o['S'] > r['S']) for o in _live)]
print()
print(f'PARETO FRONTIER on (L lower, S higher) over the {len(_live)} non-VOID arms:')
for r in sorted(FRONTIER, key=lambda x: x['L']):
    print(f"  L={r['L']:6.2f}  S={r['S']:6.2f}%  W={r['W']:5.2f}  "
          f"{r['name']}{'   <-- BASELINE' if r['name'] == 'BASELINE' else ''}")
if not any(r['name'] == 'BASELINE' for r in FRONTIER):
    print('  (the baseline is NOT on the frontier -- it is dominated by the arms above)')
print('=' * 100)
""")

co(r"""
# ===========================================================================
# THE CAPACITY HYPOTHESIS, tested rather than assumed.
# ===========================================================================
_cap = [r for r in RESULTS if r['group'] in ('capacity', 'config', 'baseline')]
_x = np.array([r['rows_per_param'] for r in _cap], dtype=float)
_y = np.array([r['S'] for r in _cap], dtype=float)
_rho, _p = spearmanr(_x, _y)
print('CAPACITY -> STABILITY')
print('-' * 78)
print(pd.DataFrame([{'arm': r['name'], 'rows/par': r['rows_per_param'],
                     'params': int(n_params(_armof[r['name']]['N'],
                                            len(_armof[r['name']]['features']),
                                            _armof[r['name']]['cov'])),
                     'S %': round(r['S'], 2), 'L med': round(r['L'], 2),
                     'W /100': round(r['W'], 2),
                     'SIDE %': round(r['occ']['SIDEWAYS'], 1),
                     'verdict': 'VOID' if r['void'] else 'ok'}
                    for r in sorted(_cap, key=lambda r: -r['rows_per_param'])
                    ]).to_string(index=False))
print('-' * 78)
print(f'Spearman rho(rows-per-param, S) = {_rho:+.3f}   p = {_p:.4f}   n = {len(_x)}')
_S_range_cap = max(r['S'] for r in _cap) - min(r['S'] for r in _cap)
_S_range_int = (max(r['S'] for r in RESULTS if r['group'] == 'S')
                - min(r['S'] for r in RESULTS if r['group'] == 'S'))
print(f'S range across CAPACITY/CONFIG arms : {_S_range_cap:.2f} pp')
print(f'S range across S1-S4 INTERVENTIONS  : {_S_range_int:.2f} pp')
print(f'baseline seed spread on S           : {S_SPREAD:.2f} pp')
if _p < 0.05 and _rho > 0:
    print('\n=> CAPACITY DRIVES STABILITY: fewer free parameters per training row goes')
    print('   with a MORE stable fit, monotonically and significantly. Config choice is')
    print('   a real stability lever, and it is measurable where forward-return ranking')
    print('   (defect 4, the "config lottery") was not.')
elif _p < 0.05 and _rho < 0:
    print('\n=> The relationship runs the WRONG WAY: more parameters per row goes with')
    print('   MORE stability here. The capacity hypothesis is not supported.')
else:
    print('\n=> NO SIGNIFICANT capacity->stability relationship at this sample size.')
    print('   The capacity hypothesis is NOT confirmed by this run. Reported as a null,')
    print('   not spun. NOTE: guards must be read alongside this -- a low-capacity arm')
    print('   that wins S by collapsing toward one label is VOID, not stable.')
""")

# ===========================================================================
md(r"""
## 10. Figures

Four figures. The first two are the frontiers the protocol asks for, the third
tests the capacity hypothesis visually, and the fourth is the **reference case** —
the one the lag fix has to survive being looked at.
""")

co(r"""
# ===========================================================================
# FIG 1 - the (L, S) frontier.  FIG 2 - the (L, W) frontier.
# ===========================================================================
SAVED_PNGS = []


def save_fig(fig, fname, dpi=130):
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    SAVED_PNGS.append(fname)
    print(f'   saved -> {fname}')
    return fname


GROUP_STYLE = {'baseline': ('#d62728', '*', 420),
               'S':        ('#1f77b4', 'o', 90),
               'L':        ('#2ca02c', 's', 90),
               'config':   ('#9467bd', 'D', 80),
               'capacity': ('#8c564b', '^', 80),
               'combo':    ('#ff7f0e', 'P', 130)}


def frontier_panel(ax, ykey, ylabel, better_note):
    # VOID arms are drawn hollow, so a guard failure is visible on the chart and
    # cannot be mistaken for a usable point on the frontier.
    for grp, (c, mk, sz) in GROUP_STYLE.items():
        ok = [r for r in RESULTS if r['group'] == grp and not r['void']]
        vd = [r for r in RESULTS if r['group'] == grp and r['void']]
        if ok:
            ax.scatter([r['L'] for r in ok], [r[ykey] for r in ok], c=c, marker=mk,
                       s=sz, zorder=4, edgecolors='black', linewidths=0.6, label=grp)
        if vd:
            ax.scatter([r['L'] for r in vd], [r[ykey] for r in vd], facecolors='none',
                       marker=mk, s=sz, zorder=4, edgecolors=c, linewidths=1.4,
                       alpha=0.75, label=f'{grp} (VOID)')
    # ---- de-overlapped annotations -------------------------------------
    # Many arms land on IDENTICAL (L, S) coordinates -- L is a median in whole
    # bars, so ties are the norm, not the exception. Annotating them at the same
    # offset renders an unreadable pile. Coincident points are clustered and
    # their labels stacked vertically, so every arm stays legible and its
    # membership of the cluster stays obvious.
    ax.relim(); ax.autoscale_view()
    _xr = np.ptp(ax.get_xlim()) or 1.0
    _yr = np.ptp(ax.get_ylim()) or 1.0
    clusters = []
    for r in sorted(RESULTS, key=lambda r: (r['L'], -r[ykey])):
        for c in clusters:
            if (abs(r['L'] - c[0][0]) <= 0.09 * _xr
                    and abs(r[ykey] - c[0][1]) <= 0.03 * _yr):
                c[1].append(r)
                break
        else:
            clusters.append(([r['L'], r[ykey]], [r]))
    # A little room on the right so the right-most name is not clipped. The
    # y-axis is NOT stretched: S is a percentage and an axis running past 100%
    # would be nonsense. Instead a stack that sits high in the panel is written
    # DOWNWARDS, into the empty space below it.
    _x0, _x1 = ax.get_xlim()
    ax.set_xlim(_x0, _x1 + 0.12 * (_x1 - _x0))
    _mid = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
    _rx0, _rx1 = ax.get_xlim()
    for (cx, cy), members in clusters:
        down = cy > _mid
        # a cluster near the right edge writes LEFTWARDS, so its stack does not
        # run off the panel or collide with whatever sits to its right
        left = (cx - _rx0) / (_rx1 - _rx0) > 0.72
        for k, r in enumerate(members):
            dy = (-1 if down else 1) * (6 + 9.0 * k)
            # names are truncated on the chart only; the results table above
            # carries every name in full
            _nm = r['name'] if len(r['name']) <= 19 else r['name'][:18] + '~'
            ax.annotate(_nm + (' [VOID]' if r['void'] else ''), (cx, cy),
                        fontsize=6.5, xytext=(-9 if left else 8, dy),
                        textcoords='offset points',
                        ha='right' if left else 'left',
                        va='top' if down else 'bottom',
                        color='#999999' if r['void'] else '#222222',
                        arrowprops=dict(arrowstyle='-', lw=0.4, color='#bbbbbb',
                                        shrinkA=0, shrinkB=2) if len(members) > 1
                        else None)
    ax.axvline(BASE['L'], color='#d62728', ls=':', lw=1.0, zorder=1)
    ax.axhline(BASE[ykey], color='#d62728', ls=':', lw=1.0, zorder=1)
    ax.set_xlabel('L  —  transition lag, median bars  (LOWER IS BETTER  <—)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.set_title(better_note, fontsize=10, loc='left')


fig, ax = plt.subplots(figsize=(13, 8))
frontier_panel(ax, 'S', 'S  —  mean pairwise label agreement %  (HIGHER IS BETTER)',
               'The dotted red cross is the BASELINE. The UPPER-LEFT quadrant is the '
               'only place a dominating candidate can live.')
_pf = sorted(FRONTIER, key=lambda x: x['L'])
ax.plot([r['L'] for r in _pf], [r['S'] for r in _pf], color='#444444', lw=1.2,
        ls='--', zorder=2, label='Pareto frontier (L down, S up)')
# The ONLY region where a dominating candidate can live: strictly lower L AND
# strictly higher S than the baseline. Shaded so the emptiness is the message.
_x0, _x1 = ax.get_xlim(); _y0, _y1 = ax.get_ylim()
ax.add_patch(plt.Rectangle((_x0, BASE['S']), BASE['L'] - _x0, _y1 - BASE['S'],
                           color='#2ca02c', alpha=0.07, zorder=0))
ax.annotate('DOMINATING QUADRANT\n(lower lag AND more stable)',
            ((_x0 + BASE['L']) / 2, _y1), fontsize=9, color='#2ca02c',
            ha='center', va='top', fontweight='bold')
ax.set_xlim(_x0, _x1); ax.set_ylim(_y0, _y1)
ax.legend(fontsize=8, loc='lower right', ncol=2)
fig.suptitle('FIG 1 — the (L, S) FRONTIER: transition lag vs fit stability'
             + ('  [SYNTHETIC - ILLUSTRATIVE ONLY]' if TAC_SYNTH else '  [real data]'),
             fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig1_frontier_L_vs_S.png')
plt.show()

fig, ax = plt.subplots(figsize=(13, 8))
frontier_panel(ax, 'W', 'W  —  switches per 100 bars  (LOWER IS BETTER — this is the '
                        'anti-gaming guard)',
               f'G3 voids anything above W = {W_GUARD_MAX}. A lag fix that cuts L by '
               'inflating W has bought nothing.')
ax.axhline(W_GUARD_MAX, color='black', ls='--', lw=1.2)
ax.annotate(f'G3 limit  W = {W_GUARD_MAX}', (ax.get_xlim()[0], W_GUARD_MAX),
            fontsize=9, va='bottom', ha='left')
fig.suptitle('FIG 2 — the (L, W) FRONTIER: does lower lag simply buy more whipsaw?'
             + ('  [SYNTHETIC - ILLUSTRATIVE ONLY]' if TAC_SYNTH else '  [real data]'),
             fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig2_frontier_L_vs_W.png')
plt.show()
""")

co(r"""
# ===========================================================================
# FIG 3 - the capacity hypothesis: S against rows-per-parameter.
# ===========================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
_cx = np.array([r['rows_per_param'] for r in _cap], float)
_cy = np.array([r['S'] for r in _cap], float)
_cv = np.array([r['void'] for r in _cap])
axA.scatter(_cx[~_cv], _cy[~_cv], s=110, c='#1f77b4', edgecolors='black',
            linewidths=0.7, zorder=3, label='guards pass')
axA.scatter(_cx[_cv], _cy[_cv], s=110, c='none', edgecolors='#d62728',
            linewidths=1.4, zorder=3, label='VOID (guard failed)')
for r in _cap:
    axA.annotate(r['name'], (r['rows_per_param'], r['S']), fontsize=7,
                 xytext=(5, 4), textcoords='offset points')
if len(_cx) >= 2 and np.ptp(_cx) > 0:
    _b = np.polyfit(_cx, _cy, 1)
    _xx = np.linspace(_cx.min(), _cx.max(), 50)
    axA.plot(_xx, np.polyval(_b, _xx), color='#888888', ls='--', lw=1.2,
             label=f'least squares (slope {_b[0]:+.3f} pp per row/param)')
axA.set_xlabel('rows per free parameter  (HIGHER = lower capacity = the hypothesis '
               'says MORE stable)', fontsize=9)
axA.set_ylabel('S — mean pairwise label agreement %', fontsize=10)
axA.set_title(f'A. capacity vs stability   Spearman rho = {_rho:+.3f}  (p = {_p:.4f})',
              fontsize=11, loc='left')
axA.grid(alpha=0.25)
axA.legend(fontsize=8)

_w = 0.35
_ix = np.arange(len(_cap))
_ord = sorted(range(len(_cap)), key=lambda i: -_cap[i]['rows_per_param'])
axB.bar(_ix - _w / 2, [_cap[i]['S'] for i in _ord], _w, color='#1f77b4', label='S %')
axB2 = axB.twinx()
axB2.bar(_ix + _w / 2, [_cap[i]['L'] for i in _ord], _w, color='#2ca02c', label='L med')
axB.set_xticks(_ix)
axB.set_xticklabels([_cap[i]['name'] for i in _ord], rotation=40, ha='right', fontsize=7)
axB.set_ylabel('S %', color='#1f77b4')
axB2.set_ylabel('L median bars', color='#2ca02c')
axB.axhline(BASE['S'], color='#d62728', ls=':', lw=1.2)
# S sits in a narrow band near 100%; a zero-anchored axis would hide every
# difference the panel exists to show. Limits are set to the DATA range.
_slo = min(r['S'] for r in _cap)
axB.set_ylim(_slo - 0.05 * (100 - _slo) - 0.5, 100.6)
axB2.set_ylim(0, max(r['L'] for r in _cap) * 1.25)
axB.set_title('B. S (blue, left — NOTE: axis is NOT zero-anchored) and L (green, '
              'right), ordered by rows/param.\n   dotted red = baseline S; '
              'RED tick labels = VOID (a guard failed)', fontsize=9, loc='left')
for _t, i in zip(axB.get_xticklabels(), _ord):
    if _cap[i]['void']:
        _t.set_color('#d62728')
        _t.set_fontweight('bold')
fig.suptitle('FIG 3 — does MODEL CAPACITY drive fit instability?'
             + ('  [SYNTHETIC - ILLUSTRATIVE ONLY]' if TAC_SYNTH else '  [real data]'),
             fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig3_capacity_vs_stability.png')
plt.show()
""")

co(r"""
# ===========================================================================
# FIG 4 - THE REFERENCE CASE: the May-2025 V-bottom window, baseline vs the best
# lag candidate, in the MASTER NOTEBOOK'S regime-background style:
#   black price line, FULL-HEIGHT regime bands, TIGHT y-limits (NOT zero-anchored),
#   asserted EQUAL across panels.
# ===========================================================================
REF_START, REF_END = pd.Timestamp('2025-05-05'), pd.Timestamp('2025-05-26')

_m = (DATES >= REF_START) & (DATES <= REF_END)
if _m.sum() < 6:
    # The reference window is outside this data span (can happen on the synthetic
    # fallback or a short yfinance window). Fall back to the LARGEST up-swing, and
    # say so on the figure rather than silently plotting something else.
    _up = max([s for s in SWINGS if s[2] > 0],
              key=lambda s: (CLOSE[s[1]] - CLOSE[s[0]]) / CLOSE[s[0]])
    _a, _b = max(0, _up[0] - 12), min(N_BARS - 1, _up[1] + 12)
    _m = np.zeros(N_BARS, bool); _m[_a:_b + 1] = True
    REF_NOTE = (f'REFERENCE WINDOW {REF_START:%Y-%m-%d} -> {REF_END:%Y-%m-%d} IS NOT IN '
                f'THIS DATA SPAN — substituted the largest up-swing in the series')
else:
    REF_NOTE = f'REFERENCE CASE: {REF_START:%Y-%m-%d} -> {REF_END:%Y-%m-%d}'

REF_IDX = DATES[_m]
REF_CLOSE = CLOSE[_m]
REF_I0 = int(np.flatnonzero(_m)[0])
_ref_move = 100 * (REF_CLOSE[-1] - REF_CLOSE.min()) / REF_CLOSE.min()

# the best LAG candidate = lowest L among non-VOID L-axis and combo arms
_lagcands = [r for r in RESULTS if r['group'] in ('L', 'combo') and not r['void']]
if not _lagcands:
    _lagcands = [r for r in RESULTS if r['group'] in ('L', 'combo')]
    print('NOTE: every lag candidate is VOID; the chart shows the lowest-L one anyway, '
          'marked VOID.')
BEST_LAG = min(_lagcands, key=lambda r: r['L'])
print(f"reference-case comparison: BASELINE  vs  {BEST_LAG['name']}"
      f"{' [VOID]' if BEST_LAG['void'] else ''}")

def shade_regimes_contiguous(ax, series, alpha=0.35):
    '''Regime bands that BUTT UP against each other, with no unpainted gaps.

    The master's `regime_blocks` ends a band on the LAST BAR of its run, so the
    interval between one run's last bar and the next run's first bar is left
    white. Over 2900 bars that is invisible; on a 45-bar zoom that crosses a
    weekend it is a conspicuous white stripe, and it reads as "no regime here",
    which is false -- the regime holds across the gap.

    Each band is therefore extended to the START of the next band. The PAINTER is
    still the master's `shade_bands`, unmodified, so colour, alpha, full-height
    geometry and the autolim fix are all inherited rather than reimplemented.
    '''
    vals, idx = np.asarray(series.values), series.index
    spans, start = [], 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            spans.append((vals[start], idx[start], idx[i]))
            start = i
    spans.append((vals[start], idx[start], idx[-1]))
    shade_bands(ax, spans, alpha=alpha)


_panels = [('BASELINE', BASE), (BEST_LAG['name'], BEST_LAG)]
fig, axes = plt.subplots(len(_panels), 1, figsize=(16, 9), sharex=True)
_ylim = None
for ax, (nm, r) in zip(axes, _panels):
    lab = pd.Series(LABELS[nm][0], index=DATES)[_m]      # seed set 0, both panels
    shade_regimes_contiguous(ax, lab, alpha=0.35)
    ax.plot(REF_IDX, REF_CLOSE, color='black', linewidth=1.7, zorder=4)
    set_price_ylim(ax, REF_CLOSE, pad=0.03, tag=f'FIG4 {nm}')
    if _ylim is None:
        _ylim = ax.get_ylim()
    ax.set_ylim(*_ylim)                    # EXPLICIT and EQUAL across panels
    ax.set_xlim(REF_IDX[0], REF_IDX[-1])
    ax.set_ylabel('Nifty close', fontsize=9)

    # THE LAG, drawn: for every ZigZag swing starting inside the window, mark the
    # swing start, mark the first correctly-directed emitted bar, and span the
    # gap. That span IS L for that swing -- this is the picture the metric is a
    # summary of, so the two can be checked against each other by eye.
    _d = emitted_direction(lab.values)
    _shown = []
    for i0, i1, sgn in SWINGS:
        if not _m[i0]:
            continue
        c = '#0033cc' if sgn > 0 else '#cc0033'
        t0 = DATES[i0]
        ax.axvline(t0, color=c, ls='--', lw=1.3, zorder=6)
        _loc = np.flatnonzero(np.asarray(lab.index) == np.datetime64(t0))
        _off = int(_loc[0]) if _loc.size else 0
        _hit = np.flatnonzero(_d[_off:] == sgn)
        _y = _ylim[0] + 0.93 * (_ylim[1] - _ylim[0]) - 0.06 * (_ylim[1] - _ylim[0]) * len(_shown)
        if _hit.size:
            _lag = int(_hit[0])
            t1 = lab.index[_off + _lag]
            ax.annotate('', xy=(t1, _y), xytext=(t0, _y),
                        arrowprops=dict(arrowstyle='<->', color=c, lw=1.5))
            ax.plot([t1], [REF_CLOSE[_off + _lag]], marker='o', ms=9, mfc='none',
                    mec=c, mew=2.0, zorder=7)
            ax.annotate(f"swing {'UP' if sgn > 0 else 'DOWN'} — L = {_lag} bars",
                        ((t0 + (t1 - t0) / 2) if _lag else t0, _y), fontsize=9,
                        color=c, ha='center', va='bottom', fontweight='bold')
        else:
            ax.annotate(f"swing {'UP' if sgn > 0 else 'DOWN'} — NEVER matched",
                        (t0, _y), fontsize=9, color=c, ha='left', va='bottom',
                        fontweight='bold')
        _shown.append(i0)

    _first = np.flatnonzero(_d > 0)
    _ft = (f'first BULL bar in window: {lab.index[_first[0]]:%Y-%m-%d %H:%M}'
           if _first.size else 'NEVER labelled BULL in this window')
    ax.set_title(f"{nm}{'  [VOID — a guard failed]' if r['void'] else ''}   "
                 f"S={r['S']:.2f}%   L={r['L']:.2f} bars (whole series)   "
                 f"W={r['W']:.2f}   |   {_ft}",
                 fontsize=11, fontweight='bold', loc='left')

# The regime key goes ABOVE the panels, not inside them: at this zoom an in-axes
# legend covers the price line it is supposed to be explaining.
fig.legend(handles=[mpatches.Patch(color=REGIME_COLORS[r], alpha=0.7, label=r)
                    for r in REGIME_LABELS],
           loc='upper center', bbox_to_anchor=(0.5, 0.925), ncol=5, fontsize=9,
           frameon=False)
axes[-1].set_xlabel('date', fontsize=10)
assert axes[0].get_ylim() == axes[1].get_ylim(), 'panel y-limits differ'
assert axes[0].get_ylim()[0] > 0, 'y-axis is zero-anchored -- tight limits required'
fig.suptitle('FIG 4 — ' + REF_NOTE
             + f'   ({len(REF_IDX)} bars, low->close +{_ref_move:.2f}%)'
             + ('\n[SYNTHETIC - ILLUSTRATIVE ONLY — this is NOT the real May-2025 '
                'V-bottom]' if TAC_SYNTH else '')
             + '\nsame data, same fit seeds, same y-limits — only the labelling differs',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.905))
print(f'ASSERT OK: both panels share y-limits {axes[0].get_ylim()} '
      f'(tight, NOT zero-anchored)')
save_fig(fig, 'fig4_reference_case_may2025.png')
plt.show()
""")

# ===========================================================================
md(r"""
## 11. Verification block and config hand-off
""")

co(r"""
# ===========================================================================
# FINAL VERIFICATION -- the leakage tripwire, the decrees, the figure inventory.
# ===========================================================================
print('=' * 90)
print('VERIFICATION')
print('=' * 90)
print(f'  label_bars calls (all guarded)      : {LABEL_CALLS}')
print(f'  HMM fits                            : {FIT_COUNT}')
print(f'  ZigZag computations                 : {ZZ_CALLS}')
print(f'  arms measured                       : {len(RESULTS)}')
print(f'  seed sets per arm (R)               : {R_SEED_SETS}  {seed_sets(ENSEMBLE_K)}')
assert BAR_DIR_WEIGHT == 0.0
print(f'  BAR_DIR_WEIGHT                      : {BAR_DIR_WEIGHT}  (asserted on every call)')

# The leakage assertion, restated as an explicit end-of-run check.
try:
    label_bars(None)
except AssertionError as e:
    print(f'  ZigZag leakage tripwire (live test) : ARMED -> {str(e)[:60]}...')
else:
    raise AssertionError('the leakage tripwire did NOT fire after the eval phase')

print(f'  PNGs written                        : {len(SAVED_PNGS)}')
for f in SAVED_PNGS:
    print(f'      {f}')
print(f'  total runtime                       : {time.time() - T_START:.1f}s')
print('=' * 90)
if TAC_SYNTH:
    print('*** SYNTHETIC - ILLUSTRATIVE ONLY. No verdict above is decision-grade. ***')
    print('*** Re-run on Kaggle with real yfinance data.                          ***')
    print('=' * 90)
""")

co(r"""
# ===========================================================================
# CONFIG HAND-OFF BLOCK -- pasteable Python recording every parameter of the
# arms that matter, so a later decision can be folded into the master.
# ===========================================================================
def _handoff(r):
    a = _armof[r['name']]
    return (f"    {r['name']!r}: dict(\n"
            f"        # S={r['S']:.2f}%  L_med={r['L']:.2f}  L_p75={r['L75']:.2f}  "
            f"W={r['W']:.2f}  rows_per_param={r['rows_per_param']}\n"
            f"        # guards: " + ' '.join(f"{k}={'PASS' if v[0] else 'FAIL'}"
                                             for k, v in r['guards'].items())
            + f"   -> {'VOID' if r['void'] else 'ok'}\n"
            f"        N_STATES={a['N']}, COVARIANCE_TYPE={a['cov']!r},\n"
            f"        FEATURES={list(a['features'])!r},\n"
            f"        ENSEMBLE_K={a['K']}, INIT={a['init']!r},\n"
            f"        CONFIRM_BARS={a['lab'].get('confirm_bars', CONFIRM_BARS)},\n"
            f"        ESCALATION_DURING_HOLD="
            f"{a['lab'].get('escalation_during_hold', ESCALATION_DURING_HOLD)!r},\n"
            f"        CONFIRM_POLICY={a['pol']!r}, ENSEMBLE_POST={a['post']!r},\n"
            f"        DIRECTION_VOTE={a['mass']!r},\n"
            f"    ),")


_show = [BASE, BEST_LAG] + ([r for r in FRONTIER if r['name'] not in
                             ('BASELINE', BEST_LAG['name'])])
_seen, _uniq = set(), []
for r in _show:
    if r['name'] not in _seen:
        _seen.add(r['name']); _uniq.append(r)

print('# ' + '=' * 78)
print('# REGDET V1.1 -- STABILITY/LAG CONFIG HAND-OFF BLOCK')
print(f'# generated {pd.Timestamp.now():%Y-%m-%d %H:%M}   '
      f'data = {"SYNTHETIC (ILLUSTRATIVE ONLY)" if TAC_SYNTH else "REAL yfinance 2h"}')
print(f'# bars={N_BARS}  fit_window={N_FIT}  R={R_SEED_SETS} disjoint seed sets  '
      f'ZigZag={ZZ_PCT}%')
print('#')
print('# FIXED BY DECREE FOR ALL ARMS BELOW:')
print(f'BAR_DIR_WEIGHT = {W_FIXED}          # settled; not swept in this notebook')
print(f'INTENSITY_MODE = {INTENSITY_MODE!r}')
print(f'DIRECTION_MODE = {DIRECTION_MODE!r}')
print(f'H_TARGET_RATE, H_EXIT_SLACK = {H_TARGET_RATE}, {H_EXIT_SLACK}')
print(f'DIRECTION_EXCLUDE = {DIRECTION_EXCLUDE!r}')
print(f'BASE_SEED = {BASE_SEED}')
print()
print('# VERDICT: ' + ('dominating -> ' + ', '.join(r['name'] for r in DOMINATING)
                       if DOMINATING else 'NO DOMINATING CONFIGURATION'))
print(f"# HEADROOM: baseline seed spread  S {S_SPREAD:.2f} pp   L {L_SPREAD:.2f} bars"
      f"   W {W_SPREAD:.2f}")
print('STABILITY_LAG_ARMS = {')
for r in _uniq:
    print(_handoff(r))
print('}')
print('# ' + '=' * 78)
""")

md(r"""
## 12. What this notebook does and does not establish

**Does.** It measures S, L and W on one shared, cached set of fits with `R = 4`
disjoint seed sets; it proves no ZigZag output can reach a label; it proves the
degeneracy guard actually bites; and it reports every arm's three numbers together
with its guard verdict.

**Does not.** It does not compute a forward return, a Sharpe, or any economic
metric — deliberately, because ranking configs on those is exactly the
noise-dominated procedure that produced the "config lottery" (defect 4). It does
not tune a threshold to make anything pass. It does not combine S, L and W into a
score. And on a sandbox run it does not establish anything at all about the real
market: the synthetic fallback is a plumbing test.

**The decision it is meant to support** is a choice of point on a frontier, made
by a human looking at figure 4 and asking whether the lag actually improved enough
to be worth what it cost in stability and whipsaw.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'wrote {OUT} - {len(cells)} cells '
      f'({sum(1 for c in cells if c["cell_type"] == "code")} code)')
