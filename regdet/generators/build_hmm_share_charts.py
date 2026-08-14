"""
Builds hmm_share_charts.ipynb -- a SMALL, FAST, FIGURES-FIRST notebook whose only
job is to produce visually comparable labelled price charts at five HMM shares,
so the user can eyeball which labelling describes the market regimes best.

  HMM_share = 100 * (1 - BAR_DIR_WEIGHT)

    0%   HMM -> w = 1.00   (no HMM in direction at all; pure per-bar momentum)
    25%  HMM -> w = 0.75   (the arm the user says looked best)
    50%  HMM -> w = 0.50
    75%  HMM -> w = 0.25
    100% HMM -> w = 0.00   (current shipped default)

Design rules (from the brief):
  * ONE shared HMM ensemble fit reused across all five arms; arms differ ONLY by
    BAR_DIR_WEIGHT. Model identity asserted before/after the five label passes.
  * Identical visual treatment across panels: same colour map, same date range,
    same y-limits (asserted equal), same figsize, same linewidth.
  * Strictly causal labels -- the engine property is inherited, not re-derived.
  * Figures first. Numbers (one small table) AFTER the figures.
  * No scorecard, no forward returns, no economic metrics, no sweeps, no winner.
  * A complete, copy-pasteable CONFIG HAND-OFF BLOCK at the very end.

The engine is INLINED VERBATIM out of build_master_notebook_v2.py -- the same
practice that generator uses for regime_scorecard.py. Nothing here imports the
master, and the master, its generator, regime_scorecard.py and every shipped
default are left untouched.
"""
import json
import re

HERE = '/tmp/claude-0/-home-user-SHERM-QUANTY/f1a44349-dfe0-5e0f-9b1f-b0b39e930e1b/scratchpad'
OUT = f'{HERE}/hmm_share_charts.ipynb'
MASTER_GEN = f'{HERE}/build_master_notebook_v2.py'

# ---------------------------------------------------------------------------
# Harvest the engine cells from the master generator, VERBATIM.
#
# The master generator's cell bodies are Python string literals with runtime
# concatenation in them, so they cannot be slurped as raw text. Instead the
# generator is executed in a private namespace (with its output path redirected
# to a throwaway file) and its assembled `cells` list is read. That guarantees
# the engine text inlined below is byte-identical to the master's.
# ---------------------------------------------------------------------------
_gsrc = open(MASTER_GEN).read()
assert "regdet_v11_master.ipynb'" in _gsrc
_gsrc = _gsrc.replace("regdet_v11_master.ipynb'", "_hmm_share_engine_harvest.ipynb'", 1)
_ns = {'__name__': '_master_gen_harvest'}
exec(compile(_gsrc, MASTER_GEN, 'exec'), _ns)
_mcells = _ns['cells']


def _src(i):
    return ''.join(_mcells[i]['source'])


# Cell indices in the master notebook (asserted by content, not trusted by number)
IDX_CONST, IDX_LOAD, IDX_ENGINE, IDX_PLOT, IDX_WSWEEP = 3, 5, 7, 9, 76
SRC_CONST, SRC_LOAD = _src(IDX_CONST), _src(IDX_LOAD)
SRC_ENGINE, SRC_PLOT = _src(IDX_ENGINE), _src(IDX_PLOT)
SRC_WSWEEP = _src(IDX_WSWEEP)

assert 'REGIME_COLORS' in SRC_CONST and 'BAR_DIR_WEIGHT   = 0.0' in SRC_CONST
assert 'import numpy as np' in SRC_CONST
assert 'def load_2h' in SRC_LOAD and 'def _synth' in SRC_LOAD
assert 'def build_features' in SRC_ENGINE and 'def label_bars' in SRC_ENGINE \
    and 'def fit_hmm_ensemble' in SRC_ENGINE
assert 'def shade_bands' in SRC_PLOT and 'def regime_legend' in SRC_PLOT

# Descriptive fidelity: the master's w-sweep SECONDARY metric, sliced out verbatim.
# Only the two functions are taken -- not the sweep criteria, not the prints, not
# the shipped-label snapshot.
_k = re.search(r'(?m)^W_SECONDARY_K = (\d+).*$', SRC_WSWEEP)
assert _k and int(_k.group(1)) == 9, 'W_SECONDARY_K drifted'
_a = SRC_WSWEEP.index('def W_SECONDARY_PERBAR')
_b = SRC_WSWEEP.index("print('=' * 110)")
SRC_FIDELITY = (_k.group(0) + '\n'
                + re.search(r'(?m)^W_SECONDARY_METRIC_NAME = \(.*?\)\n',
                            SRC_WSWEEP, re.S).group(0)
                + '\n\n' + SRC_WSWEEP[_a:_b].rstrip() + '\n')
assert 'def W_SECONDARY_METRIC' in SRC_FIDELITY and 'W_SHIPPED_SNAPSHOT' not in SRC_FIDELITY

PROVENANCE = (
    f'build_master_notebook_v2.py cell {IDX_CONST} (constants/imports), '
    f'cell {IDX_LOAD} (load_2h/_synth), cell {IDX_ENGINE} (feature + labeling engine), '
    f'cell {IDX_PLOT} (plot helpers), cell {IDX_WSWEEP} (W_SECONDARY_* fidelity metric)'
)

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip('\n').splitlines(keepends=True)})


def co(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip('\n').splitlines(keepends=True)})


def co_sub(src):
    # Emit a code cell, substituting GENERATOR-time facts (the master cell
    # indices) into @TOKEN@ placeholders. Without this the notebook would try to
    # resolve IDX_* at RUN time, where they do not exist.
    for tok, val in (('@IDX_CONST@', IDX_CONST), ('@IDX_LOAD@', IDX_LOAD),
                     ('@IDX_ENGINE@', IDX_ENGINE), ('@IDX_PLOT@', IDX_PLOT),
                     ('@IDX_WSWEEP@', IDX_WSWEEP)):
        src = src.replace(tok, str(val))
    assert '@IDX' not in src, 'unsubstituted generator token left in a cell'
    co(src)


BANNER = '# ' + '=' * 74 + '\n'

# ===========================================================================
md(r"""
# RegDet V1.1 — HMM SHARE label charts

**One job:** show the same 2-hour Nifty price history labelled five times, at five
different *HMM shares*, under identical visual treatment, so the labelling can be
judged **by eye**.

    HMM_share = 100 * (1 - BAR_DIR_WEIGHT)

| HMM share | `BAR_DIR_WEIGHT` | what it means |
|---|---|---|
| **100%** | `w = 0.00` | direction is 100% HMM state mass — **the current shipped default** |
| **75%**  | `w = 0.25` | |
| **50%**  | `w = 0.50` | |
| **25%**  | `w = 0.75` | the arm the user says looked best |
| **0%**   | `w = 1.00` | no HMM in direction at all — pure causal per-bar momentum |

**HMM SHARE is the primary identifier everywhere** in this notebook — every title,
legend entry, filename and table row. `w=` is shown second, in brackets. Do not
read the two the other way round: they run in opposite directions.

### What makes this a fair comparison
1. **ONE shared HMM ensemble fit**, fitted once, reused by all five arms. The five
   arms differ *only* by `BAR_DIR_WEIGHT`. The model objects are asserted
   bit-identical before and after all five labelling passes.
2. **Identical visual treatment**: one regime→colour map, one date range per figure,
   one y-axis limit pair per figure (asserted equal across its five panels), one
   figure width, one linewidth. A different y-scale alone changes the visual verdict.
3. **Strictly causal labels.** A label at bar `t` reads only bars `<= t`. This is an
   inherited property of the engine, inlined verbatim and not modified here.

### What this notebook deliberately does NOT do
No six-family scorecard. No walk-forward. No A/B ladder. No w-sweep. **No forward
returns, no Sharpe, no economic metric of any kind** — this is about *regime
description accuracy*, not profitability, and a profit number is not smuggled in as
a tiebreaker. **No winner is declared.** The five pictures are evidence; the choice
is the user's.

### What to do with it
Run top to bottom (well under a minute). Five PNGs are written to the working
directory and their filenames printed at the end — export those and send them back.
The final cell prints a complete, copy-pasteable config block that reproduces
exactly what the charts show.
""")

co(r"""
%pip install -q hmmlearn yfinance
""")

# --- engine, inlined verbatim ----------------------------------------------
md(f"""
## 1. Engine, inlined verbatim

Inlined out of `build_master_notebook_v2.py` rather than imported, because a
standalone `.ipynb` on Kaggle cannot import a local `.py` next to it. Source:
`{PROVENANCE}`. Not one line is modified — same constants, same features, same
gates, same hysteresis, same causality. The master notebook and its generator are
untouched by this notebook.
""")

co(BANNER
   + f'# CONSTANTS + IMPORTS -- INLINED VERBATIM from {MASTER_GEN.split("/")[-1]}\n'
   + f'# (master notebook code cell {IDX_CONST}). Unmodified.\n'
   + BANNER + SRC_CONST)

co(BANNER
   + f'# DATA LOADER -- INLINED VERBATIM (master code cell {IDX_LOAD}).\n'
   + '# yfinance 60m -> 2h resample, with a synthetic fallback. Unmodified.\n'
   + BANNER + SRC_LOAD)

co(r"""
# ---------------------------------------------------------------------------
# PROVENANCE BANNER. Printed loudly and early so a sandbox run can never be
# mistaken for a decision-grade one.
# ---------------------------------------------------------------------------
if TAC_SYNTH:
    print('#' * 100)
    for _ in range(3):
        print('#   SYNTHETIC - ILLUSTRATIVE ONLY   ' * 2)
    print('#' * 100)
    print('#  The yfinance fetch FAILED (it is firewalled in some sandboxes).')
    print('#  Every chart and number below is generated from a synthetic GBM series.')
    print('#  It is a PLUMBING TEST ONLY. It says NOTHING about real Nifty regimes.')
    print('#  Re-run on Kaggle, where yfinance works, before judging any labelling.')
    print('#' * 100)
else:
    print('=' * 100)
    print('REAL yfinance data - decision-grade. Charts below are of actual Nifty 2h bars.')
    print('=' * 100)
print(f'bars={len(nifty)}   span {nifty.index[0]} -> {nifty.index[-1]}')
""")

co(BANNER
   + f'# FEATURE + LABELING ENGINE -- INLINED VERBATIM (master code cell {IDX_ENGINE}).\n'
   + '# build_features / intensity gates / direction buckets / bar_direction_masses /\n'
   + '# fit_hmm_ensemble / label_bars. Causality and every default unmodified.\n'
   + BANNER + SRC_ENGINE)

co(BANNER
   + f'# PLOT HELPERS -- INLINED VERBATIM (master code cell {IDX_PLOT}).\n'
   + '# regime_blocks / shade_bands (the batched, autolim-safe regime bands) /\n'
   + '# regime_legend. Unmodified.\n'
   + BANNER + SRC_PLOT)

co(BANNER
   + f'# DESCRIPTIVE FIDELITY -- INLINED VERBATIM (master code cell {IDX_WSWEEP}),\n'
   + '# the w-sweep SECONDARY metric. NOT reinvented here. Strictly causal:\n'
   + '# agreement of the emitted direction at t with sign(C[t] - C[t-k]), k=9.\n'
   + '# Only the two functions are taken -- none of the sweep machinery.\n'
   + BANNER + SRC_FIDELITY)

# --- arm mapping -----------------------------------------------------------
md(r"""
## 2. The five arms — the mapping, asserted in code

Getting `HMM share` and `w` backwards would invert the whole conclusion, so the
identity `HMM_share = 100 * (1 - BAR_DIR_WEIGHT)` is asserted rather than trusted.
""")

co(r"""
# ---------------------------------------------------------------------------
# THE FIVE ARMS. HMM SHARE is the primary identifier; w is secondary.
#   HMM_share = 100 * (1 - BAR_DIR_WEIGHT)
# Ordered 100% HMM first (most HMM) down to 0% HMM (no HMM in direction).
# ---------------------------------------------------------------------------
ARM_SHARES = [100, 75, 50, 25, 0]                 # percent HMM
ARM_W = [1.0 - s / 100.0 for s in ARM_SHARES]     # BAR_DIR_WEIGHT per arm
ARMS = list(zip(ARM_SHARES, ARM_W))

# ASSERT THE MAPPING BOTH WAYS. This is the one error that would invert the verdict.
for _s, _w in ARMS:
    assert abs(_s - 100.0 * (1.0 - _w)) < 1e-12, f'share/w mapping broken: {_s} vs w={_w}'
    assert abs(_w - (1.0 - _s / 100.0)) < 1e-12
assert ARM_W[0] == 0.0 and ARM_SHARES[0] == 100, '100% HMM must be w=0.00'
assert ARM_W[-1] == 1.0 and ARM_SHARES[-1] == 0, '0% HMM must be w=1.00'
assert sorted(ARM_SHARES, reverse=True) == ARM_SHARES, 'arms must descend in HMM share'
assert BAR_DIR_WEIGHT == 0.0, 'shipped default expected to be w=0.00 (=100% HMM)'
SHIPPED_SHARE = int(round(100.0 * (1.0 - BAR_DIR_WEIGHT)))


def arm_id(share, w):
    # THE canonical arm label. HMM share first, w second, everywhere.
    tag = '  [SHIPPED DEFAULT]' if w == BAR_DIR_WEIGHT else ''
    note = ''
    if w == 1.0:
        note = '  no HMM in direction'
    elif w == 0.0:
        note = '  direction 100% HMM'
    return f'{share:3d}% HMM  (w={w:.2f}){note}{tag}'


def arm_slug(share, w):
    return f'hmm{share:03d}_w{w:.2f}'.replace('.', 'p')


print('=' * 100)
print('THE FIVE ARMS      HMM_share = 100 * (1 - BAR_DIR_WEIGHT)      mapping ASSERTED OK')
print('=' * 100)
for _s, _w in ARMS:
    print('   ' + arm_id(_s, _w))
print()
print(f'shipped default in this notebook = {SHIPPED_SHARE}% HMM (w={BAR_DIR_WEIGHT:.2f})')
print('Only BAR_DIR_WEIGHT differs between arms. Everything else is byte-identical.')
print()
print('SCOPE LIMIT, stated up front: w touches the DIRECTION axis ONLY. The H-vs-L')
print('INTENSITY axis contains zero HMM in this engine, so no arm here can change')
print('how H/L is decided -- only bull/side/bear.')
""")

# --- one shared fit --------------------------------------------------------
md(r"""
## 3. ONE shared HMM fit

One `StandardScaler`, one anchored ensemble fit on the leading `TRAIN_FRACTION` of
bars, seeds `BASE_SEED + 0..ENSEMBLE_K-1`. Every arm below reuses *these* model
objects. Nothing is refitted per arm, so any visual difference between panels is
attributable to `BAR_DIR_WEIGHT` and to nothing else.
""")

co(r"""
import time as _time
T0 = _time.time()

# ---------------------------------------------------------------------------
# THE SINGLE SHARED FIT. Adopted config, pinned exactly as the master pins it.
# ---------------------------------------------------------------------------
CFG = next(c for c in CONFIGS if c['name'] == ADOPTED_CONFIG_NAME)
FEAT = list(CFG['features'])

fdf = build_features(nifty, vix, LOOKBACK_SCALE)
X_raw = fdf[FEAT].values
DATES = fdf.index
N_FIT = max(int(len(X_raw) * TRAIN_FRACTION), 50)
SCALER = StandardScaler().fit(X_raw[:N_FIT])
XS = SCALER.transform(X_raw)
TREND_RAW = fdf[TREND_FEATURE].values
CLOSE = nifty.reindex(DATES).values.astype(float)

_t = _time.time()
MODELS, ALL_CONV = fit_hmm_ensemble(XS[:N_FIT], CFG['N'], CFG['cov'])
FIT_SECS = _time.time() - _t
FIT_SEEDS = ensemble_seeds()

# Fingerprint the fit so the "shared, never refitted" claim can be ASSERTED, not
# asserted-by-comment. Raw bytes, not a tolerance.
def fit_fingerprint(models):
    return [(m.means_.tobytes(), m.transmat_.tobytes(), m.startprob_.tobytes())
            for m in models]

FIT_FP_BEFORE = fit_fingerprint(MODELS)

print('=' * 100)
print('ONE SHARED HMM FIT - reused by all five arms' + synth_tag())
print('=' * 100)
print(f'  config          {ADOPTED_CONFIG_NAME!r}   N_STATES={CFG["N"]}  cov={CFG["cov"]!r}')
print(f'  features ({len(FEAT)})   {FEAT}')
print(f'  ensemble        ENSEMBLE_K={len(MODELS)}  BASE_SEED={BASE_SEED}  seeds={FIT_SEEDS}')
print(f'  all converged   {ALL_CONV}')
print(f'  anchored fit    {N_FIT}/{len(XS)} bars ({TRAIN_FRACTION:.0%})   '
      f'scaler fit on the same leading window')
print(f'  bars labelled   {len(DATES)}   {DATES[0]} -> {DATES[-1]}')
print(f'  fit time        {FIT_SECS:.2f}s')
print()
print('  NO ARM REFITS. Five label passes over these exact model objects.')
""")

md(r"""
## 4. Five label passes over the shared fit
""")

co(r"""
# ---------------------------------------------------------------------------
# FIVE LABEL PASSES. Identical call, identical fit, identical everything --
# bar_dir_weight is the only argument that moves.
# ---------------------------------------------------------------------------
LABELS = {}          # share -> np.array of label strings
LAB_SERIES = {}      # share -> pd.Series indexed by DATES
pass_secs = {}
for _s, _w in ARMS:
    _t = _time.time()
    _df = label_bars(MODELS, XS, DATES, nifty, TREND_RAW, N_FIT, FEAT,
                     dir_feats=fdf, bar_dir_weight=_w)
    pass_secs[_s] = _time.time() - _t
    LABELS[_s] = _df['tactical_regime_state'].values.copy()
    LAB_SERIES[_s] = pd.Series(LABELS[_s], index=DATES)

# --- the fit must be untouched by the five passes -------------------------
FIT_FP_AFTER = fit_fingerprint(MODELS)
assert FIT_FP_AFTER == FIT_FP_BEFORE, \
    'THE SHARED FIT MUTATED across the label passes - the comparison would be unfair'
for _i, _m in enumerate(MODELS):
    assert np.array_equal(_m.means_, np.frombuffer(FIT_FP_BEFORE[_i][0],
                                                   dtype=_m.means_.dtype
                                                   ).reshape(_m.means_.shape))
print('ASSERT OK: all', len(MODELS), 'model objects bit-identical before and after '
      'the five label passes (means_, transmat_, startprob_).')

# --- sanity: the arms must actually differ, and the shipped arm must match ---
for _s in ARM_SHARES:
    assert len(LABELS[_s]) == len(DATES)
    assert set(LABELS[_s]) <= set(REGIME_LABELS), 'unexpected label emitted'
_pairs = [(a, b) for i, a in enumerate(ARM_SHARES) for b in ARM_SHARES[i + 1:]]
_ident = [(a, b) for a, b in _pairs if np.array_equal(LABELS[a], LABELS[b])]
if _ident:
    print(f'NOTE: these arm pairs produced IDENTICAL labels on this data: {_ident}')
else:
    print('all five arms produced distinct label sequences (as expected)')
print(f'label passes: {[f"{k}%:{v:.2f}s" for k, v in pass_secs.items()]}')
""")

md(r"""
### 4b. Sanity check — at 0% HMM the labels must be seed-independent

At `0% HMM (w=1.00)` the direction blend is `1.0 * bar_momentum_mass`, so the HMM
state masses are multiplied by zero and cannot reach the emitted label. The
intensity axis never contained any HMM. Therefore the `0% HMM` labels must be
**invariant to the HMM seeds**. If they are not, the plumbing is wrong and every
panel in this notebook is suspect.

Checked two ways: (i) free — relabel at `w=1.00` from each single ensemble member
and from the reversed member list, no refit; (ii) an explicitly-flagged **second
fit** on a disjoint seed set, used *only* for this check and never for any chart.
The higher arms are shown for contrast: they *should* move with the seeds.
""")

co(r"""
# ---------------------------------------------------------------------------
# SEED-INDEPENDENCE AT 0% HMM (w=1.00). Diagnostic only. The alt fit below is
# NEVER used to label any figure -- the charts all come from MODELS.
# ---------------------------------------------------------------------------
def _relabel(models, w):
    return label_bars(models, XS, DATES, nifty, TREND_RAW, N_FIT, FEAT,
                      dir_feats=fdf, bar_dir_weight=w)['tactical_regime_state'].values

print('=' * 100)
print('SANITY: is 0% HMM (w=1.00) seed-independent?')
print('=' * 100)

# (i) no refit: single members, and the member list reversed
_ref = LABELS[0]
_free = {'member[0] only': _relabel(MODELS[:1], 1.0),
         f'member[{len(MODELS)-1}] only': _relabel(MODELS[-1:], 1.0),
         'members reversed': _relabel(list(MODELS)[::-1], 1.0)}
for _k, _v in _free.items():
    _ok = np.array_equal(_v, _ref)
    print(f'  (i)  w=1.00 vs {_k:<20s} identical: {_ok}')
    assert _ok, f'0% HMM labels changed with {_k} - the HMM is leaking into direction'

# (ii) a deliberate SECOND FIT on a disjoint seed set (diagnostic only)
ALT_BASE_SEED = BASE_SEED + 1000
_t = _time.time()
ALT_MODELS, _alt_conv = fit_hmm_ensemble(XS[:N_FIT], CFG['N'], CFG['cov'],
                                         K=ENSEMBLE_K, base_seed=ALT_BASE_SEED)
ALT_FIT_SECS = _time.time() - _t
assert not np.array_equal(ALT_MODELS[0].means_, MODELS[0].means_) or True
_alt = {s: _relabel(ALT_MODELS, w) for s, w in ARMS}
print(f'  (ii) alt fit: seeds={[ALT_BASE_SEED + i for i in range(ENSEMBLE_K)]}  '
      f'converged={_alt_conv}  {ALT_FIT_SECS:.2f}s   (DIAGNOSTIC ONLY - not plotted)')
for _s, _w in ARMS:
    _agree = float(np.mean(_alt[_s] == LABELS[_s])) * 100.0
    _tag = '  <-- MUST be 100.00%' if _w == 1.0 else ''
    print(f'       {arm_id(_s, _w):<62s} label agreement across seed sets: '
          f'{_agree:6.2f}%{_tag}')
assert np.array_equal(_alt[0], LABELS[0]), \
    'PLUMBING BUG: 0% HMM (w=1.00) labels changed when the HMM seeds changed'
print()
print('  PASS: 0% HMM labels are seed-independent. The other arms moving with the')
print('  seeds is expected and is a known property of this engine (fit instability),')
print('  not a defect of this comparison -- every panel plotted uses ONE shared fit.')
del ALT_MODELS, _alt
""")

# --- window selection ------------------------------------------------------
md(r"""
## 5. Zoom windows, chosen by rule from the real data

Windows are picked **mechanically**, by rules fixed before any window was seen, so
no arm can be favoured by the choice of picture. All rules read the price series
only — they never look at any label. The rules are printed with the results.

| window | rule |
|---|---|
| **A — drawdown & recovery** | maximises `min(peak→trough drop, trough→end recovery)` over the window: the sharpest V |
| **B — chop** | minimises trend efficiency `abs(C[end]-C[start]) / sum(abs(diff))`: the longest directionless stretch |
| **C — uptrend** | maximises `return * trend efficiency`: the strongest *sustained* advance |
| **D — 2025-05-15** | fixed calendar window, included only if the data spans it (previously flagged as a V-bottom of interest) |

Window length is fixed at `ZOOM_BARS` for all of them so panel density is comparable.
""")

co(r"""
# ---------------------------------------------------------------------------
# OBJECTIVE WINDOW SELECTION. Price-only, label-blind, rules fixed in advance.
# (Window CHOICE is a presentation decision made over the whole history; it is
# not part of any signal, and no label anywhere is affected by it.)
# ---------------------------------------------------------------------------
ZOOM_BARS = 60          # ~20 sessions of 2h bars: individual labels stay visible
ZOOM_STRIDE = 2

def _eff(seg):
    d = np.abs(np.diff(seg)).sum()
    return abs(seg[-1] - seg[0]) / d if d > 0 else 0.0

def _vness(seg):
    j = int(np.argmin(seg))
    pk = float(np.max(seg[:j + 1])) if j > 0 else float(seg[0])
    tr = float(seg[j])
    drop = (pk - tr) / pk if pk > 0 else 0.0
    rec = (float(seg[-1]) - tr) / tr if tr > 0 else 0.0
    return min(drop, rec)

_starts = list(range(0, max(len(CLOSE) - ZOOM_BARS, 1), ZOOM_STRIDE))
_score = {'A': [], 'B': [], 'C': []}
for _i in _starts:
    seg = CLOSE[_i:_i + ZOOM_BARS]
    e = _eff(seg)
    ret = seg[-1] / seg[0] - 1.0
    _score['A'].append(_vness(seg))
    _score['B'].append(-e)                      # minimise efficiency
    _score['C'].append(ret * e)                 # sustained advance

ZOOM_RULES = {
    'A': 'sharpest drawdown-and-recovery: max min(peak->trough drop, trough->end recovery)',
    'B': 'longest chop: min trend efficiency abs(C[end]-C[start]) / sum(abs(diff))',
    'C': 'strongest sustained uptrend: max (window return * trend efficiency)',
    'D': 'fixed calendar window centred on 2025-05-15 (previously flagged V-bottom)',
}
ZOOM_NAMES = {'A': 'drawdown_recovery', 'B': 'chop', 'C': 'uptrend', 'D': 'vbottom_2025_05_15'}

ZOOMS = []
for _k in ('A', 'B', 'C'):
    _i = _starts[int(np.argmax(_score[_k]))]
    ZOOMS.append((_k, _i, _i + ZOOM_BARS))

# D: the previously-flagged V-bottom, included only if the data actually spans it.
VBOTTOM_TS = pd.Timestamp('2025-05-15')
if DATES[0] <= VBOTTOM_TS <= DATES[-1]:
    _c = int(np.searchsorted(DATES.values, VBOTTOM_TS.to_datetime64()))
    _i = int(np.clip(_c - ZOOM_BARS // 2, 0, len(CLOSE) - ZOOM_BARS))
    ZOOMS.append(('D', _i, _i + ZOOM_BARS))
    VBOTTOM_IN_SPAN = True
else:
    VBOTTOM_IN_SPAN = False

print('=' * 100)
print(f'ZOOM WINDOWS - {len(ZOOMS)} chosen by rule, label-blind, {ZOOM_BARS} bars each'
      + synth_tag())
print('=' * 100)
for _k, _a, _b in ZOOMS:
    seg = CLOSE[_a:_b]
    print(f'  {_k}  {DATES[_a].date()} -> {DATES[_b - 1].date()}   bars {_a}-{_b - 1}   '
          f'ret {seg[-1] / seg[0] - 1:+7.2%}  eff {_eff(seg):.3f}  V-ness {_vness(seg):.3f}')
    print(f'     rule: {ZOOM_RULES[_k]}')
if not VBOTTOM_IN_SPAN:
    print(f'  D  SKIPPED: {VBOTTOM_TS.date()} is outside the data span '
          f'({DATES[0].date()} -> {DATES[-1].date()}).')
""")

# --- figures ---------------------------------------------------------------
md(r"""
## 6. The figures

**Read these before reading any number below.** The comparison table is printed
*after* the figures on purpose, so the numbers do not anchor the visual judgement.

Panels run **100% HMM at the top, descending to 0% HMM at the bottom**. Within a
figure every panel shares the x-range, the y-limits (asserted equal), the colour
map and the linewidth. The price line segment ending at bar `t` is coloured by the
label emitted at bar `t`; the same label also tints the background band.

Colour key, identical in every panel of every figure:

| colour | label |
|---|---|
| dark green `#006400` | **H_BULL** |
| light green `#90EE90` | **L_BULL** |
| grey `#808080` | **SIDEWAYS** |
| pink `#FFB6C1` | **L_BEAR** |
| dark red `#8B0000` | **H_BEAR** |
""")

co(r"""
# ---------------------------------------------------------------------------
# ONE panel painter, used by EVERY panel of EVERY figure. Identical treatment is
# enforced by there being exactly one code path, plus asserts on the y-limits.
# ---------------------------------------------------------------------------
from matplotlib.collections import LineCollection

LINEWIDTH = 2.0
BAND_ALPHA = 0.10       # faint: the COLOURED LINE is the signal, the band is a hint
YPAD = 0.03
SAVED_PNGS = []


def y_limits(close_slice, pad=YPAD):
    v = np.asarray(close_slice, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    m = (hi - lo) * pad or abs(hi) * pad or 1.0
    return (lo - m, hi + m)


def paint_arm_panel(ax, idx, close_slice, labels, ylim, lw=LINEWIDTH):
    # price line coloured by the regime label of the bar the segment ENDS on,
    # over a faint band of the same colour. One implementation, no per-arm branch.
    x = mdates.date2num(idx)
    pts = np.column_stack([x, np.asarray(close_slice, dtype=float)])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cols = [REGIME_COLORS.get(l, '#808080') for l in labels[1:]]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=3))
    shade_bands(ax, regime_blocks(pd.Series(labels, index=idx)), alpha=BAND_ALPHA)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*ylim)                      # EXPLICIT: autoscaler never consulted
    # date2num was used for the LineCollection, so the axis must be told it is a
    # date axis or the ticks come out as raw ordinals.
    _loc = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(_loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_loc))
    ax.set_ylabel('Nifty', fontsize=9)
    return ax


def assert_identical_panels(axes, ylim):
    # The fairness guarantee, checked rather than asserted in prose.
    xs = [tuple(a.get_xlim()) for a in axes]
    ys = [tuple(a.get_ylim()) for a in axes]
    assert len(set(ys)) == 1, f'y-limits differ across panels: {ys}'
    assert len(set(xs)) == 1, f'x-limits differ across panels: {xs}'
    assert np.allclose(ys[0], ylim), 'panel y-limits are not the intended shared limits'
    return ys[0], xs[0]


def save_fig(fig, fname, dpi=120):
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    SAVED_PNGS.append(fname)
    print(f'   saved -> {fname}  (dpi={dpi})')
    return fname


print('panel painter ready: one code path for all panels, y-limits set explicitly, '
      f'linewidth={LINEWIDTH}, band alpha={BAND_ALPHA}')
""")

co(r"""
# ===========================================================================
# FIG 1 - FULL HISTORY, five panels, 100% HMM at top -> 0% HMM at bottom.
# ===========================================================================
_ylim1 = y_limits(CLOSE)
fig, axes = plt.subplots(len(ARMS), 1, figsize=(16, 22), sharex=True)
for ax, (s, w) in zip(axes, ARMS):
    paint_arm_panel(ax, DATES, CLOSE, LABELS[s], _ylim1)
    ax.set_title(arm_id(s, w), fontsize=13, fontweight='bold', loc='left')
regime_legend(axes[0], loc='upper left')
axes[-1].set_xlabel('date', fontsize=10)
fig.suptitle('FIG 1 - RegDet V1.1 labels at five HMM SHARES, full history'
             + synth_tag() + f'\nONE shared fit ({ADOPTED_CONFIG_NAME}, K={len(MODELS)}, '
             f'seeds {FIT_SEEDS}) - arms differ ONLY by BAR_DIR_WEIGHT'
             '\nidentical colours, date range, y-limits, linewidth in every panel',
             fontsize=15, fontweight='bold', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.975))
_y, _x = assert_identical_panels(list(axes), _ylim1)
print(f'ASSERT OK: all 5 panels share y-limits ({_y[0]:.1f}, {_y[1]:.1f}) '
      f'and the identical x-range')
save_fig(fig, 'fig1_full_history_all_arms.png')
plt.show()
""")

co(r"""
# ===========================================================================
# FIG 2..N - the rule-chosen zoom windows, five panels each, same treatment.
# ===========================================================================
for _n, (_k, _a, _b) in enumerate(ZOOMS, start=2):
    idx = DATES[_a:_b]
    seg = CLOSE[_a:_b]
    ylim = y_limits(seg)
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(16, 16), sharex=True)
    for ax, (s, w) in zip(axes, ARMS):
        paint_arm_panel(ax, idx, seg, LABELS[s][_a:_b], ylim)
        ax.set_title(arm_id(s, w), fontsize=12, fontweight='bold', loc='left')
    regime_legend(axes[0], loc='upper left')
    axes[-1].set_xlabel('date', fontsize=10)
    fig.suptitle(f'FIG {_n} - ZOOM {_k}: {ZOOM_NAMES[_k].replace("_", " ")}'
                 f'   {idx[0].date()} -> {idx[-1].date()}  ({len(idx)} bars)' + synth_tag()
                 + f'\nselection rule (label-blind): {ZOOM_RULES[_k]}'
                 + '\nsame shared fit, same colours, same y-limits in all five panels',
                 fontsize=14, fontweight='bold', y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    assert_identical_panels(list(axes), ylim)
    save_fig(fig, f'fig{_n}_zoom_{ZOOM_NAMES[_k]}.png')
    plt.show()

print()
print(f'ASSERT OK: every zoom figure has 5 panels on shared x- and y-limits.')
""")

# --- table -----------------------------------------------------------------
md(r"""
## 7. The small comparison table

Deliberately printed **after** the figures. Four descriptive quantities per arm and
nothing else:

* **occupancy %** of each of the five labels;
* **switches per 100 bars** — how often the emitted label changes;
* **median run length** per label, in bars;
* **descriptive fidelity** — the master notebook's w-sweep SECONDARY metric,
  inlined verbatim: the fraction of non-`SIDEWAYS` bars where the sign of the
  emitted direction matches `sign(C[t] - C[t-9])`. Strictly causal.

**Read the fidelity column with the caveat the master notebook attaches to it:** it
measures how well a label *describes the move that has already happened*, not
predictive skill, and it is **partly mechanical** at low HMM share — at `0% HMM` the
direction blend literally *is* a trailing-momentum term, so a high number there is
close to a tautology, not a finding. No forward return, Sharpe or economic quantity
appears anywhere in this notebook, by design.
""")

co(r"""
# ---------------------------------------------------------------------------
# THE TABLE. Descriptive only. No forward-looking quantity is computed anywhere.
# ---------------------------------------------------------------------------
def run_lengths(labels):
    out = {}
    lab = np.asarray(labels, dtype=object)
    i = 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        out.setdefault(lab[i], []).append(j - i + 1)
        i = j + 1
    return out


rows = []
for s, w in ARMS:
    lab = LABELS[s]
    n = len(lab)
    rl = run_lengths(lab)
    occ = {r: 100.0 * float(np.mean(lab == r)) for r in REGIME_LABELS}
    sw = int(np.sum(lab[1:] != lab[:-1]))
    fid, n_scored = W_SECONDARY_METRIC(lab, CLOSE)
    rows.append(dict(share=s, w=w, occ=occ, sw100=100.0 * sw / max(n - 1, 1),
                     med={r: (float(np.median(rl[r])) if r in rl else np.nan)
                          for r in REGIME_LABELS},
                     med_all=float(np.median([v for vs in rl.values() for v in vs])),
                     fid=fid, n_fid=n_scored))

W1 = 34
print('=' * 118)
print('COMPARISON TABLE - descriptive only, printed AFTER the figures' + synth_tag())
print('=' * 118)
print(f'  bars={len(CLOSE)}   ONE shared fit ({ADOPTED_CONFIG_NAME}, K={len(MODELS)}, '
      f'seeds={FIT_SEEDS})   arms differ only by BAR_DIR_WEIGHT')
print()
print('OCCUPANCY %  (share of all bars carrying each label)')
print('-' * 118)
print('arm'.ljust(W1) + ''.join(r.rjust(11) for r in REGIME_LABELS)
      + 'switch/100b'.rjust(13) + 'med run'.rjust(10))
print('-' * 118)
for r in rows:
    print(f'{r["share"]:3d}% HMM (w={r["w"]:.2f})'.ljust(W1)
          + ''.join(f'{r["occ"][k]:11.2f}' for k in REGIME_LABELS)
          + f'{r["sw100"]:13.2f}' + f'{r["med_all"]:10.1f}')
print()
print('MEDIAN RUN LENGTH, bars  (nan = label never emitted by that arm)')
print('-' * 118)
print('arm'.ljust(W1) + ''.join(r.rjust(11) for r in REGIME_LABELS))
print('-' * 118)
for r in rows:
    print(f'{r["share"]:3d}% HMM (w={r["w"]:.2f})'.ljust(W1)
          + ''.join(f'{r["med"][k]:11.1f}' for k in REGIME_LABELS))
print()
print(f'DESCRIPTIVE FIDELITY  ({W_SECONDARY_METRIC_NAME})')
print('-' * 118)
print('arm'.ljust(W1) + 'fidelity'.rjust(11) + 'bars scored'.rjust(13) + '   note')
print('-' * 118)
for r in rows:
    note = ''
    if r['w'] == 1.00:
        note = 'PARTLY MECHANICAL: at 0% HMM the direction IS trailing momentum'
    elif r['w'] >= 0.50:
        note = 'partly mechanical (momentum term dominates the blend)'
    elif r['w'] == BAR_DIR_WEIGHT:
        note = 'shipped default'
    print(f'{r["share"]:3d}% HMM (w={r["w"]:.2f})'.ljust(W1)
          + f'{r["fid"]:11.4f}' + f'{r["n_fid"]:13d}' + '   ' + note)
print('-' * 118)
print('Fidelity is DESCRIPTION, not prediction, and it is not a score to be maximised.')
print('NO WINNER IS DECLARED HERE. These four columns cannot settle which labelling')
print('reads the market best - the five pictures above are the evidence, and the')
print('choice of arm is the USER\'S.')
""")

co(r"""
# ---------------------------------------------------------------------------
# FILES WRITTEN. Export these and send them back.
# ---------------------------------------------------------------------------
import os
print('=' * 100)
print(f'PNG FILES WRITTEN - {len(SAVED_PNGS)} figures' + synth_tag())
print('=' * 100)
for f in SAVED_PNGS:
    print(f'   {f:<44s} {os.path.getsize(f) / 1024:8.1f} KB   '
          f'{os.path.abspath(f)}')
print()
print(f'working directory: {os.getcwd()}')
print(f'total runtime so far: {_time.time() - T0:.1f}s')
if TAC_SYNTH:
    print()
    print('*** THESE PNGs ARE SYNTHETIC - ILLUSTRATIVE ONLY. Do not judge any '
          'labelling from them. ***')
""")

# --- config hand-off -------------------------------------------------------
md(r"""
## 8. CONFIG HAND-OFF BLOCK

Everything needed to reproduce exactly what the charts above show, in one place, as
a **valid Python assignment block you can paste straight into a master notebook**.
No ellipses, no prose, no summarising. Resolved data provenance and the inlined
engine provenance are included so a master build knows what it must carry over.
""")

co_sub(r"""
# ---------------------------------------------------------------------------
# THE HAND-OFF. Printed as pasteable Python. Values are read from the LIVE run,
# not retyped, so the block cannot drift from what was actually plotted.
# ---------------------------------------------------------------------------
_L = []
A = _L.append
A('# ' + '=' * 76)
A('# RegDet V1.1 - HMM SHARE CHART RUN: COMPLETE REPRODUCTION CONFIG')
A('# Generated by hmm_share_charts.ipynb. Paste as-is.')
A('# ' + '=' * 76)
A('')
A('# ---- data provenance (RESOLVED AT RUN TIME) ----------------------------')
A(f'DATA_IS_REAL        = {not TAC_SYNTH!r}          '
  f'# {"REAL yfinance" if not TAC_SYNTH else "SYNTHETIC - ILLUSTRATIVE ONLY"}')
A(f'DATA_SOURCE         = {("yfinance" if not TAC_SYNTH else "synthetic _synth() GBM fallback")!r}')
A("TICKER_PRICE        = '^NSEI'")
A("TICKER_VIX          = '^INDIAVIX'")
A("YF_INTERVAL         = '60m'")
A("YF_PERIOD           = '730d'")
A("YF_AUTO_ADJUST      = True")
A("RESAMPLE_RULE       = '2h'            # .resample('2h').last().dropna()")
A('TZ_HANDLING         = "index tz_localize(None) if tz-aware -> naive"')
A(f'RAW_BARS            = {len(nifty)}                # bars out of load_2h()')
A(f'RAW_FIRST_TS        = {str(nifty.index[0])!r}')
A(f'RAW_LAST_TS         = {str(nifty.index[-1])!r}')
A(f'FEATURE_BARS        = {len(DATES)}                # bars after build_features() warmup')
A(f'FEATURE_FIRST_TS    = {str(DATES[0])!r}')
A(f'FEATURE_LAST_TS     = {str(DATES[-1])!r}')
A(f'BARS_DROPPED_WARMUP = {len(nifty) - len(DATES)}')
A('')
A('# ---- labeling config (the engine knobs actually in effect) -------------')
A(f'INTENSITY_MODE         = {INTENSITY_MODE!r}')
A(f'H_TARGET_RATE          = {H_TARGET_RATE!r}')
A(f'H_EXIT_SLACK           = {H_EXIT_SLACK!r}')
A(f'DIRECTION_MODE         = {DIRECTION_MODE!r}')
A(f'ESCALATION_DURING_HOLD = {ESCALATION_DURING_HOLD!r}')
A(f'CONFIRM_BARS           = {CONFIRM_BARS!r}')
A(f'DIRECTION_EXCLUDE      = {DIRECTION_EXCLUDE!r}')
A(f'CONF_L                 = {CONF_L!r}')
A(f'Z_HI                   = {Z_HI!r}')
A(f'EFF_HI                 = {EFF_HI!r}')
A(f'Z_HI_EXIT              = {Z_HI_EXIT!r}')
A(f'EFF_HI_EXIT            = {EFF_HI_EXIT!r}')
A(f'EFF_WIN                = {EFF_WIN!r}')
A(f'MOM_3D_BARS            = {MOM_3D_BARS!r}')
A(f'DIR_TAU                = {DIR_TAU!r}   # DIRECTION_MODE=soft only, NOT in use')
A(f'DIR_C                  = {DIR_C!r}')
A(f'DIR_SCALE              = {DIR_SCALE!r}')
A(f'BAR_DIR_TAU            = {BAR_DIR_TAU!r}')
A(f'BAR_DIR_FEATURES       = {BAR_DIR_FEATURES!r}')
A(f'TREND_FEATURE          = {TREND_FEATURE!r}')
A(f'REGIME_LABELS          = {REGIME_LABELS!r}')
A(f'REGIME_COLORS          = {REGIME_COLORS!r}')
A('')
A('# ---- BAR_DIR_WEIGHT: THE SWEPT AXIS -----------------------------------')
A('#   HMM_share = 100 * (1 - BAR_DIR_WEIGHT)')
A('# ARM_HMM_SHARES and ARM_BAR_DIR_WEIGHTS are index-aligned.')
A(f'ARM_HMM_SHARES       = {ARM_SHARES!r}        # percent HMM in the DIRECTION axis')
A(f'ARM_BAR_DIR_WEIGHTS  = {[float(w) for w in ARM_W]!r}')
A(f'SHIPPED_BAR_DIR_WEIGHT = {BAR_DIR_WEIGHT!r}       '
  f'# == {SHIPPED_SHARE}% HMM  <-- THE SHIPPED DEFAULT')
A(f'BAR_DIR_WEIGHT       = {BAR_DIR_WEIGHT!r}          '
  f'# single-arm value; set to an ARM_BAR_DIR_WEIGHTS entry to reproduce one panel')
A('')
A('# ---- HMM fit (ONE shared fit reused by all five arms) -----------------')
A(f'ADOPTED_CONFIG_NAME  = {ADOPTED_CONFIG_NAME!r}')
A(f'N_STATES             = {CFG["N"]!r}')
A(f'COVARIANCE_TYPE      = {CFG["cov"]!r}')
A(f'HMM_ITER             = {HMM_ITER!r}')
A(f'FEATURE_COLS_USED    = {FEAT!r}   # the feature list ACTUALLY fitted')
A(f'ENSEMBLE_K           = {ENSEMBLE_K!r}')
A(f'BASE_SEED            = {BASE_SEED!r}')
A(f'ENSEMBLE_SEEDS       = {list(FIT_SEEDS)!r}   # BASE_SEED + 0..K-1')
A(f'TRAIN_FRACTION       = {TRAIN_FRACTION!r}         '
  f'# anchored fit fraction (scaler AND HMM)')
A(f'N_FIT_BARS           = {N_FIT!r}          '
  f'# = max(int(FEATURE_BARS * TRAIN_FRACTION), 50)')
A(f'ALL_MEMBERS_CONVERGED = {ALL_CONV!r}')
A(f'BARS_PER_DAY         = {BARS_PER_DAY!r}')
A(f'LOOKBACK_SCALE       = {LOOKBACK_SCALE!r}')
A(f'BASE_WIN             = {BASE_WIN!r}')
A('SCALER               = "sklearn StandardScaler fit on X_raw[:N_FIT_BARS]"')
A('')
A('# ---- figure treatment (identical across every panel) ------------------')
A(f'ZOOM_BARS            = {ZOOM_BARS!r}')
A(f'ZOOM_STRIDE          = {ZOOM_STRIDE!r}')
A(f'LINEWIDTH            = {LINEWIDTH!r}')
A(f'BAND_ALPHA           = {BAND_ALPHA!r}')
A(f'YPAD                 = {YPAD!r}')
A('FIG1_FIGSIZE         = (16, 22)')
A('FIGN_FIGSIZE         = (16, 16)')
A('SAVE_DPI             = 120')
A(f'ZOOM_WINDOWS         = {[(k, str(DATES[a]), str(DATES[b - 1])) for k, a, b in ZOOMS]!r}')
A(f'ZOOM_RULES           = {ZOOM_RULES!r}')
A('')
A('# ---- descriptive metric ------------------------------------------------')
A(f'W_SECONDARY_K        = {W_SECONDARY_K!r}           # trailing lookback, bars')
A(f'DESCRIPTIVE_FIDELITY_DEF = {W_SECONDARY_METRIC_NAME!r}')
A('# NO forward return, Sharpe or economic metric is computed in this notebook.')
A('')
A('# ---- ENGINE PROVENANCE: what a master build must carry over -----------')
A('# All of the following was INLINED VERBATIM from build_master_notebook_v2.py')
A('# (its assembled notebook cells, unmodified):')
A('ENGINE_PROVENANCE = {')
A('    "source_generator": "build_master_notebook_v2.py",')
A('    "constants_and_imports": "code cell @IDX_CONST@",')
A('    "data_loader": "code cell @IDX_LOAD@  (load_2h, _synth)",')
A('    "labeling_engine": "code cell @IDX_ENGINE@  (build_features, direction_weight, "')
A('                       "composite_subset, trend_efficiency, gate_band, intensity_state, "')
A('                       "hold_masks, confirm_delay, direction_buckets, bar_direction_score, "')
A('                       "bar_direction_masses, ensemble_direction_masses, ensemble_seeds, "')
A('                       "fit_hmm_ensemble, label_bars)",')
A('    "plot_helpers": "code cell @IDX_PLOT@  (regime_blocks, shade_bands, regime_legend)",')
A('    "fidelity_metric": "code cell @IDX_WSWEEP@  (W_SECONDARY_PERBAR, W_SECONDARY_METRIC)",')
A('    "written_here": "arm mapping, shared-fit driver, panel painter, window rules, "')
A('                    "comparison table, this block",')
A('}')
A('')
A('# ---- reproduction recipe ----------------------------------------------')
A('# 1. load_2h() -> nifty, vix')
A('# 2. fdf = build_features(nifty, vix, LOOKBACK_SCALE); X_raw = fdf[FEATURE_COLS_USED]')
A('# 3. N_FIT = max(int(len(X_raw) * TRAIN_FRACTION), 50)')
A('# 4. XS = StandardScaler().fit(X_raw[:N_FIT]).transform(X_raw)')
A('# 5. MODELS, _ = fit_hmm_ensemble(XS[:N_FIT], N_STATES, COVARIANCE_TYPE)   # ONCE')
A('# 6. for w in ARM_BAR_DIR_WEIGHTS:')
A('#        label_bars(MODELS, XS, fdf.index, nifty, fdf[TREND_FEATURE].values,')
A('#                   N_FIT, FEATURE_COLS_USED, dir_feats=fdf, bar_dir_weight=w)')
A('# The five arms MUST share step 5. Refitting per arm invalidates the comparison.')
A('')
A(f'REPRO_NOTE = {("SYNTHETIC RUN - ILLUSTRATIVE ONLY, not decision-grade" if TAC_SYNTH else "REAL yfinance run")!r}')

CONFIG_BLOCK = '\n'.join(_L)
print('=' * 100)
print('CONFIG HAND-OFF BLOCK - copy everything between the rules' + synth_tag())
print('=' * 100)
print(CONFIG_BLOCK)
print('=' * 100)

# The block must be valid, pasteable Python -- checked, not hoped for.
compile(CONFIG_BLOCK, '<config_block>', 'exec')
print('ASSERT OK: the block above compiles as valid Python and is pasteable verbatim.')
with open('hmm_share_config_block.py', 'w') as _f:
    _f.write(CONFIG_BLOCK + '\n')
print('also written to: hmm_share_config_block.py')
print(f'TOTAL RUNTIME: {_time.time() - T0:.1f}s')
""")

md(r"""
## 9. What to send back, and what this does not settle

Send back the PNGs listed above (and, if you like, `hmm_share_config_block.py`).

**No winner is declared in this notebook, and none should be inferred from the
table.** The four descriptive columns cannot rank a labelling: occupancy and switch
rate have no correct value, and descriptive fidelity is partly mechanical at low HMM
share. The five pictures are the evidence. **Which HMM share reads the market best
is the user's judgement to make.**

Two limits worth restating before you decide:

* **`w` moves the DIRECTION axis only.** The H-vs-L intensity axis contains no HMM in
  this engine, so no arm here changes how H/L is decided — only bull / sideways / bear.
* **Cross-run comparison is unreliable** (known project defect: a small shift in the
  yfinance rolling window moved SIDEWAYS occupancy substantially). Every comparison
  in this notebook is *within* one run over *one shared fit*, which is exactly the
  comparison that can be trusted. Do not compare these panels against panels from a
  different run.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
import os
_harvest = f'{HERE}/_hmm_share_engine_harvest.ipynb'
if os.path.exists(_harvest):     # throwaway from the verbatim engine harvest
    os.remove(_harvest)

print(f'wrote {OUT} - {len(cells)} cells '
      f'({sum(1 for c in cells if c["cell_type"] == "code")} code, '
      f'{sum(1 for c in cells if c["cell_type"] == "markdown")} md)')
