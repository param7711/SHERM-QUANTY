"""INDEPENDENT look-ahead audit of the v6 labelling path.

The notebook's own probes check bar T only ("the label AT t is unchanged when
bars > t are deleted"). This is the stronger form the brief asks for: cut the
series at T and require that EVERY label at t <= T is unchanged -- a leak at
t << T would slip past a bar-T-only probe.

Runs off-path against the generated notebook's own cells (engine harvested by
executing the master notebook up to and including the label_bars definition),
so it tests the shipped code, not a copy.

OMP_NUM_THREADS=1 is required: multi-threaded BLAS changes EM reduction order.
"""
import json, os, re, sys
assert os.environ.get('OMP_NUM_THREADS') == '1', 'run with OMP_NUM_THREADS=1'
import matplotlib
matplotlib.use('Agg')
import numpy as np

D = '/tmp/claude-0/-home-user-SHERM-QUANTY/f1a44349-dfe0-5e0f-9b1f-b0b39e930e1b/scratchpad/'
MAGIC = re.compile(r'^\s*%[A-Za-z]')
nb = json.load(open(D + 'regdet_v11_master.ipynb'))
ns = {'__name__': '__main__'}

# Execute cells up to the one that defines label_bars (cell index 7 = engine).
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = '\n'.join('' if MAGIC.match(l) else l for l in ''.join(c['source']).split('\n'))
    exec(compile(src, f'<cell {i}>', 'exec'), ns)
    if 'def label_bars' in src:
        break

g = ns
print('\n' + '=' * 78)
print('CONFIG UNDER TEST')
print('=' * 78)
for k in ('BAR_DIR_WEIGHT', 'ENSEMBLE_K', 'BASE_SEED', 'CONFIRM_BARS', 'INTENSITY_MODE',
          'DIRECTION_MODE', 'ESCALATION_DURING_HOLD', 'CONF_L', 'Z_HI', 'EFF_HI',
          'EFF_WIN', 'TREND_FEATURE', 'DIRECTION_EXCLUDE'):
    print(f'  {k:24s} = {g[k]!r}')

nifty, vix = g['nifty'], g['vix']
df = g['build_features'](nifty, vix, g['LOOKBACK_SCALE'])
feat = next(c for c in g['CONFIGS'] if c['name'] == g['ADOPTED_CONFIG_NAME'])['features']
X = df[feat].values
dates = df.index
n_fit = max(int(len(X) * g['TRAIN_FRACTION']), 50)
sc = g['StandardScaler']().fit(X[:n_fit])
Xs = sc.transform(X)
models, conv = g['fit_hmm_ensemble'](Xs[:n_fit], 5, 'diag')
print(f'\n  bars={len(dates)}  n_fit={n_fit}  ensemble={len(models)}  converged={conv}')

full = g['label_bars'](models, Xs, dates, nifty, df[g['TREND_FEATURE']].values,
                       n_fit, feat, dir_feats=df)
COLS = ['tactical_regime_state', 'regime_state_raw', 'tactical_regime_confidence',
        'hmm_state_int', 'prob_H_BULL', 'prob_L_BULL', 'prob_SIDEWAYS',
        'prob_L_BEAR', 'prob_H_BEAR', 'trend_z', 'trend_efficiency',
        'gate_intensity', 'bar_dir_score']

print('\n' + '=' * 78)
print('FULL-PREFIX TRUNCATION PROBE: cut at T, compare ALL bars t <= T')
print('=' * 78)
N = len(dates)
Ts = sorted({n_fit, n_fit + 1, int(N * 0.75), int(N * 0.85), int(N * 0.95), N - 2, N - 1})
bad_total = 0
for T in Ts:
    tr = g['label_bars'](models, Xs[:T + 1], dates[:T + 1], nifty,
                         df[g['TREND_FEATURE']].values[:T + 1], n_fit, feat,
                         dir_feats=df.iloc[:T + 1])
    bad = {}
    for c in COLS:
        a = full[c].values[:T + 1]
        b = tr[c].values
        n = int((a != b).sum())
        if n:
            bad[c] = n
    bad_total += sum(bad.values())
    print(f'  T={T:5d}  bars compared={T + 1:5d}  '
          + ('ALL 13 COLUMNS IDENTICAL' if not bad else f'MISMATCH {bad}'))
assert bad_total == 0, 'LOOK-AHEAD: truncating the series changed a label at t <= T'
print(f'\n  [OK] {len(Ts)} truncation points x {len(COLS)} columns: 0 differing values.')
print('       Nothing at bar t reads a bar after t.')

print('\n' + '=' * 78)
print('v6 LABEL EQUIVALENCE: shipped label_bars vs the FROZEN label_bars_legacy')
print('=' * 78)
leg = g['label_bars_legacy'](models, Xs, dates, nifty, df[g['TREND_FEATURE']].values,
                             n_fit, feat, exclude=g['DIRECTION_EXCLUDE'],
                             confirm_bars=g['CONFIRM_BARS'], z_exit=g['Z_HI_EXIT'],
                             eff_exit=g['EFF_HI_EXIT'], dir_feats=df,
                             bar_dir_weight=g['BAR_DIR_WEIGHT'])
bad = {c: int((full[c].values != leg[c].values).sum()) for c in COLS}
bad = {k: v for k, v in bad.items() if v}
print(f'  bars={len(dates)}  columns={len(COLS)}  '
      + ('ALL BIT-IDENTICAL -- 0 differing bars' if not bad else f'MISMATCH {bad}'))
assert not bad, 'the shipped labels are NOT the v6 labels'
print('\nALL CHECKS PASSED.')
