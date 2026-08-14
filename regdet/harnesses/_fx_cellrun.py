"""Run intraday_fx.ipynb the way Jupyter does: each cell compiled SEPARATELY
into its own code object against ONE shared namespace. Concatenating cells into
a single file breaks `from __future__ import annotations` and has wasted time on
this project before -- so it is not done here.

Usage:  python3 _fx_cellrun.py [FIG_SUBDIR]
        NB_DIR=/some/dir python3 _fx_cellrun.py     # notebook lives elsewhere

Cell 0 is `%pip install` -- a Jupyter magic and a SyntaxError in plain Python.
Magic lines are blanked, NOT treated as failures.
"""
import json, re, sys, time, traceback, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Was pinned to one session's scratchpad, which made the harness unusable from
# a fresh clone. Default to the repo's notebooks/ dir; NB_DIR overrides.
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get('NB_DIR', os.path.join(os.path.dirname(HERE), 'notebooks'))
FD = os.path.join(D, sys.argv[1] if len(sys.argv) > 1 else '_fx_figs')
MAGIC = re.compile(r'^\s*%[A-Za-z]')

nb = json.load(open(os.path.join(D, 'intraday_fx.ipynb')))
ns = {'__name__': '__main__'}
os.makedirs(FD, exist_ok=True)
figs, fails = 0, []
t0 = time.time()
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = '\n'.join('' if MAGIC.match(l) else l
                    for l in ''.join(c['source']).split('\n'))
    print(f'\n===== CELL {i} =====', flush=True)
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception as e:
        fails.append((i, traceback.format_exc()[-3000:]))
        print(f'!!! CELL {i} FAILED: {type(e).__name__}: {e}', flush=True)
    for n in plt.get_fignums():
        figs += 1
        plt.figure(n).savefig(f'{FD}/fig{figs:02d}_cell{i}.png', dpi=110,
                              bbox_inches='tight')
    plt.close('all')
print(f'\nCELLS_FAILED {len(fails)}  FIGURES {figs}  RUNTIME {time.time()-t0:.1f}s')
for i, tb in fails:
    print('=' * 20, i, '\n', tb)
sys.exit(1 if fails else 0)
