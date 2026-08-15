"""Execute a notebook cell-by-cell and write an EXECUTED copy with outputs.

Same execution model the verification harnesses use -- each cell compiled
SEPARATELY against one shared namespace, Jupyter magics blanked (cell 0 is
always `%pip install`, a SyntaxError in plain Python and NOT a failure). The
difference is that stdout and figures are captured per cell and stored back
into the notebook JSON, so the delivered .ipynb carries its own results.

Usage:  python3 _execute_nb.py <notebook.ipynb> <out_dir>
Writes: <out_dir>/<name>_executed.ipynb  and  <out_dir>/<name>_figNN.png
"""
import base64, io, json, os, re, sys, time, traceback, contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = os.path.abspath(sys.argv[1])
OUTD = os.path.abspath(sys.argv[2])
NAME = os.path.splitext(os.path.basename(SRC))[0]
os.makedirs(OUTD, exist_ok=True)
MAGIC = re.compile(r'^\s*%[A-Za-z]')

nb = json.load(open(SRC))
ns = {'__name__': '__main__'}
figs, fails = 0, []
t0 = time.time()

for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = '\n'.join('' if MAGIC.match(l) else l
                    for l in ''.join(c['source']).split('\n'))
    buf = io.StringIO()
    outputs = []
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception as e:                                    # noqa: BLE001
        fails.append((i, traceback.format_exc()[-2000:]))
        outputs.append({'output_type': 'stream', 'name': 'stderr',
                        'text': traceback.format_exc()[-2000:].splitlines(True)})
        print(f'!!! CELL {i} FAILED: {type(e).__name__}: {e}', flush=True)

    text = buf.getvalue()
    if text:
        outputs.insert(0, {'output_type': 'stream', 'name': 'stdout',
                           'text': text.splitlines(keepends=True)})
    for n in plt.get_fignums():
        figs += 1
        f = plt.figure(n)
        png = f'{OUTD}/{NAME}_fig{figs:02d}.png'
        f.savefig(png, dpi=130, bbox_inches='tight')
        with open(png, 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode('ascii')
        outputs.append({'output_type': 'display_data',
                        'data': {'image/png': b64}, 'metadata': {}})
    plt.close('all')

    c['outputs'] = outputs
    c['execution_count'] = i
    print(f'cell {i:3}  {len(text):6} chars stdout', flush=True)

dst = f'{OUTD}/{NAME}_executed.ipynb'
with open(dst, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'\nWROTE {dst}')
print(f'CELLS_FAILED {len(fails)}  FIGURES {figs}  '
      f'RUNTIME {time.time() - t0:.1f}s')
for i, tb in fails:
    print('=' * 20, i, '\n', tb)
sys.exit(1 if fails else 0)
