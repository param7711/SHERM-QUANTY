import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
nb=json.load(open('regdet_v11_master.ipynb'))
code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g={'__name__':'__main__'}
figs=0; fails=[]
for i,src in enumerate(code):
    try:
        exec(compile(src,f'<cell{i}>','exec'),g)   # each cell its own code object
    except Exception as e:
        fails.append((i,type(e).__name__,str(e)[:120]))
    figs+=len(plt.get_fignums()); plt.close('all')
print('CELLS',len(code),'FAILED',len(fails),'FIGURES',figs)
for f in fails: print('  FAIL',f)
