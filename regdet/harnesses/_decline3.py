import json,warnings,io,contextlib,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
nb=json.load(open('regdet_v11_master.ipynb'))
code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
code=[s for s in code if not s.lstrip().startswith('%')]
g={'__name__':'__main__'}
for i,src in enumerate(code):
    with contextlib.redirect_stdout(io.StringIO()):
        try: exec(compile(src,f'<c{i}>','exec'),g)
        except Exception: pass
    if 'label_bars' in g and i>=6: break
from sklearn.preprocessing import StandardScaler
close=pd.read_pickle('_probe_close.pkl'); close.name='nifty'
r=np.log(close/close.shift(1))
vix=(r.rolling(20).std()*np.sqrt(252)*100).bfill().ffill(); vix.name='vix'

def run(scale,label):
    feat=g['build_features'](close,vix,scale=scale)
    FS=list(feat.columns); X=feat.values; n=len(X); nfit=int(n*0.70)
    sc=StandardScaler().fit(X[:nfit]); Xs=sc.transform(X)
    models,_=g['fit_hmm_ensemble'](Xs[:nfit],5,'diag')
    out=g['label_bars'](models,Xs,feat.index,close,feat[g['TREND_FEATURE']].values,nfit,FS,dir_feats=feat)
    lab=out['tactical_regime_state'].values
    px=close.reindex(feat.index).values
    dec=pd.Series(px).pct_change(20).values<=-0.08
    bear=np.isin(lab,['L_BEAR','H_BEAR']).astype(float)
    bull=np.isin(lab,['H_BULL','L_BULL']).astype(float)
    side=(lab=='SIDEWAYS').astype(float)
    print(f'{label:38s} bear {100*bear[dec].mean():5.1f}%   bull {100*bull[dec].mean():5.1f}%   side {100*side[dec].mean():5.1f}%   (n={dec.sum()})')
    return None
print('LABELS INSIDE SUSTAINED DECLINES (20-bar return <= -8%), same data, same fit config')
print('effective reach = longest feature window (SWING_WIN=20 bars) x scale\n')
run(0.333,"NEW cfg equiv: mom 5d, context 12d")
run(0.186,"OLD cfg equiv: mom 2.8d, context 6.7d")
run(0.556,"context 20d (measured saturation)")
