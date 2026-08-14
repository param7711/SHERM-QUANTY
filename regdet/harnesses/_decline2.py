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

close=pd.read_pickle('_probe_close.pkl'); close.name='nifty'
r=np.log(close/close.shift(1))
vix=(r.rolling(20).std()*np.sqrt(252)*100).bfill().ffill(); vix.name='vix'   # proxy only
feat=g['build_features'](close,vix)
FS=[c for c in feat.columns]
X=feat.values; n=len(X); nfit=int(n*0.70)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler().fit(X[:nfit]); Xs=sc.transform(X)
models,_=g['fit_hmm_ensemble'](Xs[:nfit], 5, 'diag')
out=g['label_bars'](models,Xs,feat.index,close,feat[g['TREND_FEATURE']].values,nfit,FS,dir_feats=feat)
lab=out['tactical_regime_state'].values; conf=out['tactical_regime_confidence'].values
px=close.reindex(feat.index).values

# steep sustained declines: 20-bar return <= -8%
fwd=pd.Series(px).pct_change(20).values
dec=fwd<=-0.08
print(f'\nbars={n}  decline bars (20d ret <= -8%): {dec.sum()} ({100*dec.mean():.1f}%)')
print('\nLABEL MIX')
print(f'{"label":10s}{"all bars":>10s}{"IN DECLINE":>12s}')
for L in ['H_BULL','L_BULL','SIDEWAYS','L_BEAR','H_BEAR']:
    print(f'{L:10s}{100*(lab==L).mean():9.1f}%{100*(lab[dec]==L).mean() if dec.sum() else 0:11.1f}%')
bear=np.isin(lab,['L_BEAR','H_BEAR']); bull=np.isin(lab,['H_BULL','L_BULL'])
print(f'\n  BEAR during declines : {100*bear[dec].mean():.1f}%   (want: high)')
print(f'  BULL during declines : {100*bull[dec].mean():.1f}%   (want: ~0)')
print(f'  SIDE during declines : {100*(lab[dec]=="SIDEWAYS").mean():.1f}%')
print('\nWHY -- winning direction mass (CONF_L override fires below %.2f)'%g['CONF_L'])
print(f'  median winning mass, all bars      : {np.median(conf):.3f}')
print(f'  median winning mass, in declines   : {np.median(conf[dec]):.3f}')
print(f'  %% bars below CONF_L, all           : {100*(conf<g["CONF_L"]).mean():.1f}%')
print(f'  %% bars below CONF_L, in declines   : {100*(conf[dec]<g["CONF_L"]).mean():.1f}%')
