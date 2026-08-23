"""Prove the Pine port reproduces the engine.

Mirrors regdet.pine LINE FOR LINE in Python, then diffs against the real
engine. Three separate questions, kept separate:
  A. does the PINE LOGIC match the engine at w=1.0, same (frozen) baseline?
     -> if yes, the port is correct
  B. what does the EXPANDING baseline (the live-chart deviation) cost?
  C. what does dropping the HMM (w=0.5 -> 1.0) cost?
"""
import json, re, io, contextlib, numpy as np, pandas as pd
NB='/workspace/sherm-quanty/regdet/notebooks/fx_v9.ipynb'
nb=json.load(open(NB)); MAGIC=re.compile(r'^\s*%[A-Za-z]')
def cell(i): return '\n'.join('' if MAGIC.match(l) else l for l in ''.join(nb['cells'][i]['source']).split('\n'))
ns={'__name__':'__main__'}
with contextlib.redirect_stdout(io.StringIO()):
    for i in (2,4,6,8,10):
        exec(compile(cell(i),f'<c{i}>','exec'),ns)
g=ns
nifty,vix=g['nifty'],g['vix']
build_features=g['build_features']; label_bars=g['label_bars']
fit_hmm_ensemble=g['fit_hmm_ensemble']; FEATURE_COLS=g['FEATURE_COLS']
TREND_FEATURE=g['TREND_FEATURE']; TRAIN_FRACTION=g['TRAIN_FRACTION']
EFF_WIN=g['EFF_WIN']; BASE_WIN=g['BASE_WIN']; CONF_L=g['CONF_L']
CONFIRM_BARS=g['CONFIRM_BARS']; Z_HI=g['Z_HI']; EFF_HI=g['EFF_HI']
Z_HI_EXIT=g['Z_HI_EXIT']; EFF_HI_EXIT=g['EFF_HI_EXIT']; TAU=g['BAR_DIR_TAU']

edf=build_features(nifty,vix,g['LOOKBACK_SCALE'])
eX=edf[FEATURE_COLS].values; n_fit=max(int(len(eX)*TRAIN_FRACTION),50)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler().fit(eX[:n_fit]); eXs=sc.transform(eX)
with contextlib.redirect_stdout(io.StringIO()):
    models,_=fit_hmm_ensemble(eXs[:n_fit],5,'diag',K=g['ENSEMBLE_K'],base_seed=g['BASE_SEED'])
close=nifty.reindex(edf.index)

def engine(w):
    with contextlib.redirect_stdout(io.StringIO()):
        d=label_bars(models,eXs,edf.index,nifty,edf[TREND_FEATURE].values,n_fit,
                     FEATURE_COLS,dir_feats=edf,bar_dir_weight=w)
    return d['tactical_regime_state'].values

# ---------------- the Pine mirror ----------------
def pine(expanding):
    px=close.values; N=len(px)
    def z(x):
        x=np.asarray(x,float); out=np.zeros(N)
        if expanding:
            s=np.cumsum(x); s2=np.cumsum(x*x); n=np.arange(1,N+1)
            mu=s/n; sd=np.sqrt(np.maximum(s2/n-mu*mu,0.0))
        else:                                   # frozen leading-n_fit baseline
            mu=np.full(N,x[:n_fit].mean()); sd=np.full(N,x[:n_fit].std())
        ok=sd>0
        out[ok]=(x[ok]-mu[ok])/sd[ok]
        return out
    zr,z1=z(edf['ret_2h'].values),z(edf['mom_1d'].values)
    z3,z5=z(edf['mom_3d'].values),z(edf['mom_5d'].values)
    zd=z(edf['dist_ma'].values)
    comp=(1.0*zr+0.4*z1+0.4*z3+0.4*z5+1.0*zd)/3.2
    s=z(comp)
    e=s/TAU; a=np.abs(e)
    eb,es_,er=np.exp(e-a),np.exp(-a),np.exp(-e-a); tot=eb+es_+er
    bull,side,bear=eb/tot,es_/tot,er/tot
    conf=np.maximum(np.maximum(bull,side),bear)
    draw=np.where((bull>=side)&(bull>=bear),1,np.where((bear>=side)&(bear>=bull),-1,0))
    draw=np.where(conf<CONF_L,0,draw)
    # confirm_delay
    de=np.empty(N,int); de[0]=draw[0]; cand,run=draw[0],0
    for t in range(1,N):
        if draw[t]==de[t-1]: de[t]=draw[t]; cand,run=draw[t],0
        else:
            if draw[t]==cand: run+=1
            else: cand,run=draw[t],1
            if run>=CONFIRM_BARS: de[t]=cand; run=0
            else: de[t]=de[t-1]
    tz=z3
    net=np.full(N,np.nan); net[EFF_WIN:]=np.abs(px[EFF_WIN:]-px[:-EFF_WIN])
    dif=np.abs(np.diff(px,prepend=px[0]))
    path=pd.Series(dif).rolling(EFF_WIN).sum().values
    with np.errstate(invalid='ignore',divide='ignore'):
        eff=np.clip(np.where(path>0,net/path,0.0),0,1)
    eff=np.nan_to_num(eff)
    intens=np.zeros(N,int); cur=0
    for t in range(N):
        sgn=1 if tz[t]>0 else (-1 if tz[t]<0 else 0)
        ce=abs(tz[t])>=Z_HI and eff[t]>=EFF_HI
        ch=abs(tz[t])>=Z_HI_EXIT and eff[t]>=EFF_HI_EXIT
        cur=cur if (cur!=0 and ch and sgn==cur) else (sgn if (ce and sgn!=0) else 0)
        intens[t]=cur
    lab=np.where(de==1,np.where(intens==1,'H_BULL','L_BULL'),
        np.where(de==-1,np.where(intens==-1,'H_BEAR','L_BEAR'),'SIDEWAYS'))
    return lab

e10,e05=engine(1.0),engine(0.5)
pf,pe=pine(False),pine(True)
N=len(e10)
def agree(a,b): return 100.0*np.mean(a==b)
print(f'bars compared: {N}   instrument: EUR/USD 8h   (frozen baseline = leading {n_fit})')
print('='*72)
print(f'A. PINE LOGIC vs ENGINE, both w=1.0, same frozen baseline : {agree(pf,e10):6.2f}% identical')
print(f'   -> this is the port-correctness test')
print(f'B. expanding baseline vs frozen, pine logic both          : {agree(pe,pf):6.2f}% identical')
print(f'   -> cost of the live-chart baseline deviation')
print(f'C. engine w=1.0 (no HMM) vs w=0.5 (shipped)               : {agree(e10,e05):6.2f}% identical')
print(f'   -> cost of dropping the HMM half')
print(f'D. PINE as shipped (expanding) vs SHIPPED engine (w=0.5)  : {agree(pe,e05):6.2f}% identical')
occ=lambda a:{k:round(100*np.mean(a==k),1) for k in ['H_BULL','L_BULL','SIDEWAYS','L_BEAR','H_BEAR']}
print('\noccupancy %')
for nm,a in (('engine w=0.5 (shipped)',e05),('engine w=1.0',e10),('pine frozen',pf),('pine expanding',pe)):
    print(f'  {nm:<24}{occ(a)}')
