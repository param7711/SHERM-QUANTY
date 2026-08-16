"""WHY did adaptive land on ctx12 instead of between the two controls?
The mechanism assumes LOW-VOL coincides with the GRIND cell. Test that
directly -- no HMM fits needed, so this is cheap."""
import os, io, contextlib
import numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SHERM-QUANTY/54e2e996-5402-59e6-a072-0f125e030353/scratchpad'
GEN='/workspace/sherm-quanty/regdet/generators/build_intraday_fx.py'
os.environ.setdefault('REGDET_OUT_DIR', SP+'/probe')
src=open(GEN).read().replace("'intraday_fx.ipynb'","'_w.ipynb'",1)
g={'__name__':'_w','__file__':GEN}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src,GEN,'exec'),g)
ns={'__name__':'__main__'}
with contextlib.redirect_stdout(io.StringIO()):
    for i in [3,4,5,6,7,9,10,11,12,13,14,18,19,20,22]:
        exec(compile(''.join(g['cells'][i]['source']),f'<c{i}>','exec'),ns)
np_,pd_=ns['np'],ns['pd']
build_features=ns['build_features']; bars_per_day=ns['bars_per_day']
rvp=ns['realised_vol_proxy']; load=ns['load_fx_h1']; TF=ns['TRAIN_FRACTION']
BPD=24; EFF_W=ns['EFF_WIN']; SLOW=20*BPD
INST=[('EURUSD','EUR/USD',(0.9,1.7)),('GBPUSD','GBP/USD',(0.9,1.7)),
      ('USDJPY','USD/JPY',(75.,160.)),('XAUUSD','XAU/USD',(1000.,2100.)),
      ('AUDUSD','AUD/USD',(0.55,1.15)),('USDCAD','USD/CAD',(0.9,1.6)),
      ('USDCHF','USD/CHF',(0.7,1.15))]
def feats_at(c,v,cd):
    o=ns['CONTEXT_DAYS']; ns['CONTEXT_DAYS']=cd
    try:
        with bars_per_day(BPD): return build_features(c,v,ns['LOOKBACK_SCALE'])
    finally: ns['CONTEXT_DAYS']=o
def eff(px,w):
    lp=np.log(px); d=np.abs(np.diff(lp,prepend=lp[0]))
    net=np.full(len(px),np.nan); net[w:]=np.abs(lp[w:]-lp[:-w])
    tot=np.convolve(d,np.ones(w),'full')[:len(px)]
    with np.errstate(invalid='ignore',divide='ignore'): e=net/tot
    return np.clip(e,0,1)
print(f"{'instrument':<10}{'P(LOWVOL|grind)':>17}{'P(LOWVOL|violent)':>19}{'P(LOWVOL)':>11}{'lift':>8}")
print('-'*70)
rows=[]
for sym,name,band in INST:
    h1,_=load(dict(sym=sym,name=name,med_band=band))
    c=h1['close'].astype(float); c.name=sym
    with bars_per_day(BPD): v=rvp(c,BPD)
    f12,f24=feats_at(c,v,12),feats_at(c,v,24)
    idx=f24.index.intersection(f12.index)
    f12,f24=f12.loc[idx],f24.loc[idx]
    n=len(idx); nfit=max(int(n*TF),50)
    sel=f12['vol_expansion'].values
    base=sel[:nfit]; base=base[np.isfinite(base)]
    en,ex=np.quantile(base,0.40),np.quantile(base,0.60)
    st=np.zeros(n,bool); cur=bool(np.isfinite(sel[0]) and sel[0]<en)
    for t in range(n):
        x=sel[t]
        if np.isfinite(x):
            if cur and x>ex: cur=False
            elif (not cur) and x<en: cur=True
        st[t]=cur
    px=c.reindex(idx).values
    ef,es=eff(px,EFF_W),eff(px,SLOW)
    m=np.zeros(n,bool); m[nfit:]=True; m&=np.isfinite(ef)&np.isfinite(es)
    mf,ms=np.nanmedian(ef[nfit:]),np.nanmedian(es[nfit:])
    grind=m&(ef<mf)&(es>=ms)
    q=np.nanquantile(ef[nfit:],0.8); viol=m&(ef>=q)
    pg,pv,pb=st[grind].mean()*100,st[viol].mean()*100,st[m].mean()*100
    rows.append((pg,pv,pb))
    print(f"{name:<10}{pg:>17.1f}{pv:>19.1f}{pb:>11.1f}{pg-pb:>+8.1f}")
print('-'*70)
a=np.array(rows)
print(f"{'MEAN':<10}{a[:,0].mean():>17.1f}{a[:,1].mean():>19.1f}{a[:,2].mean():>11.1f}{(a[:,0]-a[:,2]).mean():>+8.1f}")
print()
print('READ: if P(LOW-VOL | grind) is close to the base rate P(LOW-VOL), the')
print('volatility selector carries almost NO information about whether the')
print('market is in a grind -- and the adaptive window is then widening at')
print('times unrelated to the problem it was built to solve.')
