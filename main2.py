import pandas as pd;import numpy as np;import optuna;import warnings;import os;import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.style.use('ggplot')
from data.yfinance_data import download_yf;from data.ccxt_data import download_cx;from features.macroeconomics import macroeconomicos
from model.bases_models.ligthGBM_model import objective_global,train_final_and_predict_test as lgb_predict_test
from model.bases_models.catboost_model import objective_catboost_global,train_final_and_predict_test as cb_predict_test
from model.bases_models.timexer_model import objective_timexer_global,train_final_and_predict_test as tx_predict_test
from model.bases_models.moraiMOE_model import objective_moirai_moe_global,preload_moirai_module,train_final_and_predict_test as moirai_predict_test
from model.meta_model.lstm_model import optimize_lstm_meta,get_average_weights
from preprocessing.oof_generators import build_oof_dataframe
from preprocessing.walk_forward import wfrw;from features.tecnical_indicators import TA;from features.top_n import top_k
from sklearn.preprocessing import MinMaxScaler;import torch
warnings.filterwarnings("ignore")
try:
    from numba import njit,prange;HAS_NUMBA=True
except ImportError:HAS_NUMBA=False
if HAS_NUMBA:
    @njit(parallel=True,cache=True)
    def _recon(yl,cp,n):
        o=np.empty(n,dtype=np.float64)
        for i in prange(n):o[i]=cp[i]*np.exp(yl[i])
        return o
else:
    def _recon(yl,cp,n):return cp*np.exp(yl)
def met(y,p):y,p=np.asarray(y,np.float64),np.asarray(p,np.float64);mse=np.mean((y-p)**2);mae=np.mean(np.abs(y-p));ss=np.sum((y-p)**2);st=np.sum((y-np.mean(y))**2);return {'MSE':round(mse,6),'RMSE':round(np.sqrt(mse),6),'MAE':round(mae,6),'R2':round(1-ss/st if st>0 else 0.,6)}
MDL={'LGB':('#1f77b4','LightGBM'),'CB':('#2ca02c','CatBoost'),'TX':('#9467bd','TimeXer'),'MO':('#ff7f0e','Moirai-MoE'),'MT':('#d62728','Meta LSTM')}
# ===== CONFIG =====
TOKEN='KO'
N_LGB,N_CB,N_TX,N_MO,N_MT=3,3,3,3,3
START,END='2020-01-01','2025-12-31'
# ==================
print(f'[1/9] Descargando datos...');download_yf(['KO','AAPL','NVDA','JNJ','^GSPC','GC=F','CBOE'],START,END);download_cx(['BTC/USDT','ETH/USDT'],START,END)
df=pd.read_csv(os.path.join(os.path.dirname(__file__),'data','tokens',f'{TOKEN}_2020-2025.csv'));lc=np.log(df['Close']/df['Close'].shift(1)).dropna();lc_n=(lc-lc.min())/(lc.max()-lc.min())

# Features
print(f'[2/9] TA + Macro...');df_ta=TA(df);df_ma=macroeconomicos(df['Date_final'])
# MIC
print(f'[3/9] MIC...')
df_ta_r=df_ta.reset_index(drop=True);df_ma_r=df_ma.reset_index(drop=True);lc_r=lc.reset_index(drop=True)
df_f=pd.concat([df_ta_r,df_ma_r],axis=1).iloc[1:];ml=min(len(df_f),len(lc_r));df_f=df_f.iloc[:ml].reset_index(drop=True);lc_r=lc_r.iloc[:ml].reset_index(drop=True)
drop=[c for c in df_f.columns if df_f[c].max()-df_f[c].min()<1e-8];df_f=df_f.drop(columns=drop).replace([np.inf,-np.inf],0.0);lc_r=lc_r.replace([np.inf,-np.inf],0.0)
ts=int(len(df_f)*.9);Xtr,Xte=df_f.iloc[:ts].copy(),df_f.iloc[ts:].copy();ytr,yte=lc_r.iloc[:ts].copy(),lc_r.iloc[ts:].copy()
sf=MinMaxScaler();Xtr_s=pd.DataFrame(sf.fit_transform(Xtr),columns=Xtr.columns,index=Xtr.index);Xte_s=pd.DataFrame(sf.transform(Xte),columns=Xte.columns,index=Xte.index)
sct=MinMaxScaler();ytr_s=pd.Series(sct.fit_transform(ytr.values.reshape(-1,1)).flatten(),index=ytr.index,name='lc');yte_s=pd.Series(sct.transform(yte.values.reshape(-1,1)).flatten(),index=yte.index,name='lc')
feats,mic_v=top_k(Xtr_s,ytr_s,15);di=pd.DataFrame(list(mic_v.items()),columns=['Feature','Score']).sort_values('Score',ascending=True)
Xt,Xe=Xtr_s[feats].reset_index(drop=True),Xte_s[feats].reset_index(drop=True);yt,ye=ytr_s.reset_index(drop=True),yte_s.reset_index(drop=True)

# Walk Forward
print(f'[4/9] Walk-Forward...')
k=5;sp=wfrw(yt,k=k,fh_val=30)

# Training
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(f'[5/9] Entrenando ({device})...')
oof_l,oof_c,oof_t,oof_m={},{},{},{}
print('  LGB...');sl=optuna.create_study(direction='minimize');sl.optimize(lambda t:objective_global(t,Xt,yt,sp,oof_storage=oof_l),n_trials=N_LGB,n_jobs=1);bp_l=oof_l.get('params',sl.best_params)
print('  CB...');sc_=optuna.create_study(direction='minimize');sc_.optimize(lambda t:objective_catboost_global(t,Xt,yt,sp,oof_storage=oof_c),n_trials=N_CB,n_jobs=1);bp_c=oof_c.get('params',sc_.best_params)
print('  TX...');st_=optuna.create_study(direction='minimize');st_.optimize(lambda t:objective_timexer_global(t,Xt,yt,sp,device=device,seq_len=96,pred_len=30,features='MS',oof_storage=oof_t),n_trials=N_TX,n_jobs=1);bp_t=st_.best_params
print('  MO...');preload_moirai_module(model_size='small');sm=optuna.create_study(direction='minimize');sm.optimize(lambda t:objective_moirai_moe_global(t,Xt,yt,sp,device=device,pred_len=30,model_size='small',freq='D',use_full_train=True,oof_storage=oof_m),n_trials=N_MO,n_jobs=1);bp_m=sm.best_params

# Meta LSTM
print(f'[6/9] Meta LSTM...')
oof_df=build_oof_dataframe(oof_l,oof_c,oof_t,oof_m,yt)
print(f'  OOF matrix shape: {oof_df.shape}')
meta_model,mae_meta,meta_results,bp_mt,study_mt=optimize_lstm_meta(oof_df,device,n_trials=N_MT)
if meta_model is not None:
    print(f'  Meta LSTM MAE: {mae_meta:.6f}')
    ws_meta=bp_mt.get('window_size',10)
else:
    print('  WARN: Meta LSTM no entrenado (datos insuficientes)')
    ws_meta=10

# Predictions
print(f'[7/9] Predicciones...')
pl,_=lgb_predict_test(Xt,yt,Xe,bp_l);pc,_=cb_predict_test(Xt,yt,Xe,bp_c)
pt,_,_=tx_predict_test(Xt,yt,Xe,ye,bp_t,device,seq_len=96,pred_len=1,features='MS')
if len(pt)<len(ye):tmp=np.full(len(ye),np.nan);tmp[len(ye)-len(pt):]=pt;pt=tmp
pm,_=moirai_predict_test(yt,ye,bp_m,model_size='small',freq='D')
if len(pm)<len(ye):tmp=np.full(len(ye),np.nan);tmp[len(ye)-len(pm):]=pm;pm=tmp

# Meta LSTM prediction on test (combine base model predictions)
pmt=np.full(len(ye),np.nan)
if meta_model is not None:
    # Build test prediction matrix [n_test, 4] from base model predictions (normalized scale)
    test_matrix=np.column_stack([pl,pc,pt,pm]).astype(np.float32)
    meta_model.eval()
    with torch.no_grad():
        for i in range(ws_meta-1,len(ye)):
            window=test_matrix[i-ws_meta+1:i+1]
            if not np.isnan(window).any():
                x_t=torch.from_numpy(window).unsqueeze(0).to(device)
                pmt[i]=meta_model(x_t).cpu().item()

print(f'[8/9] Metricas...')
yv=ye.values;n=len(yv);idx=np.arange(n);preds={'LGB':pl,'CB':pc,'TX':pt,'MO':pm,'MT':pmt}
# Log return scale
yt_log=sct.inverse_transform(yv.reshape(-1,1)).flatten()
inv=lambda p:sct.inverse_transform(np.where(np.isnan(p),0,p).reshape(-1,1)).flatten()
pl_l,pc_l=inv(pl),inv(pc)
pt_l=np.full_like(pt,np.nan);vt=~np.isnan(pt)
if vt.any():pt_l[vt]=sct.inverse_transform(pt[vt].reshape(-1,1)).flatten()
pm_l=np.full_like(pm,np.nan);vm=~np.isnan(pm)
if vm.any():pm_l[vm]=sct.inverse_transform(pm[vm].reshape(-1,1)).flatten()
pmt_l=np.full_like(pmt,np.nan);vmt=~np.isnan(pmt)
if vmt.any():pmt_l[vmt]=sct.inverse_transform(pmt[vmt].reshape(-1,1)).flatten()
preds_l={'LGB':pl_l,'CB':pc_l,'TX':pt_l,'MO':pm_l,'MT':pmt_l}
# Price scale
cp=df['Close'].values;gi=np.arange(ts,ts+n);val=gi<len(cp);gi_v=gi[val];prev=cp[gi_v-1]
pr_r=_recon(yt_log[val],prev,int(val.sum()));pr_l=_recon(pl_l[val],prev,int(val.sum()));pr_c=_recon(pc_l[val],prev,int(val.sum()))
pr_t=np.where(~np.isnan(pt_l[val]),prev*np.exp(pt_l[val]),np.nan);pr_m=np.where(~np.isnan(pm_l[val]),prev*np.exp(pm_l[val]),np.nan)
pr_mt=np.where(~np.isnan(pmt_l[val]),prev*np.exp(pmt_l[val]),np.nan)
preds_p={'LGB':pr_l,'CB':pr_c,'TX':pr_t,'MO':pr_m,'MT':pr_mt}
# ===== MATPLOTLIB REPORT =====
print(f'[9/9] Generando report.png...')
def _add_preds_ax(ax,x,real,data,rl='Real'):
    ax.plot(x,real,color='black',linewidth=1.5,label=rl)
    for km,(cl,nm) in MDL.items():
        v=data[km];m=~np.isnan(v)
        if m.any():ax.plot(np.asarray(x)[m],v[m],color=cl,linewidth=1.0,label=nm)
    ax.legend(fontsize=5)
mp=[];[(lambda y2,p2,nm:mp.append({'Modelo':nm,**met(y2,p2)}))(pr_r[~np.isnan(v)],v[~np.isnan(v)],MDL[km][1]) for km,v in preds_p.items() if (~np.isnan(v)).any()];mp.sort(key=lambda x:x['MAE'])
fig,axes=plt.subplots(7,3,figsize=(26,40))
fig.suptitle(f'{TOKEN} - Reporte Ensemble',fontsize=22,fontweight='bold',y=0.995)
plt.subplots_adjust(hspace=0.35,wspace=0.3)
# R0C0: Close price
ax=axes[0,0];ax.plot(df['Close'].values,color='#1f77b4',lw=1);ax.set_title(f'Precio de Cierre - {TOKEN}',fontsize=10,fontweight='bold');ax.set_xlabel('Periodo',fontsize=8);ax.set_ylabel('USD',fontsize=8)
# R0C1: Log returns
ax=axes[0,1];ax.plot(lc.values,color='#1f77b4',lw=0.5);ax.set_title('Log Returns',fontsize=10,fontweight='bold');ax.set_xlabel('Periodo',fontsize=8);ax.set_ylabel('Log Return',fontsize=8)
# R0C2: Normalized returns
ax=axes[0,2];ax.plot(lc_n.values,color='#9467bd',lw=0.5);ax.set_title('Retornos Normalizados [0,1]',fontsize=10,fontweight='bold');ax.set_xlabel('Periodo',fontsize=8);ax.set_ylabel('Normalizado',fontsize=8)
# R1C0: MIC bar
ax=axes[1,0];ax.barh(di['Feature'],di['Score'],color='#1f77b4');ax.set_title('MIC Feature Importance (Top 15)',fontsize=10,fontweight='bold');ax.set_xlabel('MIC Score',fontsize=8);ax.tick_params(axis='y',labelsize=5)
# R1C1: Walk-Forward
ax=axes[1,1];fc=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
for fi,(ti,vi) in enumerate(sp.split(yt)):
    ax.plot(yt.index[ti],yt.iloc[ti].values,color=fc[fi],lw=0.7,alpha=0.7)
    ax.plot(yt.index[vi],yt.iloc[vi].values,color=fc[fi],lw=2,ls='--')
ax.set_title('Walk-Forward CV (K=5)',fontsize=10,fontweight='bold')
# R1C2: Predictions normalized
ax=axes[1,2];_add_preds_ax(ax,idx,yv,preds);ax.set_title('Predicciones - Normalizado [0,1]',fontsize=10,fontweight='bold')
# R2C0: Predictions log return
ax=axes[2,0];_add_preds_ax(ax,idx,yt_log,preds_l);ax.set_title('Predicciones - Log Return',fontsize=10,fontweight='bold')
# R2C1: Predictions price
ax=axes[2,1];_add_preds_ax(ax,gi_v,pr_r,preds_p);ax.set_title('Predicciones - Precio (USD)',fontsize=10,fontweight='bold')
# R2C2: Full series + overlay
ax=axes[2,2];ax.plot(cp,color='#cccccc',lw=0.8,label='Close')
for km,(cl,nm) in MDL.items():
    v=preds_p[km];m=~np.isnan(v)
    if m.any():ax.plot(gi_v[m],v[m],color=cl,lw=1.5,ls=':',label=nm)
ax.set_title(f'Serie Completa + Predicciones - {TOKEN}',fontsize=10,fontweight='bold');ax.legend(fontsize=5)
# R3C0: Zoom test
zs,ze=max(0,int(gi_v.min())-50),min(len(cp),int(gi_v.max())+50)
ax=axes[3,0];ax.plot(range(zs,ze),cp[zs:ze],color='black',lw=1.5,label='Close')
for km,(cl,nm) in MDL.items():
    v=preds_p[km];m=~np.isnan(v)
    if m.any():ax.plot(gi_v[m],v[m],color=cl,lw=1,label=nm,marker='o',ms=2)
ax.set_title('Zoom - Zona Test',fontsize=10,fontweight='bold');ax.legend(fontsize=5)
# R3C1,R3C2,R4C0,R4C1,R4C2: Individual models (5 models now)
pos=[(3,1),(3,2),(4,0),(4,1),(4,2)]
for pi,km in enumerate(MDL):
    cl,nm=MDL[km];r,c=pos[pi];ax=axes[r,c]
    ax.plot(gi_v,pr_r,color='black',lw=1.5,label='Real')
    v=preds_p[km];m=~np.isnan(v)
    if m.any():ax.plot(gi_v[m],v[m],color=cl,lw=1.5,label=nm,marker='o',ms=2)
    ax.set_title(f'{nm} vs Real (USD)',fontsize=10,fontweight='bold');ax.legend(fontsize=6)
# R5C0: Meta LSTM weights (if available)
ax=axes[5,0]
if meta_model is not None and meta_results is not None:
    wdf=get_average_weights(meta_results['weights'],meta_results['model_names'])
    ax.barh(wdf['Modelo'],wdf['Peso_Promedio'],color=['#1f77b4','#2ca02c','#9467bd','#ff7f0e'][:len(wdf)])
    ax.set_title('Pesos LSTM por Modelo Base',fontsize=10,fontweight='bold');ax.set_xlabel('Peso Promedio',fontsize=8);ax.tick_params(axis='y',labelsize=7)
else:
    ax.text(0.5,0.5,'Meta LSTM no disponible',ha='center',va='center',transform=ax.transAxes,fontsize=10);ax.set_title('Pesos LSTM',fontsize=10,fontweight='bold')
# R5C1: Meta LSTM training curve
ax=axes[5,1]
if meta_model is not None and meta_results is not None:
    ax.plot(meta_results['train_losses'],color='#1f77b4',lw=1,label='Train');ax.plot(meta_results['val_losses'],color='#ff7f0e',lw=1,label='Val')
    ax.axvline(x=meta_results['best_epoch']-1,color='#d62728',ls='--',lw=1,label=f"Best ep={meta_results['best_epoch']}")
    ax.set_title('LSTM Training Curve',fontsize=10,fontweight='bold');ax.set_xlabel('Epoch',fontsize=8);ax.set_ylabel('MSE Loss',fontsize=8);ax.legend(fontsize=6)
else:
    ax.text(0.5,0.5,'Meta LSTM no disponible',ha='center',va='center',transform=ax.transAxes,fontsize=10);ax.set_title('LSTM Training Curve',fontsize=10,fontweight='bold')
# R5C2,R6C0,R6C1,R6C2: Metrics
mpos=[(5,2),(6,0),(6,1),(6,2)];mcols=['#1f77b4','#2ca02c','#9467bd','#ff7f0e','#d62728']
for mi,mn in enumerate(['MSE','RMSE','MAE','R2']):
    r,c=mpos[mi];ax=axes[r,c]
    ax.bar([x['Modelo'] for x in mp],[x[mn] for x in mp],color=mcols[:len(mp)])
    ax.set_title('R²' if mn=='R2' else mn,fontsize=10,fontweight='bold');ax.tick_params(axis='x',labelsize=7,rotation=30)
plt.tight_layout(rect=[0,0,1,0.98])
out=os.path.join(os.path.dirname(__file__),'report.png')
fig.savefig(out,dpi=200,bbox_inches='tight',facecolor='white')
plt.close(fig);print(f'Listo: {out}')
