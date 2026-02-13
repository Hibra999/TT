import pandas as pd;import numpy as np;import optuna;import warnings;import os;import plotly.graph_objects as go;from plotly.subplots import make_subplots
from data.yfinance_data import download_yf;from data.ccxt_data import download_cx;from features.macroeconomics import macroeconomicos
from model.bases_models.ligthGBM_model import objective_global,train_final_and_predict_test as lgb_predict_test
from model.bases_models.catboost_model import objective_catboost_global,train_final_and_predict_test as cb_predict_test
from model.bases_models.timexer_model import objective_timexer_global,train_final_and_predict_test as tx_predict_test
from model.bases_models.moraiMOE_model import objective_moirai_moe_global,preload_moirai_module,train_final_and_predict_test as moirai_predict_test
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
MDL={'LGB':('#1f77b4','LightGBM'),'CB':('#2ca02c','CatBoost'),'TX':('#9467bd','TimeXer'),'MO':('#ff7f0e','Moirai-MoE')}
# ===== CONFIG =====
TOKEN='KO'
N_LGB,N_CB,N_TX,N_MO=5000,5000,5000,5000
START,END='2020-01-01','2025-12-31'
# ==================
print(f'[1/8] Descargando datos...');download_yf(['KO','AAPL','NVDA','JNJ','^GSPC','GC=F','CBOE'],START,END);download_cx(['BTC/USDT','ETH/USDT'],START,END)
df=pd.read_csv(rf"C:\Users\hibra\Desktop\TT\data\tokens\{TOKEN}_2020-2025.csv");lc=np.log(df['Close']/df['Close'].shift(-1)).dropna();lc_n=(lc-lc.min())/(lc.max()-lc.min())
figs=[]
# 1. Close price
fig=go.Figure();fig.add_trace(go.Scatter(y=df['Close'],mode='lines',name='Close',line=dict(color='#1f77b4',width=1.5)))
fig.update_layout(title=f'Precio de Cierre - {TOKEN}',xaxis_title='Periodo',yaxis_title='USD',height=450);figs.append(fig)
# 2. Log returns
fig=go.Figure();fig.add_trace(go.Scatter(y=lc,mode='lines',name='Log Return',line=dict(color='#1f77b4',width=1)))
fig.update_layout(title='Log Returns',xaxis_title='Periodo',yaxis_title='Log Return',height=450);figs.append(fig)
# 3. Normalized returns
fig=go.Figure();fig.add_trace(go.Scatter(y=lc_n,mode='lines',name='Normalizado',line=dict(color='#9467bd',width=1)))
fig.update_layout(title='Retornos Normalizados [0,1]',xaxis_title='Periodo',yaxis_title='Normalizado',height=400);figs.append(fig)
# Features
print(f'[2/8] TA + Macro...');df_ta=TA(df);df_ma=macroeconomicos(df['Date_final'])
# MIC
print(f'[3/8] MIC...')
df_ta_r=df_ta.reset_index(drop=True);df_ma_r=df_ma.reset_index(drop=True);lc_r=lc.reset_index(drop=True)
df_f=pd.concat([df_ta_r,df_ma_r],axis=1).iloc[1:];ml=min(len(df_f),len(lc_r));df_f=df_f.iloc[:ml].reset_index(drop=True);lc_r=lc_r.iloc[:ml].reset_index(drop=True)
drop=[c for c in df_f.columns if df_f[c].max()-df_f[c].min()<1e-8];df_f=df_f.drop(columns=drop).replace([np.inf,-np.inf],0.0);lc_r=lc_r.replace([np.inf,-np.inf],0.0)
ts=int(len(df_f)*.9);Xtr,Xte=df_f.iloc[:ts].copy(),df_f.iloc[ts:].copy();ytr,yte=lc_r.iloc[:ts].copy(),lc_r.iloc[ts:].copy()
sf=MinMaxScaler();Xtr_s=pd.DataFrame(sf.fit_transform(Xtr),columns=Xtr.columns,index=Xtr.index);Xte_s=pd.DataFrame(sf.transform(Xte),columns=Xte.columns,index=Xte.index)
sct=MinMaxScaler();ytr_s=pd.Series(sct.fit_transform(ytr.values.reshape(-1,1)).flatten(),index=ytr.index,name='lc');yte_s=pd.Series(sct.transform(yte.values.reshape(-1,1)).flatten(),index=yte.index,name='lc')
feats,mic_v=top_k(Xtr_s,ytr_s,15);di=pd.DataFrame(list(mic_v.items()),columns=['Feature','Score']).sort_values('Score',ascending=True)
Xt,Xe=Xtr_s[feats].reset_index(drop=True),Xte_s[feats].reset_index(drop=True);yt,ye=ytr_s.reset_index(drop=True),yte_s.reset_index(drop=True)
# 4. MIC bar
fig=go.Figure(go.Bar(y=di['Feature'],x=di['Score'],orientation='h',marker_color='#1f77b4'))
fig.update_layout(title=f'MIC Feature Importance (Top 15)',xaxis_title='MIC Score',height=500,margin=dict(l=200));figs.append(fig)
# Walk Forward
print(f'[4/8] Walk-Forward...')
k=5;sp=wfrw(yt,k=k,fh_val=30)
fig=make_subplots(k,1,shared_xaxes=True,vertical_spacing=.03,subplot_titles=[f'Fold {i+1}' for i in range(k)])
for i,(ti,vi) in enumerate(wfrw(yt,k=k,fh_val=30).split(yt)):
    fig.add_trace(go.Scatter(x=yt.index[ti].tolist(),y=yt.iloc[ti].tolist(),mode='lines',name='Train',line=dict(color='#1f77b4',width=1.5),showlegend=i==0),i+1,1)
    fig.add_trace(go.Scatter(x=yt.index[vi].tolist(),y=yt.iloc[vi].tolist(),mode='lines',name='Val',line=dict(color='#ff7f0e',width=2),showlegend=i==0),i+1,1)
fig.update_layout(title='Walk-Forward Cross-Validation (K=5)',height=800);figs.append(fig)
# Training
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(f'[5/8] Entrenando ({device})...')
oof_l,oof_c,oof_t,oof_m={},{},{},{}
print('  LGB...');sl=optuna.create_study(direction='minimize');sl.optimize(lambda t:objective_global(t,Xt,yt,sp,oof_storage=oof_l),n_trials=N_LGB,n_jobs=-1);bp_l=oof_l.get('params',sl.best_params)
print('  CB...');sc_=optuna.create_study(direction='minimize');sc_.optimize(lambda t:objective_catboost_global(t,Xt,yt,sp,oof_storage=oof_c),n_trials=N_CB,n_jobs=-1);bp_c=oof_c.get('params',sc_.best_params)
print('  TX...');st_=optuna.create_study(direction='minimize');st_.optimize(lambda t:objective_timexer_global(t,Xt,yt,sp,device=device,seq_len=96,pred_len=30,features='MS',oof_storage=oof_t),n_trials=N_TX,n_jobs=1);bp_t=st_.best_params
print('  MO...');preload_moirai_module(model_size='small');sm=optuna.create_study(direction='minimize');sm.optimize(lambda t:objective_moirai_moe_global(t,Xt,yt,sp,device=device,pred_len=30,model_size='small',freq='D',use_full_train=True,oof_storage=oof_m),n_trials=N_MO,n_jobs=1);bp_m=sm.best_params
# Predictions
print(f'[6/8] Predicciones...')
pl,_=lgb_predict_test(Xt,yt,Xe,bp_l);pc,_=cb_predict_test(Xt,yt,Xe,bp_c)
pt,_,_=tx_predict_test(Xt,yt,Xe,ye,bp_t,device,seq_len=96,pred_len=1,features='MS')
if len(pt)<len(ye):tmp=np.full(len(ye),np.nan);tmp[len(ye)-len(pt):]=pt;pt=tmp
pm,_=moirai_predict_test(yt,ye,bp_m,model_size='small',freq='D')
if len(pm)<len(ye):tmp=np.full(len(ye),np.nan);tmp[len(ye)-len(pm):]=pm;pm=tmp
print(f'[7/8] Metricas...')
yv=ye.values;n=len(yv);idx=np.arange(n);preds={'LGB':pl,'CB':pc,'TX':pt,'MO':pm}
# Log return scale
yt_log=sct.inverse_transform(yv.reshape(-1,1)).flatten()
inv=lambda p:sct.inverse_transform(np.where(np.isnan(p),0,p).reshape(-1,1)).flatten()
pl_l,pc_l=inv(pl),inv(pc)
pt_l=np.full_like(pt,np.nan);vt=~np.isnan(pt)
if vt.any():pt_l[vt]=sct.inverse_transform(pt[vt].reshape(-1,1)).flatten()
pm_l=np.full_like(pm,np.nan);vm=~np.isnan(pm)
if vm.any():pm_l[vm]=sct.inverse_transform(pm[vm].reshape(-1,1)).flatten()
preds_l={'LGB':pl_l,'CB':pc_l,'TX':pt_l,'MO':pm_l}
# Price scale
cp=df['Close'].values;gi=np.arange(ts,ts+n);val=gi<len(cp);gi_v=gi[val];prev=cp[gi_v-1]
pr_r=_recon(yt_log[val],prev,int(val.sum()));pr_l=_recon(pl_l[val],prev,int(val.sum()));pr_c=_recon(pc_l[val],prev,int(val.sum()))
pr_t=np.where(~np.isnan(pt_l[val]),prev*np.exp(pt_l[val]),np.nan);pr_m=np.where(~np.isnan(pm_l[val]),prev*np.exp(pm_l[val]),np.nan)
preds_p={'LGB':pr_l,'CB':pr_c,'TX':pr_t,'MO':pr_m}
# helper
def add_preds(fig,x,real,data,rl='Real'):
    fig.add_trace(go.Scatter(x=x,y=real,mode='lines',name=rl,line=dict(color='black',width=2)))
    for k,(c,nm) in MDL.items():
        v=data[k];m=~np.isnan(v)
        if m.any():fig.add_trace(go.Scatter(x=np.asarray(x)[m],y=v[m],mode='lines',name=nm,line=dict(color=c,width=1.5)))
# 5. Predictions normalized
fig=go.Figure();add_preds(fig,idx,yv,preds);fig.update_layout(title='Predicciones - Normalizado [0,1]',xaxis_title='Indice',yaxis_title='Normalizado',height=500);figs.append(fig)
# 6. Predictions log return
fig=go.Figure();add_preds(fig,idx,yt_log,preds_l);fig.update_layout(title='Predicciones - Log Return',xaxis_title='Indice',yaxis_title='Log Return',height=500);figs.append(fig)
# 7. Predictions price
fig=go.Figure();add_preds(fig,gi_v,pr_r,preds_p);fig.update_layout(title='Predicciones - Precio (USD)',xaxis_title='Indice',yaxis_title='USD',height=500);figs.append(fig)
# 8. Full series + predictions overlay
fig=go.Figure();fig.add_trace(go.Scatter(y=cp,mode='lines',name='Close',line=dict(color='#ccc',width=1)))
for k,(c,nm) in MDL.items():
    v=preds_p[k];m=~np.isnan(v)
    if m.any():fig.add_trace(go.Scatter(x=gi_v[m],y=v[m],mode='lines',name=nm,line=dict(color=c,width=2,dash='dot')))
fig.update_layout(title=f'Serie Completa + Predicciones - {TOKEN}',xaxis_title='Indice',yaxis_title='USD',height=500);figs.append(fig)
# 9. Zoom test zone
zs,ze=max(0,int(gi_v.min())-50),min(len(cp),int(gi_v.max())+50)
fig=go.Figure();fig.add_trace(go.Scatter(x=list(range(zs,ze)),y=cp[zs:ze].tolist(),mode='lines',name='Close',line=dict(color='black',width=2)))
for k,(c,nm) in MDL.items():
    v=preds_p[k];m=~np.isnan(v)
    if m.any():fig.add_trace(go.Scatter(x=gi_v[m].tolist(),y=v[m].tolist(),mode='lines+markers',name=nm,line=dict(color=c,width=1.5),marker=dict(size=3)))
fig.update_layout(title='Zoom - Zona Test',xaxis_title='Indice',yaxis_title='USD',height=450);figs.append(fig)
# 10-13. Individual per model
for k,(c,nm) in MDL.items():
    fig=go.Figure();fig.add_trace(go.Scatter(x=gi_v.tolist(),y=pr_r.tolist(),mode='lines',name='Real',line=dict(color='black',width=2)))
    v=preds_p[k];m=~np.isnan(v)
    if m.any():fig.add_trace(go.Scatter(x=gi_v[m].tolist(),y=v[m].tolist(),mode='lines+markers',name=nm,line=dict(color=c,width=2),marker=dict(size=3)))
    fig.update_layout(title=f'{nm} vs Real (USD)',xaxis_title='Indice',yaxis_title='USD',height=400);figs.append(fig)
# 14. Metrics bar
mp=[];[(lambda y2,p2,nm:mp.append({'Modelo':nm,**met(y2,p2)}))(pr_r[~np.isnan(v)],v[~np.isnan(v)],MDL[k][1]) for k,v in preds_p.items() if (~np.isnan(v)).any()];mp.sort(key=lambda x:x['MAE'])
fig=make_subplots(2,2,subplot_titles=['MSE','RMSE','MAE','R²'])
cols=['#1f77b4','#2ca02c','#9467bd','#ff7f0e']
for i,m_ in enumerate(['MSE','RMSE','MAE','R2']):
    fig.add_trace(go.Bar(x=[x['Modelo'] for x in mp],y=[x[m_] for x in mp],marker_color=cols[:len(mp)],showlegend=False),i//2+1,i%2+1)
fig.update_layout(title='Comparacion de Metricas (Precio USD)',height=600);figs.append(fig)
# ===== HTML =====
print(f'[8/8] Generando report.html...')
html='<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Report</title><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body style="max-width:1000px;margin:0 auto;padding:20px;font-family:sans-serif">\n'
html+=f'<h2 style="text-align:center">{TOKEN} - Reporte Ensemble</h2>\n'
for i,fig in enumerate(figs):
    html+=fig.to_html(full_html=False,include_plotlyjs=False)+'\n<hr>\n'
html+='</body></html>'
out=os.path.join(os.path.dirname(__file__),'report.html')
with open(out,'w',encoding='utf-8') as f:f.write(html)
print(f'Listo: {out}')
