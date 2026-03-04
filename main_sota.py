import pandas as pd;import numpy as np;import optuna;import warnings;import os
from data.yfinance_data import download_yf;from data.ccxt_data import download_cx;from features.macroeconomics import macroeconomicos
from model.bases_models.ligthGBM_model import objective_global,train_final_and_predict_test as lgb_predict_test
from model.bases_models.catboost_model import objective_catboost_global,train_final_and_predict_test as cb_predict_test
from model.sota.stacking_ensemble import (
    objective_xgboost_global,train_final_xgb,
    objective_base_lstm_global,train_final_base_lstm,
    build_oof_dataframe_sota,optimize_stacking_meta
)
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
MDL={'LGB':('#1f77b4','LightGBM'),'CB':('#2ca02c','CatBoost'),'XG':('#e377c2','XGBoost'),'BL':('#8c564b','Base LSTM'),'MT':('#d62728','Meta LSTM')}
# ===== CONFIG =====
TOKEN='ETH/USDT'
N_LGB,N_CB,N_XG,N_BL,N_MT=3,3,3,3,3
START,END='2020-01-01','2025-12-31'
# ==================
print(f'[1/9] Descargando datos...');download_yf(['KO','AAPL','NVDA','JNJ','^GSPC','GC=F','CBOE'],START,END);download_cx(['BTC/USDT','ETH/USDT'],START,END)
df=pd.read_csv(os.path.join(os.path.dirname(__file__),'data','tokens',f'{TOKEN.replace("/","-")}_2020-2025.csv'));lc=np.log(df['Close']/df['Close'].shift(1)).dropna();lc_n=(lc-lc.min())/(lc.max()-lc.min())

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
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(f'[5/9] Entrenando SOTA ({device})...')
oof_l,oof_c,oof_x,oof_b={},{},{},{}
print('  LGB...');sl=optuna.create_study(direction='minimize');sl.optimize(lambda t:objective_global(t,Xt,yt,sp,oof_storage=oof_l),n_trials=N_LGB,n_jobs=1);bp_l=oof_l.get('params',sl.best_params)
print('  CB...');sc_=optuna.create_study(direction='minimize');sc_.optimize(lambda t:objective_catboost_global(t,Xt,yt,sp,oof_storage=oof_c),n_trials=N_CB,n_jobs=1);bp_c=oof_c.get('params',sc_.best_params)
print('  XGB...');sx=optuna.create_study(direction='minimize');sx.optimize(lambda t:objective_xgboost_global(t,Xt,yt,sp,oof_storage=oof_x),n_trials=N_XG,n_jobs=1);bp_x=oof_x.get('params',sx.best_params)
print('  Base LSTM...');sb=optuna.create_study(direction='minimize');sb.optimize(lambda t:objective_base_lstm_global(t,Xt,yt,sp,device=device,oof_storage=oof_b),n_trials=N_BL,n_jobs=1);bp_b=oof_b.get('params',sb.best_params)

# Meta LSTM (Stacking)
print(f'[6/9] Stacking Meta LSTM...')
oof_df=build_oof_dataframe_sota(oof_l,oof_c,oof_x,oof_b,yt)
print(f'  OOF matrix shape: {oof_df.shape}')
meta_model,mae_meta,meta_results,bp_mt,study_mt=optimize_stacking_meta(oof_df,device,n_trials=N_MT)
if meta_model is not None:
    print(f'  Stacking Meta LSTM MAE: {mae_meta:.6f}')
    ws_meta=bp_mt.get('window_size',10)
else:
    print('  WARN: Stacking Meta LSTM no entrenado (datos insuficientes)')
    ws_meta=10

# Predictions
print(f'[7/9] Predicciones...')
pl,_=lgb_predict_test(Xt,yt,Xe,bp_l);pc,_=cb_predict_test(Xt,yt,Xe,bp_c)
px,_=train_final_xgb(Xt,yt,Xe,bp_x)
pb,_=train_final_base_lstm(Xt,yt,Xe,bp_b,device)

# Meta LSTM prediction on test (combine base model predictions)
pmt=np.full(len(ye),np.nan)
if meta_model is not None:
    # Build test prediction matrix [n_test, 4] from base model predictions (normalized scale)
    test_matrix=np.column_stack([pl,pc,px,pb]).astype(np.float32)
    meta_model.eval()
    with torch.no_grad():
        for i in range(ws_meta-1,len(ye)):
            window=test_matrix[i-ws_meta+1:i+1]
            if not np.isnan(window).any():
                x_t=torch.from_numpy(window).unsqueeze(0).to(device)
                pmt[i]=meta_model(x_t).cpu().item()

print(f'[8/9] Metricas...')
yv=ye.values;n=len(yv);idx=np.arange(n);preds={'LGB':pl,'CB':pc,'XG':px,'BL':pb,'MT':pmt}
# Log return scale
yt_log=sct.inverse_transform(yv.reshape(-1,1)).flatten()
inv=lambda p:sct.inverse_transform(np.where(np.isnan(p),0,p).reshape(-1,1)).flatten()
pl_l,pc_l,px_l=inv(pl),inv(pc),inv(px)
pb_l=np.full_like(pb,np.nan);vb=~np.isnan(pb)
if vb.any():pb_l[vb]=sct.inverse_transform(pb[vb].reshape(-1,1)).flatten()
pmt_l=np.full_like(pmt,np.nan);vmt=~np.isnan(pmt)
if vmt.any():pmt_l[vmt]=sct.inverse_transform(pmt[vmt].reshape(-1,1)).flatten()
preds_l={'LGB':pl_l,'CB':pc_l,'XG':px_l,'BL':pb_l,'MT':pmt_l}
# Price scale
cp=df['Close'].values;gi=np.arange(ts,ts+n);val=gi<len(cp);gi_v=gi[val];prev=cp[gi_v-1]
pr_r=_recon(yt_log[val],prev,int(val.sum()));pr_l=_recon(pl_l[val],prev,int(val.sum()));pr_c=_recon(pc_l[val],prev,int(val.sum()))
pr_x=_recon(px_l[val],prev,int(val.sum()))
pr_b=np.where(~np.isnan(pb_l[val]),prev*np.exp(pb_l[val]),np.nan)
pr_mt=np.where(~np.isnan(pmt_l[val]),prev*np.exp(pmt_l[val]),np.nan)
preds_p={'LGB':pr_l,'CB':pr_c,'XG':pr_x,'BL':pr_b,'MT':pr_mt}
# ===== HTML REPORT =====
print(f'[9/9] Generando report_sota.html...')
mp=[];[(lambda y2,p2,nm:mp.append({'Modelo':nm,**met(y2,p2)}))(pr_r[~np.isnan(v)],v[~np.isnan(v)],MDL[km][1]) for km,v in preds_p.items() if (~np.isnan(v)).any()];mp.sort(key=lambda x:x['MAE'])
zs,ze=max(0,int(gi_v.min())-50),min(len(cp),int(gi_v.max())+50)
from report_html import generate_html_report

# Generar report con nombre distinto
import json
def generate_sota_report(token,cp,gi_v,pr_r,preds_p,mp,MDL,zs,ze,out_dir):
    """Wrapper para generar report_sota.html en vez de report.html."""
    zoom_x=list(range(zs,ze));zoom_close=[float(v) for v in cp[zs:ze]]
    zoom_models={}
    for km,(cl,nm) in MDL.items():
        v=preds_p[km];m=~np.isnan(v)
        if m.any():zoom_models[nm]={'x':[int(x) for x in gi_v[m]],'y':[float(y) for y in v[m]],'color':cl}
    mp_c=[]
    for row in mp:
        r=dict(row)
        for km,(cl,nm) in MDL.items():
            if nm==r['Modelo']:r['Color']=cl;break
        mp_c.append(r)
    html=f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{token} - SOTA Stacking Ensemble</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#ffffff;color:#111;min-height:100vh;padding:24px}}
  .container{{max-width:1400px;margin:0 auto}}
  h1{{text-align:center;font-size:2.2rem;font-weight:700;color:#000;margin-bottom:8px}}
  .subtitle{{text-align:center;color:#666;font-size:.95rem;margin-bottom:32px}}
  .card{{background:#fafafa;border:1px solid #ddd;border-radius:16px;padding:24px;margin-bottom:28px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
  .card h2{{font-size:1.3rem;font-weight:600;margin-bottom:16px;color:#222;letter-spacing:.5px}}
  .metrics-table{{width:100%;border-collapse:separate;border-spacing:0;border-radius:12px;overflow:hidden}}
  .metrics-table thead th{{background:#f0f0f0;color:#000;padding:14px 18px;font-weight:600;text-align:left;font-size:.9rem;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #ccc}}
  .metrics-table tbody td{{padding:12px 18px;border-bottom:1px solid #eee;font-size:.95rem;font-variant-numeric:tabular-nums}}
  .metrics-table tbody tr:hover{{background:#f5f5f5}}
  .metrics-table tbody tr:last-child td{{border-bottom:none}}
  .model-badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:600;font-size:.85rem;color:#fff}}
  .best-badge{{display:inline-block;margin-left:8px;padding:2px 8px;border-radius:10px;background:#eee;color:#000;font-size:.7rem;font-weight:600;border:1px solid #ccc}}
  .metrics-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  @media(max-width:768px){{.metrics-grid{{grid-template-columns:1fr}}}}
  footer{{text-align:center;color:#999;font-size:.8rem;margin-top:40px;padding:20px}}
</style>
</head>
<body>
<div class="container">
  <h1>{token} \\u2014 SOTA Stacking Ensemble</h1>
  <p class="subtitle">CatBoost + LightGBM + XGBoost + Base LSTM \\u2192 Meta LSTM (pesos convexos din\\u00e1micos)</p>
  <div class="card"><h2>Zoom \\u2014 Zona Test</h2><div id="zoom-chart"></div></div>
  <div class="card"><h2>M\\u00e9tricas por Modelo</h2>
    <table class="metrics-table"><thead><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R\\u00b2</th></tr></thead><tbody>
"""
    best_vals={}
    for mn in ['MSE','RMSE','MAE','R2']:
        vals=[m_[mn] for m_ in mp_c]
        best_vals[mn]=max(vals) if mn=='R2' else min(vals)
    def _fmt(val,mn):
        s=f'{val:.6f}'
        return f'<strong>{s}</strong>' if val==best_vals[mn] else s
    for i,m_ in enumerate(mp_c):
        best='<span class="best-badge">BEST</span>' if i==0 else ''
        html+=f'<tr><td><span class="model-badge" style="background:{m_["Color"]}">{m_["Modelo"]}</span>{best}</td><td>{_fmt(m_["MSE"],"MSE")}</td><td>{_fmt(m_["RMSE"],"RMSE")}</td><td>{_fmt(m_["MAE"],"MAE")}</td><td>{_fmt(m_["R2"],"R2")}</td></tr>\n'
    html+="""    </tbody></table></div>
  <div class="card"><h2>Comparaci\\u00f3n de M\\u00e9tricas</h2>
    <div class="metrics-grid">
      <div id="chart-mse"></div><div id="chart-rmse"></div>
      <div id="chart-mae"></div><div id="chart-r2"></div>
    </div>
  </div>
  <footer>Generado autom\\u00e1ticamente \\u00b7 main_sota.py \\u00b7 SOTA Stacking Ensemble</footer>
</div>
<script>
"""
    html+=f"const zoomX={json.dumps(zoom_x)};\nconst zoomClose={json.dumps(zoom_close)};\n"
    html+=f"const zoomModels={json.dumps(zoom_models)};\nconst metricsData={json.dumps(mp_c)};\n"
    html+="""const dL={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#333',family:'Segoe UI,system-ui,sans-serif'},xaxis:{gridcolor:'#eee',linecolor:'#ccc'},yaxis:{gridcolor:'#eee',linecolor:'#ccc'},margin:{t:40,r:30,b:50,l:60},legend:{bgcolor:'rgba(0,0,0,0)',font:{size:11}}};
const zt=[{x:zoomX,y:zoomClose,type:'scatter',mode:'lines',name:'Close (USD)',line:{color:'#000',width:2}}];
for(const[n,d] of Object.entries(zoomModels))zt.push({x:d.x,y:d.y,type:'scatter',mode:'lines+markers',name:n,line:{color:d.color,width:1.5},marker:{size:4,color:d.color}});
Plotly.newPlot('zoom-chart',zt,{...dL,title:{text:'Precio Close + Predicciones SOTA (Zona Test)',font:{size:14,color:'#333'}},xaxis:{...dL.xaxis,title:'\\u00cdndice temporal'},yaxis:{...dL.yaxis,title:'USD'},hovermode:'x unified'},{responsive:true});
['MSE','RMSE','MAE','R2'].forEach((mn,i)=>{const ids=['chart-mse','chart-rmse','chart-mae','chart-r2'];const titles=['MSE','RMSE','MAE','R\\u00b2'];
Plotly.newPlot(ids[i],[{x:metricsData.map(m=>m.Modelo),y:metricsData.map(m=>m[mn]),type:'bar',marker:{color:metricsData.map(m=>m.Color),opacity:.85},text:metricsData.map(m=>m[mn].toFixed(4)),textposition:'outside',textfont:{color:'#333',size:11}}],{...dL,title:{text:titles[i],font:{size:14,color:'#333'}},xaxis:{...dL.xaxis,tickangle:-20},showlegend:false,margin:{t:50,r:20,b:60,l:60}},{responsive:true,displayModeBar:false});});
</script></body></html>"""
    out_html=os.path.join(out_dir,'report_sota.html')
    with open(out_html,'w',encoding='utf-8') as fh:fh.write(html)
    print(f'Listo: {out_html}')

generate_sota_report(TOKEN,cp,gi_v,pr_r,preds_p,mp,MDL,zs,ze,os.path.dirname(__file__))
