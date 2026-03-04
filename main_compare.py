import pandas as pd; import numpy as np; import optuna; import warnings; import os
from data.yfinance_data import download_yf; from data.ccxt_data import download_cx; from features.macroeconomics import macroeconomicos

# Modelos actuales
from model.bases_models.ligthGBM_model import objective_global, train_final_and_predict_test as lgb_predict_test
from model.bases_models.catboost_model import objective_catboost_global, train_final_and_predict_test as cb_predict_test
from model.bases_models.timexer_model import objective_timexer_global, train_final_and_predict_test as tx_predict_test
from model.bases_models.moraiMOE_model import objective_moirai_moe_global, preload_moirai_module, train_final_and_predict_test as moirai_predict_test
from model.meta_model.lstm_model import optimize_lstm_meta, get_average_weights
from preprocessing.oof_generators import build_oof_dataframe

# Modelos SOTA
from model.sota.stacking_ensemble import (
    objective_xgboost_global, train_final_xgb,
    objective_base_lstm_global, train_final_base_lstm,
    build_oof_dataframe_sota, optimize_stacking_meta
)

from preprocessing.walk_forward import wfrw; from features.tecnical_indicators import TA; from features.top_n import top_k
from sklearn.preprocessing import MinMaxScaler; import torch

warnings.filterwarnings("ignore")
try:
    from numba import njit, prange; HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _recon(yl, cp, n):
        o = np.empty(n, dtype=np.float64)
        for i in prange(n): o[i] = cp[i] * np.exp(yl[i])
        return o
else:
    def _recon(yl, cp, n): return cp * np.exp(yl)

def met(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    mse = np.mean((y - p) ** 2)
    mae = np.mean(np.abs(y - p))
    ss = np.sum((y - p) ** 2)
    st = np.sum((y - np.mean(y)) ** 2)
    return {'MSE': round(mse, 6), 'RMSE': round(np.sqrt(mse), 6), 'MAE': round(mae, 6), 'R2': round(1 - ss / st if st > 0 else 0., 6)}

# Diccionario unificado para reporte (Colores hex para distinguir)
MDL = {
    'LGB': ('#1f77b4', 'LightGBM (Base Compartido)'),
    'CB':  ('#2ca02c', 'CatBoost (Base Compartido)'),
    'TX':  ('#9467bd', 'TimeXer (Base Actual)'),
    'MO':  ('#ff7f0e', 'Moirai-MoE (Base Actual)'),
    'XG':  ('#8c564b', 'XGBoost (Base SOTA)'),
    'BL':  ('#e377c2', 'Base LSTM (Base SOTA)'),
    'MT':  ('#17becf', 'Meta LSTM (Ensamble Actual)'),
    'SM':  ('#d62728', 'Yu et al. [44] 2025')
}

# ===== CONFIG =====
TOKEN = 'ETH/USDT'
# Incrementado a 5 trials para que pueda generar OOF suficientes o al menos 3.
N_LGB, N_CB = 3, 3
N_TX, N_MO = 3, 3
N_XG, N_BL = 3, 3
N_MT, N_SM = 3, 3
START, END = '2020-01-01', '2025-12-31'
# ==================

print(f'[1/10] Descargando datos...')
download_yf(['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC', 'GC=F', 'CBOE'], START, END)
download_cx(['BTC/USDT', 'ETH/USDT'], START, END)
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'tokens', f'{TOKEN.replace("/", "-")}_2020-2025.csv'))
lc = np.log(df['Close'] / df['Close'].shift(1)).dropna()
lc_n = (lc - lc.min()) / (lc.max() - lc.min())

# Features
print(f'[2/10] Construyendo Features (TA + Macro)...')
df_ta = TA(df); df_ma = macroeconomicos(df['Date_final'])

# MIC
print(f'[3/10] Selección de Variables (MIC)...')
df_ta_r = df_ta.reset_index(drop=True); df_ma_r = df_ma.reset_index(drop=True); lc_r = lc.reset_index(drop=True)
df_f = pd.concat([df_ta_r, df_ma_r], axis=1).iloc[1:]
ml = min(len(df_f), len(lc_r))
df_f = df_f.iloc[:ml].reset_index(drop=True)
lc_r = lc_r.iloc[:ml].reset_index(drop=True)

drop = [c for c in df_f.columns if df_f[c].max() - df_f[c].min() < 1e-8]
df_f = df_f.drop(columns=drop).replace([np.inf, -np.inf], 0.0)
lc_r = lc_r.replace([np.inf, -np.inf], 0.0)

ts = int(len(df_f) * .9)
Xtr, Xte = df_f.iloc[:ts].copy(), df_f.iloc[ts:].copy()
ytr, yte = lc_r.iloc[:ts].copy(), lc_r.iloc[ts:].copy()

sf = MinMaxScaler()
Xtr_s = pd.DataFrame(sf.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
Xte_s = pd.DataFrame(sf.transform(Xte), columns=Xte.columns, index=Xte.index)

sct = MinMaxScaler()
ytr_s = pd.Series(sct.fit_transform(ytr.values.reshape(-1, 1)).flatten(), index=ytr.index, name='lc')
yte_s = pd.Series(sct.transform(yte.values.reshape(-1, 1)).flatten(), index=yte.index, name='lc')

feats, mic_v = top_k(Xtr_s, ytr_s, 15)
Xt, Xe = Xtr_s[feats].reset_index(drop=True), Xte_s[feats].reset_index(drop=True)
yt, ye = ytr_s.reset_index(drop=True), yte_s.reset_index(drop=True)

# Walk Forward
print(f'[4/10] Split Walk-Forward...')
k = 5; sp = wfrw(yt, k=k, fh_val=30)

# Training Base Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[5/10] Entrenando Modelos Base ({device})...')
oof_l, oof_c = {}, {}
print('  > LightGBM (Compartido)...')
sl = optuna.create_study(direction='minimize')
sl.optimize(lambda t: objective_global(t, Xt, yt, sp, oof_storage=oof_l), n_trials=N_LGB, n_jobs=1)
bp_l = oof_l.get('params', sl.best_params)

print('  > CatBoost (Compartido)...')
sc_ = optuna.create_study(direction='minimize')
sc_.optimize(lambda t: objective_catboost_global(t, Xt, yt, sp, oof_storage=oof_c), n_trials=N_CB, n_jobs=1)
bp_c = oof_c.get('params', sc_.best_params)

# Modelos solo del ensamble actual
oof_t, oof_m = {}, {}
print('  --- Base Models Ensamble Actual ---')
print('  > TimeXer...')
st_ = optuna.create_study(direction='minimize')
st_.optimize(lambda t: objective_timexer_global(t, Xt, yt, sp, device=device, seq_len=96, pred_len=30, features='MS', oof_storage=oof_t), n_trials=N_TX, n_jobs=1)
bp_t = st_.best_params

print('  > Moirai-MoE...')
preload_moirai_module(model_size='small')
sm = optuna.create_study(direction='minimize')
sm.optimize(lambda t: objective_moirai_moe_global(t, Xt, yt, sp, device=device, pred_len=30, model_size='small', freq='D', use_full_train=True, oof_storage=oof_m), n_trials=N_MO, n_jobs=1)
bp_m = sm.best_params

# Modelos solo del SOTA
oof_x, oof_b = {}, {}
print('  --- Base Models Ensamble SOTA ---')
print('  > XGBoost...')
sx = optuna.create_study(direction='minimize')
sx.optimize(lambda t: objective_xgboost_global(t, Xt, yt, sp, oof_storage=oof_x), n_trials=N_XG, n_jobs=1)
bp_x = oof_x.get('params', sx.best_params)

print('  > Base LSTM...')
sb = optuna.create_study(direction='minimize')
sb.optimize(lambda t: objective_base_lstm_global(t, Xt, yt, sp, device=device, oof_storage=oof_b), n_trials=N_BL, n_jobs=1)
bp_b = oof_b.get('params', sb.best_params)


# Meta Ensamble Actual
print(f'[6/10] Entrenando Meta LSTM (Ensamble Actual)...')
oof_df_actual = build_oof_dataframe(oof_l, oof_c, oof_t, oof_m, yt)
print(f'  OOF matrix shape: {oof_df_actual.shape}')
meta_model_actual, mae_meta_actual, _, bp_mt, _ = optimize_lstm_meta(oof_df_actual, device, n_trials=N_MT)
ws_meta_actual = bp_mt.get('window_size', 10) if meta_model_actual is not None else 10

# Meta Ensamble SOTA
print(f'[7/10] Entrenando Stacking Meta LSTM (Modelo SOTA)...')
oof_df_sota = build_oof_dataframe_sota(oof_l, oof_c, oof_x, oof_b, yt)
print(f'  OOF matrix shape: {oof_df_sota.shape}')
meta_model_sota, mae_meta_sota, _, bp_sm, _ = optimize_stacking_meta(oof_df_sota, device, n_trials=N_SM)
ws_meta_sota = bp_sm.get('window_size', 10) if meta_model_sota is not None else 10


# Generando Predicciones Base
print(f'[8/10] Predicciones en Set de Prueba (Modelos Base)...')
pl, _ = lgb_predict_test(Xt, yt, Xe, bp_l)
pc, _ = cb_predict_test(Xt, yt, Xe, bp_c)

pt, _, _ = tx_predict_test(Xt, yt, Xe, ye, bp_t, device, seq_len=96, pred_len=1, features='MS')
if len(pt) < len(ye): tmp = np.full(len(ye), np.nan); tmp[len(ye) - len(pt):] = pt; pt = tmp

pm, _ = moirai_predict_test(yt, ye, bp_m, model_size='small', freq='D')
if len(pm) < len(ye): tmp = np.full(len(ye), np.nan); tmp[len(ye) - len(pm):] = pm; pm = tmp

px, _ = train_final_xgb(Xt, yt, Xe, bp_x)
pb, _ = train_final_base_lstm(Xt, yt, Xe, bp_b, device)


# Generando Predicciones Meta
print(f'[9/10] Predicciones Ensambles (Meta-Learners)...')
pmt_actual = np.full(len(ye), np.nan)
pmt_sota = np.full(len(ye), np.nan)

# Alinear ambos metas al mismo índice de inicio de test para que sean 100% comparables
start_idx = max(ws_meta_actual, ws_meta_sota) - 1

if meta_model_actual is not None:
    test_matrix_actual = np.column_stack([pl, pc, pt, pm]).astype(np.float32)
    meta_model_actual.eval()
    with torch.no_grad():
        for i in range(start_idx, len(ye)):
            window = test_matrix_actual[i - ws_meta_actual + 1:i + 1]
            if not np.isnan(window).any():
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                pmt_actual[i] = meta_model_actual(x_t).cpu().item()

if meta_model_sota is not None:
    test_matrix_sota = np.column_stack([pl, pc, px, pb]).astype(np.float32)
    meta_model_sota.eval()
    with torch.no_grad():
        for i in range(start_idx, len(ye)):
            window = test_matrix_sota[i - ws_meta_sota + 1:i + 1]
            if not np.isnan(window).any():
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                pmt_sota[i] = meta_model_sota(x_t).cpu().item()

# Truncar las predicciones base para que inicien exactamente al mismo tiempo que los Meta Learners
for arr in [pl, pc, pt, pm, px, pb]:
    if start_idx < len(arr):
        arr[:start_idx] = np.nan


# Métricas y reconstrucciones
print(f'[10/10] Calculando Métricas Comparativas y Reporte...')
yv = ye.values
n = len(yv)
idx = np.arange(n)

yt_log = sct.inverse_transform(yv.reshape(-1, 1)).flatten()

cp = df['Close'].values
gi = np.arange(ts, ts + n)
val = gi < len(cp)
gi_v = gi[val]
prev = cp[gi_v - 1]

pr_r = _recon(yt_log[val], prev, int(val.sum()))

def safe_inv_recon(p_raw):
    p_log = np.full_like(p_raw, np.nan)
    v_mask = ~np.isnan(p_raw)
    if v_mask.any():
        p_log[v_mask] = sct.inverse_transform(p_raw[v_mask].reshape(-1, 1)).flatten()
    return np.where(v_mask[val], prev * np.exp(p_log[val]), np.nan)

preds_p = {
    'LGB': safe_inv_recon(pl),
    'CB':  safe_inv_recon(pc),
    'TX':  safe_inv_recon(pt),
    'MO':  safe_inv_recon(pm),
    'XG':  safe_inv_recon(px),
    'BL':  safe_inv_recon(pb),
    'MT':  safe_inv_recon(pmt_actual),
    'SM':  safe_inv_recon(pmt_sota)
}

mp = []
for km, v in preds_p.items():
    if (~np.isnan(v)).any():
        v_valid = v[~np.isnan(v)]
        y_valid = pr_r[~np.isnan(v)]
        mp.append({'Modelo': MDL[km][1], **met(y_valid, v_valid)})

# Ordenar el reporte por MAE descendente para mejor visualización
mp.sort(key=lambda x: x['MAE'])

zs, ze = max(0, int(gi_v.min()) - 50), min(len(cp), int(gi_v.max()) + 50)

import json
def generate_compare_report(token, cp, gi_v, pr_r, preds_p, mp, MDL, zs, ze, out_dir):
    """Genera report_compare.html aislador para los Ensambles"""
    zoom_x = list(range(zs, ze))
    zoom_close = [float(v) for v in cp[zs:ze]]
    
    zoom_models = {}
    for km, (cl, nm) in MDL.items():
        v = preds_p[km]
        m = ~np.isnan(v)
        if m.any():
            zoom_models[km] = {'name': nm, 'x': [int(x) for x in gi_v[m]], 'y': [float(y) for y in v[m]], 'color': cl}
    
    mp_c = []
    mp_metas = []
    for row in mp:
        r = dict(row)
        for km, (cl, nm) in MDL.items():
            if nm == r['Modelo']: 
                r['Color'] = cl
                if km in ['MT', 'SM']:
                    mp_metas.append(r)
                break
        mp_c.append(r)
        
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{token} - Comparativa de Ensambles</title>
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
  .charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  @media(max-width:1024px){{.charts-grid{{grid-template-columns:1fr}}}}
  @media(max-width:768px){{.metrics-grid{{grid-template-columns:1fr}}}}
  footer{{text-align:center;color:#999;font-size:.8rem;margin-top:40px;padding:20px}}
</style>
</head>
<body>
<div class="container">
  <h1>{token} - Comparativa: Ensamble Actual vs Nuevo SOTA</h1>
  <p class="subtitle">Predicciones Finales y Modelos Base (Alineados temporalmente)</p>
  
  <div class="charts-grid">
    <div class="card">
      <h2>Ensamble Actual (Meta LSTM + Modelos Base Actuales)</h2>
      <div id="zoom-chart-actual"></div>
    </div>
    <div class="card">
      <h2>Ensamble Nuevo ({MDL['SM'][1]} + Modelos Base SOTA)</h2>
      <div id="zoom-chart-sota"></div>
    </div>
  </div>

  <div class="card"><h2>Metricas Generales de los Meta Learners</h2>
    <table class="metrics-table"><thead><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R2</th></tr></thead><tbody>
"""
    best_vals = {}
    for mn in ['MSE', 'RMSE', 'MAE', 'R2']:
        vals = [m_[mn] for m_ in mp_metas]
        if vals: best_vals[mn] = max(vals) if mn == 'R2' else min(vals)
        
    def _fmt(val, mn):
        if mn not in best_vals: return f'{val:.6f}'
        s = f'{val:.6f}'
        return f'<strong>{s}</strong>' if val == best_vals[mn] else s
        
    for i, m_ in enumerate(mp_metas):
        best = '<span class="best-badge">BEST</span>' if i == 0 else ''
        html += f'<tr><td><span class="model-badge" style="background:{m_["Color"]}">{m_["Modelo"]}</span>{best}</td><td>{_fmt(m_["MSE"], "MSE")}</td><td>{_fmt(m_["RMSE"], "RMSE")}</td><td>{_fmt(m_["MAE"], "MAE")}</td><td>{_fmt(m_["R2"], "R2")}</td></tr>\n'
        
    html += """    </tbody></table></div>
  <div class="card"><h2>Comparacion Visual de Metricas</h2>
    <div class="metrics-grid">
      <div id="chart-mse"></div><div id="chart-rmse"></div>
      <div id="chart-mae"></div><div id="chart-r2"></div>
    </div>
  </div>
  <footer>Generado automaticamente por main_compare.py</footer>
</div>
<script>
"""
    html += f"const zoomX={json.dumps(zoom_x)};\nconst zoomClose={json.dumps(zoom_close)};\n"
    html += f"const zoomModels={json.dumps(zoom_models)};\n"
    html += f"const metricsData={json.dumps(mp_metas)};\n"
    
    html += """const dL={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#333',family:'Segoe UI,system-ui,sans-serif'},xaxis:{gridcolor:'#eee',linecolor:'#ccc'},yaxis:{gridcolor:'#eee',linecolor:'#ccc'},margin:{t:40,r:30,b:50,l:60},legend:{bgcolor:'rgba(0,0,0,0)',font:{size:11}}};

// Chart Actual
const actClose=[{x:zoomX,y:zoomClose,type:'scatter',mode:'lines',name:'Close (USD)',line:{color:'#000',width:2}}];
['LGB','CB','TX','MO','MT'].forEach(k => {
    if(zoomModels[k] && zoomModels[k].x.length>0) {
        let isMeta = (k==='MT');
        actClose.push({x:zoomModels[k].x, y:zoomModels[k].y, type:'scatter', mode:isMeta?'lines+markers':'lines', name:zoomModels[k].name, line:{color:zoomModels[k].color, width:isMeta?2.5:1.2, dash:isMeta?'solid':'dot'}, marker:{size:isMeta?5:3, color:zoomModels[k].color}});
    }
});
Plotly.newPlot('zoom-chart-actual',actClose,{...dL,title:{text:'Precio Close vs Base Models + Ensamble Actual',font:{size:14,color:'#333'}},xaxis:{...dL.xaxis,title:'Indice temporal'},yaxis:{...dL.yaxis,title:'USD'},hovermode:'x unified'},{responsive:true});

// Chart SOTA
const sotaClose=[{x:zoomX,y:zoomClose,type:'scatter',mode:'lines',name:'Close (USD)',line:{color:'#000',width:2}}];
['LGB','CB','XG','BL','SM'].forEach(k => {
    if(zoomModels[k] && zoomModels[k].x.length>0) {
        let isMeta = (k==='SM');
        sotaClose.push({x:zoomModels[k].x, y:zoomModels[k].y, type:'scatter', mode:isMeta?'lines+markers':'lines', name:zoomModels[k].name, line:{color:zoomModels[k].color, width:isMeta?2.5:1.2, dash:isMeta?'solid':'dot'}, marker:{size:isMeta?5:3, color:zoomModels[k].color}});
    }
});
Plotly.newPlot('zoom-chart-sota',sotaClose,{...dL,title:{text:'Precio Close vs Base Models + Yu et al. [44] 2025',font:{size:14,color:'#333'}},xaxis:{...dL.xaxis,title:'Indice temporal'},yaxis:{...dL.yaxis,title:'USD'},hovermode:'x unified'},{responsive:true});

['MSE','RMSE','MAE','R2'].forEach((mn,i)=>{const ids=['chart-mse','chart-rmse','chart-mae','chart-r2'];const titles=['MSE','RMSE','MAE','R2'];
Plotly.newPlot(ids[i],[{x:metricsData.map(m=>m.Modelo),y:metricsData.map(m=>m[mn]),type:'bar',marker:{color:metricsData.map(m=>m.Color),opacity:.85},text:metricsData.map(m=>m[mn].toFixed(4)),textposition:'outside',textfont:{color:'#333',size:11}}],{...dL,title:{text:titles[i],font:{size:14,color:'#333'}},xaxis:{...dL.xaxis,tickangle:0},showlegend:false,margin:{t:50,r:20,b:60,l:60}},{responsive:true,displayModeBar:false});});
</script></body></html>"""
    out_html = os.path.join(out_dir, 'report_compare.html')
    with open(out_html, 'w', encoding='utf-8') as fh: fh.write(html)
    print(f'Listo Comparativa Limpia: {out_html}')

generate_compare_report(TOKEN, cp, gi_v, pr_r, preds_p, mp, MDL, zs, ze, os.path.dirname(__file__))
