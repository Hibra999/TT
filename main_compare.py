import pandas as pd; import numpy as np; import optuna; import warnings; import os
from data.yfinance_data import download_yf; from data.ccxt_data import download_cx; from features.macroeconomics import macroeconomicos

# Modelos actuales
from model.bases_models.ligthGBM_model import objective_global, train_final_and_predict_test as lgb_predict_test
from model.bases_models.catboost_model import objective_catboost_global, train_final_and_predict_test as cb_predict_test
from model.bases_models.timexer_model import objective_timexer_global, train_final_and_predict_test as tx_predict_test
from model.bases_models.moraiMOE_model import objective_moirai_moe_global, preload_moirai_module, train_final_and_predict_test as moirai_predict_test
from model.meta_model.lstm_model import optimize_lstm_meta, get_average_weights
from preprocessing.oof_generators import build_oof_dataframe, collect_oof_predictions

# Modelos SOTA
from model.sota.stacking_ensemble import (
    objective_xgboost_global, train_final_xgb,
    objective_base_lstm_global, train_final_base_lstm,
    build_oof_dataframe_sota, optimize_stacking_meta
)

from preprocessing.walk_forward import wfrw; from features.tecnical_indicators import TA; from features.top_n import top_k
from sklearn.preprocessing import MinMaxScaler; import torch


def build_oof_dataframe_ablation(oof_lgb, oof_cb, oof_moirai, y):
    """Construye OOF para Ensamble Actual SIN TimeXer."""
    preds_lgb, idx_lgb, _ = collect_oof_predictions(oof_lgb)
    preds_cb, idx_cb, _ = collect_oof_predictions(oof_cb)
    preds_moirai, idx_moirai, _ = collect_oof_predictions(oof_moirai)
    
    df_lgb = pd.DataFrame({'idx': idx_lgb, 'lgb': preds_lgb})
    df_cb = pd.DataFrame({'idx': idx_cb, 'catboost': preds_cb})
    df_moirai = pd.DataFrame({'idx': idx_moirai, 'moirai': preds_moirai})
    
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    
    merged = df_lgb.merge(df_cb, on='idx', how='inner').merge(df_moirai, on='idx', how='inner')
    merged['target'] = merged['idx'].apply(lambda i: y_array[int(i)] if int(i) < len(y_array) else np.nan)
    merged = merged.dropna().sort_values('idx').reset_index(drop=True)
    return merged


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

@njit(cache=True)
def _met_numba(y, p):
    m_ = np.mean((y - p) ** 2); ss = np.sum((y - p) ** 2); st = np.sum((y - np.mean(y)) ** 2)
    return m_, np.sqrt(m_), np.mean(np.abs(y - p)), 1 - ss / st if st > 0 else 0.

def met(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64); mse, rmse, mae, r2 = _met_numba(y, p)
    return {'MSE': round(mse, 6), 'RMSE': round(rmse, 6), 'MAE': round(mae, 6), 'R2': round(r2, 6)}

# Diccionario unificado para reporte (Colores hex para distinguir)
MDL = {
    'LGB': ('#1f77b4', 'LightGBM (Base Compartido)'),
    'CB':  ('#2ca02c', 'CatBoost (Base Compartido)'),
    'TX':  ('#9467bd', 'TimeXer (Base Actual)'),
    'MO':  ('#ff7f0e', 'Moirai-MoE (Base Actual)'),
    'XG':  ('#8c564b', 'XGBoost (Base SOTA)'),
    'BL':  ('#e377c2', 'Base LSTM (Base SOTA)'),
    'MT':  ('#17becf', 'Meta LSTM (Ensamble Actual)'),
    'AB':  ('#e377c2', 'Ours (Ensamble Actual Sin TimeXer)'),
    'SM':  ('#d62728', 'Yu et al. [44] 2025'),
    'LSTM_EXT': ('#ff1493', 'LSTM (Externo)'),
    'GRU_EXT': ('#00ced1', 'GRU (Externo)'),
    'ARIMA_EXT': ('#ffd700', 'ARIMA (Externo)'),
    'RF_EXT': ('#32cd32', 'Random Forest (Externo)'),
    'TRANS_EXT': ('#8a2be2', 'Transformer (Externo)'),
    'XGB_META_EXT': ('#ff4500', 'Parker et al. 2025')
}

# ===== CONFIG =====
TOKEN = '^GSPC'
N_LGB, N_CB = 10, 10
N_TX, N_MO = 10, 10
N_XG, N_BL = 10, 10
N_MT, N_AB, N_SM = 10, 10, 10

from datetime import datetime
train_start = '2020-01-01'
train_end = '2025-12-31'
test_start = '2025-06-04' # Sincronizado con el inicio de predicciones externas
test_end = '2025-12-31'

START, END = train_start, test_end
# ==================

print(f'[1/11] Descargando datos...')
download_yf(['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC', 'GC=F', 'CBOE'], START, END)
download_cx(['BTC/USDT', 'ETH/USDT'], START, END)
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'tokens', f'{TOKEN.replace("/", "-")}_2020-2025.csv'))

# Cargar predicciones externas (^GSPC_all_models_predictions.csv)
csv_ext = os.path.join(os.path.dirname(__file__), '^GSPC_all_models_predictions.csv')
ext_preds_map = {}
idx_start_ext = None
if os.path.exists(csv_ext):
    print(f'[1.5/11] Alineando predicciones externas...')
    df_ext = pd.read_csv(csv_ext)
    # Coincidencia por 'Actual' (High)
    for i in range(len(df) - len(df_ext) + 1):
        if np.allclose(df['High'].iloc[i:i+len(df_ext)].values, df_ext['Actual'].values, atol=1e-2):
            idx_start_ext = i
            for col in ['LSTM', 'GRU', 'ARIMA', 'Random Forest', 'Transformer', 'XGBoost']:
                key = col.upper().replace(' ', '_') + '_EXT'
                full_raw = np.full(len(df), np.nan)
                full_raw[idx_start_ext:idx_start_ext+len(df_ext)] = df_ext[col].values
                ext_preds_map[key] = full_raw
            break

# Features
print(f'[2/11] Construyendo Features (TA + Macro)...')
df_ta = TA(df); df_ma = macroeconomicos(df['Date_final'])

# MIC
print(f'[3/11] Selección de Variables (MIC)...')
target_series = np.log(df['Close'].shift(-1) / df['Close'])

# Alinear fechas de macro con las fechas exactas del dataframe de precios
df_dates = pd.to_datetime(df['Date_final']).dt.date
df_ma_aligned = df_ma.reindex(df_dates).ffill().bfill().reset_index(drop=True)

df_ta_r = df_ta.reset_index(drop=True)
target_series_r = target_series.reset_index(drop=True)

df_combined = pd.concat([df_ta_r, df_ma_aligned], axis=1)
df_combined['target_lc'] = target_series_r
df_combined['orig_idx'] = df_combined.index

# Eliminar SOLO los NaNs resultantes de los indicadores técnicos al principio
# y el último NaN del target shift(-1). No eliminará el final completo.
df_combined = df_combined.dropna().reset_index(drop=True)

orig_idx_array = df_combined['orig_idx'].values
lc_r = df_combined['target_lc']
df_f = df_combined.drop(columns=['target_lc', 'orig_idx'])

drop = [c for c in df_f.columns if df_f[c].max() - df_f[c].min() < 1e-8]
df_f = df_f.drop(columns=drop).replace([np.inf, -np.inf], 0.0)
lc_r = lc_r.replace([np.inf, -np.inf], 0.0)

# Dividir por fecha de inicio de test (2026-01-01) en lugar de un %.
dates = pd.to_datetime(df['Date_final']).iloc[orig_idx_array].reset_index(drop=True)
mask = dates >= pd.to_datetime(test_start)
if mask.any():
    ts = mask.idxmax()
else:
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
print(f'[4/11] Split Walk-Forward...')
k = 5; sp = wfrw(yt, k=k, fh_val=30)

# Training Base Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[5/11] Entrenando Modelos Base ({device})...')
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
st_ = optuna.create_study(direction='minimize', pruner=optuna.pruners.PercentilePruner(percentile=75.0, n_warmup_steps=2))
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
print(f'[6/11] Entrenando Meta LSTM (Ensamble Actual Completo)...')
oof_df_actual = build_oof_dataframe(oof_l, oof_c, oof_t, oof_m, yt)
print(f'  OOF matrix shape: {oof_df_actual.shape}')
meta_model_actual, _, _, bp_mt, _ = optimize_lstm_meta(oof_df_actual, device, n_trials=N_MT)
ws_meta_actual = bp_mt.get('window_size', 10) if meta_model_actual is not None else 10

# Meta Ensamble Ablation (Sin TimeXer)
print(f'[7/11] Entrenando Meta LSTM Ours (Sin TimeXer)...')
oof_df_ablation = build_oof_dataframe_ablation(oof_l, oof_c, oof_m, yt)
print(f'  OOF matrix shape: {oof_df_ablation.shape}')
# Reusamos la lógica de optimización de LSTM actual ya que la estructura base no cambia (solo el # features)
meta_model_ablation, _, _, bp_ab, _ = optimize_lstm_meta(oof_df_ablation, device, n_trials=N_AB)
ws_meta_ablation = bp_ab.get('window_size', 10) if meta_model_ablation is not None else 10

# Meta Ensamble SOTA
print(f'[8/11] Entrenando Stacking Meta LSTM (Modelo Yu et al.)...')
oof_df_sota = build_oof_dataframe_sota(oof_l, oof_c, oof_x, oof_b, yt)
print(f'  OOF matrix shape: {oof_df_sota.shape}')
meta_model_sota, _, _, bp_sm, _ = optimize_stacking_meta(oof_df_sota, device, n_trials=N_SM)
ws_meta_sota = bp_sm.get('window_size', 10) if meta_model_sota is not None else 10


# Generando Predicciones Base
print(f'[9/11] Predicciones en Set de Prueba (Modelos Base)...')
pl, _ = lgb_predict_test(Xt, yt, Xe, bp_l)
pc, _ = cb_predict_test(Xt, yt, Xe, bp_c)

pt, _, _ = tx_predict_test(Xt, yt, Xe, ye, bp_t, device, seq_len=96, pred_len=1, features='MS')
if len(pt) < len(ye): tmp = np.full(len(ye), np.nan); tmp[len(ye) - len(pt):] = pt; pt = tmp

pm, _ = moirai_predict_test(yt, ye, bp_m, model_size='small', freq='D')
if len(pm) < len(ye): tmp = np.full(len(ye), np.nan); tmp[len(ye) - len(pm):] = pm; pm = tmp

px, _ = train_final_xgb(Xt, yt, Xe, bp_x)
pb, _ = train_final_base_lstm(Xt, yt, Xe, bp_b, device)


# Generando Predicciones Meta
print(f'[10/11] Predicciones Ensambles (Meta-Learners)...')
pmt_actual = np.full(len(ye), np.nan)
pmt_ablation = np.full(len(ye), np.nan)
pmt_sota = np.full(len(ye), np.nan)

# Alinear todos los metas al mismo índice de inicio de test
start_idx = max(ws_meta_actual, ws_meta_ablation, ws_meta_sota) - 1

if meta_model_actual is not None:
    # Sin _ffill: usar predicciones raw para evitar datos artificiales
    test_matrix_actual = np.column_stack([pl, pc, pt, pm]).astype(np.float32)
    meta_model_actual.eval()
    with torch.no_grad():
        for i in range(start_idx, len(ye)):
            window = test_matrix_actual[i - ws_meta_actual + 1:i + 1]
            if not np.isnan(window).any():
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                pmt_actual[i] = meta_model_actual(x_t).cpu().item()

if meta_model_ablation is not None:
    test_matrix_ablation = np.column_stack([pl, pc, pm]).astype(np.float32)
    meta_model_ablation.eval()
    with torch.no_grad():
        for i in range(start_idx, len(ye)):
            window = test_matrix_ablation[i - ws_meta_ablation + 1:i + 1]
            if not np.isnan(window).any():
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                pmt_ablation[i] = meta_model_ablation(x_t).cpu().item()

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
print(f'[11/11] Calculando Métricas Comparativas y Reporte...')
yv = ye.values
n = len(yv)
idx = np.arange(n)

yt_log = sct.inverse_transform(yv.reshape(-1, 1)).flatten()

cp = df['Close'].values
test_orig_indices = orig_idx_array[ts : ts + n]
prev = cp[test_orig_indices]
val = test_orig_indices < len(cp)

gi_v = test_orig_indices[val] + 1
prev = prev[val]

pr_r = _recon(yt_log[val], prev, int(val.sum()))

# Verificación: confirmar que métricas están en escala USD
gi_check = gi_v[gi_v < len(cp)]
if len(gi_check) > 0:
    real_close = cp[gi_check]
    recon_close = pr_r[:len(gi_check)]
    err = np.mean(np.abs(real_close - recon_close))
    print(f'  [VERIFY] Reconstrucción USD - Error medio vs Close real: {err:.4f}')
    print(f'  [VERIFY] Rango precio real: [{np.nanmin(pr_r):.2f}, {np.nanmax(pr_r):.2f}] USD')

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
    'AB':  safe_inv_recon(pmt_ablation),
    'SM':  safe_inv_recon(pmt_sota),
    'LSTM_EXT': ext_preds_map.get('LSTM_EXT', np.full(len(df), np.nan))[test_orig_indices],
    'GRU_EXT': ext_preds_map.get('GRU_EXT', np.full(len(df), np.nan))[test_orig_indices],
    'ARIMA_EXT': ext_preds_map.get('ARIMA_EXT', np.full(len(df), np.nan))[test_orig_indices],
    'RF_EXT': ext_preds_map.get('RANDOM_FOREST_EXT', np.full(len(df), np.nan))[test_orig_indices],
    'TRANS_EXT': ext_preds_map.get('TRANSFORMER_EXT', np.full(len(df), np.nan))[test_orig_indices],
    'XGB_META_EXT': ext_preds_map.get('XGBOOST_EXT', np.full(len(df), np.nan))[test_orig_indices]
}

mp = []
for km, v in preds_p.items():
    if (~np.isnan(v)).any():
        v_valid = v[~np.isnan(v)]
        y_valid = pr_r[~np.isnan(v)]
        if len(v_valid) > 0 and len(y_valid) > 0:
            mp.append({'Modelo': MDL[km][1], **met(y_valid, v_valid)})

# Ordenar el reporte por MAE descendente para mejor visualización
mp.sort(key=lambda x: x['MAE'])

# Restringir reporte al periodo de predicciones externas si existe, sino ultimo 10%
if idx_start_ext is not None:
    zs = idx_start_ext
else:
    zs = int(len(cp) * 0.9)
ze = len(cp)

# Predicciones raw de meta-learners para métricas sobre LogReturn_MinMax
p_usd_parker = preds_p['XGB_META_EXT']
pmt_parker = np.full(len(ye), np.nan)
v_mask = ~np.isnan(p_usd_parker)
if v_mask.any():
    # log(p_usd / prev)
    p_log = np.log(p_usd_parker[v_mask] / prev[v_mask])
    pmt_parker[v_mask] = sct.transform(p_log.reshape(-1, 1)).flatten()

meta_raw_preds = {'MT': pmt_actual, 'AB': pmt_ablation, 'SM': pmt_sota, 'XGB_META_EXT': pmt_parker}
# Predicciones raw de modelos base para gráficos LogReturn_MinMax
base_raw_preds = {'LGB': pl, 'CB': pc, 'TX': pt, 'MO': pm, 'XG': px, 'BL': pb}

import json
def generate_compare_report(token, cp, gi_v, pr_r, preds_p, mp, MDL, zs, ze, out_dir, ye_vals=None, meta_raw_preds=None, base_raw_preds=None):
    """Genera report_compare.html aislador para los Ensambles"""
    zoom_x = list(range(zs, ze))
    zoom_close = [float(v) for v in cp[zs:ze]]
    
    zoom_models = {}
    for km, (cl, nm) in MDL.items():
        v = preds_p[km]
        m = ~np.isnan(v)
        if m.any():
            zoom_models[km] = {'name': nm, 'x': [int(x) for x in gi_v[m]], 'y': [float(y) for y in v[m]], 'color': cl}
    
    mp_metas = []
    # Métricas USD de los Meta Learners (MT, AB, SM) para interpretabilidad
    for row in mp:
        r = dict(row)
        for km, (cl, nm) in MDL.items():
            if nm == r['Modelo']:
                r['Color'] = cl
                if km in ['MT', 'AB', 'SM', 'XGB_META_EXT']:
                    mp_metas.append(r)
                break
    mp_metas.sort(key=lambda x: x['MAE'])
        
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
  .charts-grid{{display:grid;grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));gap:20px}}
  @media(max-width:1024px){{.charts-grid{{grid-template-columns:1fr}}}}
  @media(max-width:768px){{.metrics-grid{{grid-template-columns:1fr}}}}
  footer{{text-align:center;color:#999;font-size:.8rem;margin-top:40px;padding:20px}}
</style>
</head>
<body>
<div class="container">
  <h1>{token} - Comparativa: Ensamble Actual vs Ours vs Yu et al.</h1>
  <p class="subtitle">Predicciones Finales y Modelos Base (Alineados temporalmente)</p>
  
  <div class="charts-grid">
    <div class="card">
      <h2>Ensamble Actual</h2>
      <div id="zoom-chart-actual"></div>
    </div>
    <div class="card">
      <h2>Ours (Sin TimeXer)</h2>
      <div id="zoom-chart-ablation"></div>
    </div>
    <div class="card">
      <h2>Yu et al. [44] 2025</h2>
      <div id="zoom-chart-sota"></div>
    </div>
    <div class="card">
      <h2>Parker et al. 2025</h2>
      <div id="zoom-chart-parker"></div>
    </div>
  </div>

  <div class="card"><h2>Metricas Generales de los Meta Learners</h2>
    <table class="metrics-table"><thead><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R2</th></tr></thead><tbody>
"""
    best_vals = {}
    if mp_metas:
        for mn in ['MSE', 'RMSE', 'MAE', 'R2']:
            vals = [m_[mn] for m_ in mp_metas]
            if vals: best_vals[mn] = max(vals) if mn == 'R2' else min(vals)
        
    def _fmt(val, mn):
        if mn not in best_vals: return f'{val:.6f}'
        s = f'{val:.6f}'
        return s # Removed conditional highlighting
        
    for i, m_ in enumerate(mp_metas):
        html += f'<tr><td><span class="model-badge" style="background:{m_["Color"]}">{m_["Modelo"]}</span></td><td>{_fmt(m_["MSE"], "MSE")}</td><td>{_fmt(m_["RMSE"], "RMSE")}</td><td>{_fmt(m_["MAE"], "MAE")}</td><td>{_fmt(m_["R2"], "R2")}</td></tr>\n'
        
    html += """    </tbody></table></div>
  <div class="card"><h2>Prediccion sobre LogReturn_MinMax (Variable Objetivo)</h2>
    <div class="charts-grid">
      <div id="lr-actual"></div>
      <div id="lr-ablation"></div>
      <div id="lr-sota"></div>
      <div id="lr-parker"></div>
    </div>
  </div>
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
    # Datos de series temporales LogReturn_MinMax para cada meta-learner y modelos base
    lr_data = {}
    if ye_vals is not None and meta_raw_preds is not None:
        lr_real = [float(v) for v in ye_vals]
        lr_idx = list(range(len(ye_vals)))
        for km in ['MT', 'AB', 'SM', 'XGB_META_EXT']:
            p_raw = meta_raw_preds.get(km)
            if p_raw is not None:
                m_valid = ~np.isnan(p_raw)
                if m_valid.any():
                    lr_data[km] = {
                        'idx': [int(i) for i in np.where(m_valid)[0]],
                        'y': [float(p) for p in p_raw[m_valid]],
                        'name': MDL[km][1],
                        'color': MDL[km][0]
                    }
        # Agregar predicciones base (raw LogReturn_MinMax) para cada modelo
        if base_raw_preds is not None:
            for km in ['LGB', 'CB', 'TX', 'MO', 'XG', 'BL']:
                p_raw = base_raw_preds.get(km)
                if p_raw is not None:
                    m_valid = ~np.isnan(p_raw)
                    if m_valid.any():
                        lr_data[km] = {
                            'idx': [int(i) for i in np.where(m_valid)[0]],
                            'y': [float(p) for p in p_raw[m_valid]],
                            'name': MDL[km][1],
                            'color': MDL[km][0]
                        }
        lr_data['_real'] = {'idx': lr_idx, 'y': lr_real}

    html += f"const zoomX={json.dumps(zoom_x)};\nconst zoomClose={json.dumps(zoom_close)};\n"
    html += f"const zoomModels={json.dumps(zoom_models)};\n"
    html += f"const metricsData={json.dumps(mp_metas)};\n"
    html += f"const lrData={json.dumps(lr_data)};\n"
    
    html += """const dL={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#333',family:'Segoe UI,system-ui,sans-serif'},xaxis:{gridcolor:'#eee',linecolor:'#ccc'},yaxis:{gridcolor:'#eee',linecolor:'#ccc'},margin:{t:40,r:30,b:50,l:60},legend:{bgcolor:'rgba(0,0,0,0)',font:{size:11},orientation:'h',y:-0.2}};

// Wrapper function para graficar
function drawChart(divId, titleTxt, keys, metaKey) {
    const data = [{x:zoomX, y:zoomClose, type:'scatter', mode:'lines', name:'Close (USD)', line:{color:'#000', width:2}}];
    keys.forEach(k => {
        if(zoomModels[k] && zoomModels[k].x.length > 0) {
            let isMeta = (k === metaKey);
            data.push({
                x: zoomModels[k].x, 
                y: zoomModels[k].y, 
                type: 'scatter', 
                mode: isMeta ? 'lines+markers' : 'lines', 
                name: zoomModels[k].name, 
                line: {color: zoomModels[k].color, width: isMeta ? 2.5 : 1.2, dash: isMeta ? 'solid' : 'dot'}, 
                marker: {size: isMeta ? 4 : 2, color: zoomModels[k].color}
            });
        }
    });
    Plotly.newPlot(divId, data, {...dL, title: {text: titleTxt, font: {size: 14, color: '#333'}}, xaxis: {...dL.xaxis, title: 'Indice'}, yaxis: {...dL.yaxis, title: 'USD'}, hovermode: 'x unified'}, {responsive: true});
}

drawChart('zoom-chart-actual', 'Precio Close vs Actual', ['LGB','CB','TX','MO','MT', 'LSTM_EXT', 'GRU_EXT'], 'MT');
drawChart('zoom-chart-ablation', 'Precio Close vs Ours', ['LGB','CB','MO','AB', 'ARIMA_EXT', 'RF_EXT'], 'AB');
drawChart('zoom-chart-sota', 'Precio Close vs Yu et al. 2025', ['LGB','CB','XG','BL','SM', 'TRANS_EXT'], 'SM');
drawChart('zoom-chart-parker', 'Precio Close vs Parker et al. 2025', ['LGB','CB','XGB_META_EXT'], 'XGB_META_EXT');

function drawLR(divId, titleTxt, metaKey, baseKeys) {
    if(!lrData['_real']) return;
    const real = lrData['_real'];
    const data = [
        {x: real.idx, y: real.y, type:'scatter', mode:'lines', name:'Real (LogReturn_MinMax)', line:{color:'#000', width:2}}
    ];
    // Agregar modelos base como lineas delgadas punteadas
    baseKeys.forEach(bk => {
        if(lrData[bk]) {
            data.push({x: lrData[bk].idx, y: lrData[bk].y, type:'scatter', mode:'lines', name: lrData[bk].name, line:{color: lrData[bk].color, width:1.2, dash:'dot'}});
        }
    });
    // Agregar meta-learner como linea gruesa
    if(lrData[metaKey]) {
        const pred = lrData[metaKey];
        data.push({x: pred.idx, y: pred.y, type:'scatter', mode:'lines+markers', name: pred.name, line:{color: pred.color, width:2.5}, marker:{size:3, color: pred.color}});
    }
    Plotly.newPlot(divId, data, {...dL, title: {text: titleTxt, font: {size: 14, color: '#333'}}, xaxis: {...dL.xaxis, title: 'Indice Test'}, yaxis: {...dL.yaxis, title: 'LogReturn_MinMax'}, hovermode: 'x unified'}, {responsive: true});
}

drawLR('lr-actual', 'LogReturn_MinMax: Ensamble Actual', 'MT', ['LGB','CB','TX','MO']);
drawLR('lr-ablation', 'LogReturn_MinMax: Ours (Sin TimeXer)', 'AB', ['LGB','CB','MO']);
drawLR('lr-sota', 'LogReturn_MinMax: Yu et al. 2025', 'SM', ['LGB','CB','XG','BL']);
drawLR('lr-parker', 'LogReturn_MinMax: Parker et al. 2025', 'XGB_META_EXT', ['LGB','CB']);

['MSE','RMSE','MAE','R2'].forEach((mn,i)=>{const ids=['chart-mse','chart-rmse','chart-mae','chart-r2'];const titles=['MSE','RMSE','MAE','R2'];
Plotly.newPlot(ids[i],[{x:metricsData.map(m=>m.Modelo),y:metricsData.map(m=>m[mn]),type:'bar',marker:{color:metricsData.map(m=>m.Color),opacity:.85},text:metricsData.map(m=>m[mn].toFixed(4)),textposition:'outside',textfont:{color:'#333',size:11}}],{...dL,title:{text:titles[i],font:{size:14,color:'#333'}},xaxis:{...dL.xaxis,tickangle:45},showlegend:false,margin:{t:50,r:20,b:150,l:60}},{responsive:true,displayModeBar:false});});
</script></body></html>"""
    out_html = os.path.join(out_dir, 'report_compare.html')
    with open(out_html, 'w', encoding='utf-8') as fh: fh.write(html)
    print(f'Listo Comparativa Limpia: {out_html}')

generate_compare_report(TOKEN, cp, gi_v, pr_r, preds_p, mp, MDL, zs, ze, os.path.dirname(__file__), ye_vals=yv, meta_raw_preds=meta_raw_preds, base_raw_preds=base_raw_preds)
