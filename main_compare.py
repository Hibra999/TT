import pandas as pd; import numpy as np; import optuna; import warnings; import os; import logging; from itertools import combinations
from data.yfinance_data import download_yf; from data.ccxt_data import download_cx; from features.macroeconomics import macroeconomicos

# Modelos actuales
from model.bases_models.ligthGBM_model import objective_global, train_final_and_predict_test as lgb_predict_test
from model.bases_models.catboost_model import objective_catboost_global, train_final_and_predict_test as cb_predict_test
from model.bases_models.timexer_model import objective_timexer_global, train_final_and_predict_test as tx_predict_test
from model.bases_models.moraiMOE_model import objective_moirai_moe_global, preload_moirai_module, train_final_and_predict_test as moirai_predict_test
from model.meta_model.lstm_model import optimize_lstm_meta, get_average_weights
from preprocessing.oof_generators import build_oof_dataframe, collect_oof_predictions
from statsmodels.tsa.stattools import adfuller; from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS; from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import t as t_dist; from bs4 import BeautifulSoup

# Modelos SOTA
from model.sota.stacking_ensemble import (
    objective_base_lstm_global, train_final_base_lstm,
    build_oof_dataframe_sota, optimize_stacking_meta
)

from preprocessing.walk_forward import wfrw; from features.tecnical_indicators import TA; from features.top_n import top_k
from sklearn.preprocessing import MinMaxScaler; import torch
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


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

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Porcentaje de observaciones donde el signo del cambio predicho
    coincide con el signo del cambio real.
    Requiere al menos 2 observaciones.
    """
    actual_dir    = np.sign(np.diff(y_true))
    predicted_dir = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == predicted_dir) * 100)

# Diccionario unificado para reporte (Colores hex para distinguir)
MDL = {
    'LGB': ('#1f77b4', 'LightGBM (Base Compartido)'),
    'CB':  ('#2ca02c', 'CatBoost (Base Compartido)'),
    'TX':  ('#9467bd', 'TimeXer (Base Actual)'),
    'MO':  ('#ff7f0e', 'Moirai-MoE (Base Actual)'),
    'BL':  ('#e377c2', 'Base LSTM (Base SOTA)'),
    'MT':  ('#17becf', 'Ensamble Actual'),
    'SA':  ('#aec7e8', 'Simple Average (Ensamble Actual)'),
    'WA':  ('#ffbb78', 'Weighted Average (Ensamble Actual)'),
    'RD':  ('#98df8a', 'Ridge (Ensamble Actual)'),
    'LS':  ('#ff9896', 'Lasso (Ensamble Actual)'),
    'EN':  ('#c5b0d5', 'Elastic Net (Ensamble Actual)'),
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
# TOKEN = 'BTC/USDT'
N_LGB, N_CB = 10, 10
N_TX, N_MO = 10, 10
N_BL = 10 
N_MT, N_AB, N_SM = 10 , 10, 10

from datetime import datetime
train_start = '2015-01-01'
train_end = '2025-12-31'
if TOKEN == 'BTC/USDT' or TOKEN == 'ETH/USDT':
    test_start = '2024-10-19' # Sincronizado con el inicio de predicciones externas
    test_end = '2025-12-31'
else:
    test_start = '2024-10-16' # Sincronizado con el inicio de predicciones externas
    test_end = '2025-12-31'

START, END = train_start, test_end
# ==================

# ===== CUDA STATUS CHECK - ALL MODELS =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('')
print('=' * 70)
print(' ' * 20 + 'CUDA STATUS - ALL MODELS')
print('=' * 70)
print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA Version:     {torch.version.cuda}')
    print(f'  Device Count:     {torch.cuda.device_count()}')
    print(f'  Device Name:      {torch.cuda.get_device_name(0)}')
print('-' * 70)
print('MODEL BY MODEL CUDA USAGE:')
print('-' * 70)
print(f'  LightGBM:     {"CUDA" if device.type == "cuda" else "CPU"} (via device parameter in objective)')
print(f'  CatBoost:     {"CUDA" if device.type == "cuda" else "CPU"} (via task_type parameter)')
print(f'  TimeXer:      {"CUDA" if device.type == "cuda" else "CPU"} (device={device})')
print(f'  Moirai-MoE:   {"CUDA" if device.type == "cuda" else "CPU"} (device={device})')
print(f'  Base LSTM:    {"CUDA" if device.type == "cuda" else "CPU"} (device={device})')
print(f'  Meta LSTM:    {"CUDA" if device.type == "cuda" else "CPU"} (device={device})')
print('=' * 70)
print('')
# ==========================================

print(f'[1/11] Descargando datos...')
download_yf(['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC', 'GC=F', 'CBOE'], START, END)
download_cx(['BTC/USDT', 'ETH/USDT'], START, END)
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'tokens', f'{TOKEN.replace("/", "-")}_2020-2025.csv'))

# Cargar predicciones externas (TOKEN_all_models_predictions.csv)
csv_ext = os.path.join(os.path.dirname(__file__), 'Parker_predictions', f'{TOKEN.replace("/", "-")}_all_models_predictions.csv')
ext_preds_map = {}
idx_start_ext = None

# [DEBUG] Verificar ruta y existencia
print(f'[1.5/11] Buscando predicciones externas en: {os.path.abspath(csv_ext)}')
if not os.path.exists(csv_ext):
    parker_dir = os.path.join(os.path.dirname(__file__), 'Parker_predictions')
    disponibles = os.listdir(parker_dir) if os.path.isdir(parker_dir) else []
    raise FileNotFoundError(
        f'[Parker] Archivo no encontrado: {os.path.abspath(csv_ext)}\n'
        f'  Archivos en carpeta: {disponibles}'
    )

print(f'[1.5/11] Alineando predicciones externas (por fecha)...')
df_ext = pd.read_csv(csv_ext, parse_dates=['Date'])
# [DEBUG] Verificar carga
print(f'  [Parker DEBUG] Primeras filas:\n{df_ext.head(3)}')
print(f'  [Parker DEBUG] dtypes:\n{df_ext.dtypes}')
print(f'  [Parker DEBUG] Columna Date es datetime: {pd.api.types.is_datetime64_any_dtype(df_ext["Date"])}')

# Alinear por FECHA (los valores de Parker son normalizados, no en USD)
df_date_to_idx = {pd.to_datetime(d).date(): i for i, d in enumerate(df['Date_final'])}
for col in ['LSTM', 'GRU', 'ARIMA', 'RF', 'Transformer', 'XGBoost']:
    key = col.upper().replace(' ', '_') + '_EXT'
    full_raw = np.full(len(df), np.nan)
    for _, row in df_ext.iterrows():
        date_key = row['Date'].date()
        if date_key in df_date_to_idx:
            full_raw[df_date_to_idx[date_key]] = row[col]
    ext_preds_map[key] = full_raw

# [DEBUG] Verificar alineación
aligned_count = int((~np.isnan(ext_preds_map.get('XGBOOST_EXT', np.array([np.nan])))).sum())
print(f'  [Parker DEBUG] Filas alineadas por fecha: {aligned_count}/{len(df_ext)}')
if aligned_count == 0:
    print(f'  [Parker DEBUG] ADVERTENCIA: ninguna fecha del CSV coincide con df.')
    print(f'  [Parker DEBUG] Muestra fechas CSV: {df_ext["Date"].dt.date.tolist()[:5]}')
    print(f'  [Parker DEBUG] Muestra fechas df:  {[str(d) for d in list(df_date_to_idx.keys())[:5]]}')
else:
    # Determinar idx_start_ext = primer índice con dato válido en df
    idx_start_ext = int(np.where(~np.isnan(ext_preds_map['XGBOOST_EXT']))[0][0])
    print(f'  [Parker DEBUG] idx_start_ext={idx_start_ext}')
    print(f'  [Parker DEBUG] Muestra XGBOOST_EXT (5 vals): {ext_preds_map["XGBOOST_EXT"][idx_start_ext:idx_start_ext+5]}')


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

# Dividir por fecha de inicio y fin de test
dates = pd.to_datetime(df['Date_final']).iloc[orig_idx_array].reset_index(drop=True)
mask_test = (dates >= pd.to_datetime(test_start)) & (dates <= pd.to_datetime(test_end))
if mask_test.any():
    ts = mask_test.idxmax()  # Inicio del test
    te = mask_test[::-1].idxmax()  # Fin del test (ultimo True)
else:
    ts = int(len(df_f) * .9)
    te = len(df_f)

Xtr, Xte = df_f.iloc[:ts].copy(), df_f.iloc[ts:te+1].copy()
ytr, yte = lc_r.iloc[:ts].copy(), lc_r.iloc[ts:te+1].copy()

sf = MinMaxScaler()
Xtr_s = pd.DataFrame(sf.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
Xte_s = pd.DataFrame(sf.transform(Xte), columns=Xte.columns, index=Xte.index)

sct = MinMaxScaler()
ytr_s = pd.Series(sct.fit_transform(ytr.values.reshape(-1, 1)).flatten(), index=ytr.index, name='lc')
yte_s = pd.Series(sct.transform(yte.values.reshape(-1, 1)).flatten(), index=yte.index, name='lc')

feats, mic_v = top_k(Xtr_s, ytr_s, 15)
Xt, Xe = Xtr_s[feats].reset_index(drop=True), Xte_s[feats].reset_index(drop=True)
yt, ye = ytr_s.reset_index(drop=True), yte_s.reset_index(drop=True)

# --- Gráfica MIC ---
try:
    import plotly.graph_objects as go
    import plotly.colors as pc

    # mic_v es un dict {feature_name: score} — extraer y ordenar ascendente
    # (para barras horizontales, el de arriba es el último → mayor score arriba)
    mic_items = sorted(mic_v.items(), key=lambda x: x[1])  # ascending by score
    feats_sorted = [item[0] for item in mic_items]
    mic_sorted = [item[1] for item in mic_items]

    # Normalizar scores a [0,1] para colormap
    mic_arr = np.array(mic_sorted)
    mn_mic, mx_mic = mic_arr.min(), mic_arr.max()
    norm = (mic_arr - mn_mic) / (mx_mic - mn_mic) if mx_mic > mn_mic else np.full_like(mic_arr, 0.5)
    bar_colors = pc.sample_colorscale('Viridis', norm.tolist())

    fig_mic = go.Figure()
    fig_mic.add_trace(go.Bar(
        x=mic_sorted, y=feats_sorted, orientation='h',
        marker=dict(color=bar_colors),
        text=[f'{v:.3f}' for v in mic_sorted],
        textposition='outside', textfont=dict(color='#000000', size=10),
        showlegend=False
    ))
    # Invisible scatter for colorbar
    fig_mic.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale='Viridis', cmin=mn_mic, cmax=mx_mic,
            colorbar=dict(title='MIC Score', thickness=15, len=0.8),
            showscale=True, color=[mn_mic]
        ),
        showlegend=False, hoverinfo='skip'
    ))
    fig_mic.update_layout(
        title=dict(text=f'{TOKEN} — Maximal Information Coefficient (MIC) - Feature Importance',
                   x=0.5, font=dict(size=14, color='#000')),
        xaxis=dict(title='MIC Score', range=[mn_mic - 0.001, mx_mic + 0.001],
                   showgrid=False, linecolor='#ccc'),
        yaxis=dict(tickfont=dict(size=11), showgrid=True,
                   gridcolor='rgba(200,200,200,0.3)', linecolor='#ccc'),
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        width=1200, height=max(400, len(feats) * 40),
        margin=dict(l=200, r=80, t=60, b=50)
    )
    mic_dir = os.path.join(os.path.dirname(__file__), 'mic_features')
    os.makedirs(mic_dir, exist_ok=True)
    safe_tok = TOKEN.replace('/', '-').replace('^', '').replace('=', '-')
    mic_path = os.path.join(mic_dir, f'mic_{safe_tok}.png')
    fig_mic.write_image(mic_path, scale=2)
    print(f'  [MIC] Gráfica guardada en: {mic_path}')
except Exception as e:
    logging.warning(f'[MIC] No se pudo generar la gráfica MIC: {e}')

# Walk Forward
print(f'[4/11] Split Walk-Forward...')
# Usar 75% del data para training para asegurar que TimeXer/Moirai tengan suficientes muestras
# (seq_len=96 + pred_len=30 + 10 = 136 mínimo requerido)
k = 5; sp = wfrw(yt, k=k, fh_val=30, window_ratio=0.75)

# Training Base Models
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
oof_b = {}
print('  --- Base Models Ensamble SOTA ---')
print('  > Base LSTM...')
sb = optuna.create_study(direction='minimize')
sb.optimize(lambda t: objective_base_lstm_global(t, Xt, yt, sp, device=device, oof_storage=oof_b), n_trials=N_BL, n_jobs=1)
bp_b = oof_b.get('params', sb.best_params)


# Meta Ensamble Actual
print(f'[6/11] Entrenando Meta LSTM (Ensamble Actual Completo)...')
oof_df_actual = build_oof_dataframe(oof_l, oof_c, oof_t, oof_m, yt)
print(f'  OOF matrix shape: {oof_df_actual.shape}')
print(f'  OOF columns: {oof_df_actual.columns.tolist()}')
print(f'  OOF NaN count per column: {oof_df_actual.isna().sum().to_dict()}')
if len(oof_df_actual) < 20:
    print(f'  [ERROR] OOF dataframe too small ({len(oof_df_actual)} samples). Required: >= 20')
meta_model_actual, _, _, bp_mt, _ = optimize_lstm_meta(oof_df_actual, device, n_trials=N_MT)
ws_meta_actual = bp_mt.get('window_size', 10) if meta_model_actual is not None else 10
if meta_model_actual is None:
    print(f'  [ERROR] meta_model_actual is None. Training failed.')

# Meta Ensamble Ablation (Sin TimeXer)
print(f'[7/11] Entrenando Meta LSTM Ours (Sin TimeXer)...')
oof_df_ablation = build_oof_dataframe_ablation(oof_l, oof_c, oof_m, yt)
print(f'  OOF matrix shape: {oof_df_ablation.shape}')
# Reusamos la lógica de optimización de LSTM actual ya que la estructura base no cambia (solo el # features)
meta_model_ablation, _, _, bp_ab, _ = optimize_lstm_meta(oof_df_ablation, device, n_trials=N_AB)
ws_meta_ablation = bp_ab.get('window_size', 10) if meta_model_ablation is not None else 10

# Meta Ensamble SOTA
print(f'[8/11] Entrenando Stacking Meta LSTM (Modelo Yu et al.)...')
oof_df_sota = build_oof_dataframe_sota(oof_l, oof_c, oof_b, yt)
print(f'  OOF matrix shape: {oof_df_sota.shape}')
meta_model_sota, _, _, bp_sm, _ = optimize_stacking_meta(oof_df_sota, device, n_trials=N_SM)
ws_meta_sota = bp_sm.get('window_size', 10) if meta_model_sota is not None else 10


# Generando Predicciones Base
print(f'[9/11] Predicciones en Set de Prueba (Modelos Base)...')

# LGB y CatBoost no necesitan contexto secuencial
pl, _ = lgb_predict_test(Xt, yt, Xe, bp_l)
pc, _ = cb_predict_test(Xt, yt, Xe, bp_c)

# ── TimeXer: prepend contexto de entrenamiento ──
ctx_tx = 96  # seq_len
Xt_tail_tx = Xt.iloc[-ctx_tx:]
yt_tail_tx = yt.iloc[-ctx_tx:]
Xe_ctx_tx = pd.concat([Xt_tail_tx, Xe], axis=0).reset_index(drop=True)
ye_ctx_tx = pd.concat([yt_tail_tx, ye], axis=0).reset_index(drop=True)

pt_full, _, _ = tx_predict_test(
    Xt, yt, Xe_ctx_tx, ye_ctx_tx, bp_t, device,
    seq_len=96, pred_len=1, features='MS'
)
pt = pt_full[-len(ye):]  # Recortar: solo la porción de test real

# ── Moirai: prepend contexto de entrenamiento ──
ctx_mo = bp_m.get('context_length', 96)
yt_tail_mo = yt.iloc[-ctx_mo:]
ye_ctx_mo = pd.concat([yt_tail_mo, ye], axis=0).reset_index(drop=True)

pm_full, _ = moirai_predict_test(
    pd.concat([yt.iloc[-ctx_mo:], yt], axis=0).reset_index(drop=True),
    ye_ctx_mo, bp_m, model_size='small', freq='D'
)
pm = pm_full[-len(ye):]  # Recortar: solo la porción de test real

# Base LSTM (maneja su propio contexto secuencial)
pb, _ = train_final_base_lstm(Xt, yt, Xe, bp_b, device)

# Verificar que TODOS los base models cubren el test completo
for name, arr in [('LGB', pl), ('CB', pc), ('TX', pt), ('MO', pm),
                  ('BL', pb)]:
    n_valid = (~np.isnan(arr)).sum()
    print(f'  [{name}] {n_valid}/{len(ye)} predicciones válidas')


# Generando Predicciones Meta
print(f'[10/11] Predicciones Ensambles (Meta-Learners)...')
pmt_actual = np.full(len(ye), np.nan)
pmt_ablation = np.full(len(ye), np.nan)
pmt_sota = np.full(len(ye), np.nan)

if meta_model_actual is not None:
    test_matrix_actual = np.column_stack([pl, pc, pt, pm]).astype(np.float32)
    meta_model_actual.eval()
    with torch.no_grad():
        for i in range(ws_meta_actual - 1, len(ye)):
            window = test_matrix_actual[i - ws_meta_actual + 1:i + 1]
            if not np.isnan(window).any():
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                pmt_actual[i] = meta_model_actual(x_t).cpu().item()
    print(f'  [MT] {(~np.isnan(pmt_actual)).sum()}/{len(ye)} válidas')

if meta_model_ablation is not None:
    test_matrix_ablation = np.column_stack([pl, pc, pm]).astype(np.float32)
    meta_model_ablation.eval()
    with torch.no_grad():
        for i in range(ws_meta_ablation - 1, len(ye)):
            window = test_matrix_ablation[i - ws_meta_ablation + 1:i + 1]
            if not np.isnan(window).any():
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                pmt_ablation[i] = meta_model_ablation(x_t).cpu().item()
    print(f'  [AB] {(~np.isnan(pmt_ablation)).sum()}/{len(ye)} válidas')

if meta_model_sota is not None:
    test_matrix_sota = np.column_stack([pl, pc, pb]).astype(np.float32)
    meta_model_sota.eval()
    with torch.no_grad():
        for i in range(ws_meta_sota - 1, len(ye)):
            window = test_matrix_sota[i - ws_meta_sota + 1:i + 1]
            if not np.isnan(window).any():
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                pmt_sota[i] = meta_model_sota(x_t).cpu().item()
    print(f'  [SM] {(~np.isnan(pmt_sota)).sum()}/{len(ye)} válidas')

# --- PARTE 1A: Simple Average ---
print('  > Simple Average (Ensamble Actual)...')
pmt_simple_avg = np.full(len(ye), np.nan)
try:
    mask_sa = (~np.isnan(pl)) & (~np.isnan(pc)) & (~np.isnan(pt)) & (~np.isnan(pm))
    pmt_simple_avg[mask_sa] = np.mean(
        np.column_stack([pl[mask_sa], pc[mask_sa], pt[mask_sa], pm[mask_sa]]), axis=1
    )
    print(f'  [SA] {(~np.isnan(pmt_simple_avg)).sum()}/{len(ye)} válidas')
except Exception as e:
    logging.warning(f'[SA] Simple Average falló: {e}')
    pmt_simple_avg = np.full(len(ye), np.nan)

# --- PARTE 1B: Weighted Average (Optuna) ---
print('  > Weighted Average (Ensamble Actual)...')
pmt_weighted_avg = np.full(len(ye), np.nan)
try:
    def objective_weighted_avg(trial, oof_df):
        w = np.array([
            trial.suggest_float('w_lgb', 0.0, 1.0),
            trial.suggest_float('w_cb',  0.0, 1.0),
            trial.suggest_float('w_tx',  0.0, 1.0),
            trial.suggest_float('w_mo',  0.0, 1.0),
        ])
        w = w / w.sum()
        cols = ['lgb', 'catboost', 'timexer', 'moirai']
        pred = (oof_df[cols].values * w).sum(axis=1)
        return mean_squared_error(oof_df['target'].values, pred)

    study_wa = optuna.create_study(direction='minimize')
    study_wa.optimize(lambda t: objective_weighted_avg(t, oof_df_actual), n_trials=N_MT, n_jobs=1)
    best_w = np.array([
        study_wa.best_params['w_lgb'],
        study_wa.best_params['w_cb'],
        study_wa.best_params['w_tx'],
        study_wa.best_params['w_mo'],
    ])
    best_w = best_w / best_w.sum()
    print(f'  [WA] Pesos óptimos: LGB={best_w[0]:.4f}, CB={best_w[1]:.4f}, TX={best_w[2]:.4f}, MO={best_w[3]:.4f}')
    mask_wa = (~np.isnan(pl)) & (~np.isnan(pc)) & (~np.isnan(pt)) & (~np.isnan(pm))
    X_test_wa = np.column_stack([pl[mask_wa], pc[mask_wa], pt[mask_wa], pm[mask_wa]])
    pmt_weighted_avg[mask_wa] = (X_test_wa * best_w).sum(axis=1)
    print(f'  [WA] {(~np.isnan(pmt_weighted_avg)).sum()}/{len(ye)} válidas')
except Exception as e:
    logging.warning(f'[WA] Weighted Average falló: {e}')
    pmt_weighted_avg = np.full(len(ye), np.nan)

# --- PARTE 2: Meta modelos lineales (Ridge, Lasso, Elastic Net) ---
oof_cols = ['lgb', 'catboost', 'timexer', 'moirai']
X_oof = oof_df_actual[oof_cols].values
y_oof = oof_df_actual['target'].values
X_test_linear = np.column_stack([pl, pc, pt, pm])

# Ridge
print('  > Ridge (Ensamble Actual)...')
pmt_ridge = np.full(len(ye), np.nan)
try:
    def objective_ridge(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
        kf = KFold(n_splits=5, shuffle=False)
        scores = []
        for tr_idx, va_idx in kf.split(X_oof):
            m = Ridge(alpha=alpha)
            m.fit(X_oof[tr_idx], y_oof[tr_idx])
            scores.append(mean_squared_error(y_oof[va_idx], m.predict(X_oof[va_idx])))
        return np.mean(scores)

    study_rd = optuna.create_study(direction='minimize')
    study_rd.optimize(objective_ridge, n_trials=N_MT, n_jobs=1)
    best_ridge = Ridge(alpha=study_rd.best_params['alpha'])
    best_ridge.fit(X_oof, y_oof)
    mask_rd = ~np.any(np.isnan(X_test_linear), axis=1)
    pmt_ridge[mask_rd] = best_ridge.predict(X_test_linear[mask_rd])
    print(f'  [RD] alpha={study_rd.best_params["alpha"]:.6f}, {(~np.isnan(pmt_ridge)).sum()}/{len(ye)} válidas')
except Exception as e:
    logging.warning(f'[RD] Ridge falló: {e}')
    pmt_ridge = np.full(len(ye), np.nan)

# Lasso
print('  > Lasso (Ensamble Actual)...')
pmt_lasso = np.full(len(ye), np.nan)
try:
    def objective_lasso(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
        kf = KFold(n_splits=5, shuffle=False)
        scores = []
        for tr_idx, va_idx in kf.split(X_oof):
            m = Lasso(alpha=alpha, max_iter=10000)
            m.fit(X_oof[tr_idx], y_oof[tr_idx])
            scores.append(mean_squared_error(y_oof[va_idx], m.predict(X_oof[va_idx])))
        return np.mean(scores)

    study_ls = optuna.create_study(direction='minimize')
    study_ls.optimize(objective_lasso, n_trials=N_MT, n_jobs=1)
    best_lasso = Lasso(alpha=study_ls.best_params['alpha'], max_iter=10000)
    best_lasso.fit(X_oof, y_oof)
    mask_ls = ~np.any(np.isnan(X_test_linear), axis=1)
    pmt_lasso[mask_ls] = best_lasso.predict(X_test_linear[mask_ls])
    print(f'  [LS] alpha={study_ls.best_params["alpha"]:.6f}, {(~np.isnan(pmt_lasso)).sum()}/{len(ye)} válidas')
except Exception as e:
    logging.warning(f'[LS] Lasso falló: {e}')
    pmt_lasso = np.full(len(ye), np.nan)

# Elastic Net
print('  > Elastic Net (Ensamble Actual)...')
pmt_elasticnet = np.full(len(ye), np.nan)
try:
    def objective_elasticnet(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        kf = KFold(n_splits=5, shuffle=False)
        scores = []
        for tr_idx, va_idx in kf.split(X_oof):
            m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            m.fit(X_oof[tr_idx], y_oof[tr_idx])
            scores.append(mean_squared_error(y_oof[va_idx], m.predict(X_oof[va_idx])))
        return np.mean(scores)

    study_en = optuna.create_study(direction='minimize')
    study_en.optimize(objective_elasticnet, n_trials=N_MT, n_jobs=1)
    best_en = ElasticNet(alpha=study_en.best_params['alpha'], l1_ratio=study_en.best_params['l1_ratio'], max_iter=10000)
    best_en.fit(X_oof, y_oof)
    mask_en = ~np.any(np.isnan(X_test_linear), axis=1)
    pmt_elasticnet[mask_en] = best_en.predict(X_test_linear[mask_en])
    print(f'  [EN] alpha={study_en.best_params["alpha"]:.6f}, l1_ratio={study_en.best_params["l1_ratio"]:.4f}, {(~np.isnan(pmt_elasticnet)).sum()}/{len(ye)} válidas')
except Exception as e:
    logging.warning(f'[EN] Elastic Net falló: {e}')
    pmt_elasticnet = np.full(len(ye), np.nan)

# Solo truncar las primeras ws-1 posiciones de base models (warm-up del meta)
start_idx = max(ws_meta_actual, ws_meta_ablation, ws_meta_sota) - 1
for arr in [pl, pc, pt, pm, pb]:
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
    'BL':  safe_inv_recon(pb),
    'MT':  safe_inv_recon(pmt_actual),
    'SA':  safe_inv_recon(pmt_simple_avg),
    'WA':  safe_inv_recon(pmt_weighted_avg),
    'RD':  safe_inv_recon(pmt_ridge),
    'LS':  safe_inv_recon(pmt_lasso),
    'EN':  safe_inv_recon(pmt_elasticnet),
    'AB':  safe_inv_recon(pmt_ablation),
    'SM':  safe_inv_recon(pmt_sota),
    # Predicciones externas (Parker): los valores están en escala LogReturn_MinMax [0,1]
    # igual que nuestros modelos, hay que aplicar safe_inv_recon para obtener USD.
    # safe_inv_recon espera un array de longitud n (test completo) y aplica [val] internamente.
    'LSTM_EXT':     safe_inv_recon(ext_preds_map.get('LSTM_EXT',        np.full(len(df), np.nan))[test_orig_indices]),
    'GRU_EXT':      safe_inv_recon(ext_preds_map.get('GRU_EXT',         np.full(len(df), np.nan))[test_orig_indices]),
    'ARIMA_EXT':    safe_inv_recon(ext_preds_map.get('ARIMA_EXT',       np.full(len(df), np.nan))[test_orig_indices]),
    'RF_EXT':       safe_inv_recon(ext_preds_map.get('RF_EXT',          np.full(len(df), np.nan))[test_orig_indices]),
    'TRANS_EXT':    safe_inv_recon(ext_preds_map.get('TRANSFORMER_EXT', np.full(len(df), np.nan))[test_orig_indices]),
    'XGB_META_EXT': safe_inv_recon(ext_preds_map.get('XGBOOST_EXT',    np.full(len(df), np.nan))[test_orig_indices]),
}
preds_p['Ensamble Actual'] = preds_p['MT']  # Alias para acceso por nombre

mp = []
for km, v in preds_p.items():
    if km not in MDL:  # Saltar alias (e.g. 'Ensamble Actual')
        continue
    if (~np.isnan(v)).any():
        v_valid = v[~np.isnan(v)]
        y_valid = pr_r[~np.isnan(v)]
        if len(v_valid) > 0 and len(y_valid) > 0:
            da_val = directional_accuracy(y_valid, v_valid) if len(v_valid) >= 2 else 0.0
            mp.append({'Modelo': MDL[km][1], **met(y_valid, v_valid), 'DA': round(da_val, 2)})

# Ordenar el reporte por MAE descendente para mejor visualización
mp.sort(key=lambda x: x['MAE'])

# [DEBUG] Verificar que 'Ensamble Actual' (MT) esté presente en mp
mt_nombres = [m_['Modelo'] for m_ in mp]
print(f"[DEBUG] Modelos en mp: {mt_nombres}")
if 'Ensamble Actual' not in mt_nombres:
    logging.warning(
        "[WARN] 'Ensamble Actual' no aparece en mp. "
        "meta_model_actual puede ser None o test_matrix_actual tiene NaNs en TX/MO."
    )

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

meta_raw_preds = {'MT': pmt_actual, 'SA': pmt_simple_avg, 'WA': pmt_weighted_avg,
                  'RD': pmt_ridge, 'LS': pmt_lasso, 'EN': pmt_elasticnet,
                  'AB': pmt_ablation, 'SM': pmt_sota, 'XGB_META_EXT': pmt_parker}
meta_raw_preds['Ensamble Actual'] = meta_raw_preds['MT']  # Alias para acceso por nombre

# Verificar existencia de meta_raw_preds['Ensamble Actual']
if 'Ensamble Actual' not in meta_raw_preds:
    raise KeyError('meta_raw_preds["Ensamble Actual"] no encontrado. El meta modelo actual no fue entrenado correctamente.')
# Predicciones raw de modelos base para gráficos LogReturn_MinMax
base_raw_preds = {'LGB': pl, 'CB': pc, 'TX': pt, 'MO': pm, 'BL': pb}

# Convertir predicciones externas (USD) a LogReturn_MinMax para gráficos LR
for ext_key in ['LSTM_EXT', 'GRU_EXT', 'ARIMA_EXT', 'RF_EXT', 'TRANS_EXT']:
    p_usd_ext = preds_p.get(ext_key)
    if p_usd_ext is not None:
        pmt_ext = np.full(len(ye), np.nan)
        v_mask_ext = ~np.isnan(p_usd_ext)
        if v_mask_ext.any():
            p_log_ext = np.log(p_usd_ext[v_mask_ext] / prev[v_mask_ext])
            pmt_ext[v_mask_ext] = sct.transform(p_log_ext.reshape(-1, 1)).flatten()
        base_raw_preds[ext_key] = pmt_ext

import json
def generate_compare_report(token, cp, gi_v, pr_r, preds_p, mp, MDL, zs, ze, out_dir, ye_vals=None, meta_raw_preds=None, base_raw_preds=None, met_fn=None, da_fn=None):
    """Genera report_compare.html aislador para los Ensambles"""
    # --- Bug 1 fix: Close (USD) debe tener el mismo dominio X que las predicciones.
    # gi_v son los índices globales en cp correspondientes al set de prueba.
    # pr_r son los precios Close reconstruidos alineados a gi_v.
    # Usamos pr_r/gi_v como serie Close para evitar el recorte.
    close_x = [int(x) for x in gi_v]
    close_y = [float(v) for v in pr_r]

    zoom_models = {}
    for km, (cl, nm) in MDL.items():
        v = preds_p[km]
        m = ~np.isnan(v)
        if m.any():
            zoom_models[km] = {'name': nm, 'x': [int(x) for x in gi_v[m]], 'y': [float(y) for y in v[m]], 'color': cl}

    # [DEBUG] Verificar Parker antes de graficar
    parker_v = preds_p.get('XGB_META_EXT', np.array([]))
    parker_valid = parker_v[~np.isnan(parker_v)] if len(parker_v) > 0 else np.array([])
    print(f'  [Parker PLOT DEBUG] Puntos válidos XGB_META_EXT para graficar: {len(parker_valid)}')
    if len(parker_valid) == 0:
        print('  [Parker PLOT DEBUG] ADVERTENCIA: XGB_META_EXT está vacío/todo-NaN. No aparecerá en la gráfica.')
    else:
        print(f'  [Parker PLOT DEBUG] Rango USD: [{parker_valid.min():.2f}, {parker_valid.max():.2f}]')
        print(f'  [Parker PLOT DEBUG] Rango Close USD: [{pr_r.min():.2f}, {pr_r.max():.2f}]')
    if len(parker_valid) == 0:
        logging.warning('[Parker] XGB_META_EXT tiene 0 puntos válidos — no aparecerá en la gráfica.')

    
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
    # Forzar orden explícito de filas
    orden = [
        'Ensamble Actual',
        'Ours (Ensamble Actual Sin TimeXer)',
        'Yu et al. [44] 2025',
        'Parker et al. 2025'
    ]
    mp_metas.sort(key=lambda x: orden.index(x['Modelo'])
                  if x['Modelo'] in orden else 99)
        
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
  <h1>{token} - Comparativa: Ours vs SOTA. </h1>
  <p class="subtitle">Predicciones Finales y Modelos Base (Alineados temporalmente)</p>
  
  <div class="charts-grid">
    <div class="card">
      <h2>Ours</h2>
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
    <table class="metrics-table"><thead><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R2</th><th>DA (%)</th></tr></thead><tbody>
"""
    best_vals = {}
    if mp_metas:
        for mn in ['MSE', 'RMSE', 'MAE', 'R2', 'DA']:
            vals = [m_[mn] for m_ in mp_metas if mn in m_ and not np.isnan(m_[mn])]
            if vals:
                best_vals[mn] = max(vals) if mn in ['R2', 'DA'] else min(vals)

    def _fmt(val, mn):
        if mn == 'DA':
            s = f'{val:.2f}%'
        else:
            s = f'{val:.6f}'
        if mn in best_vals and abs(val - best_vals[mn]) < 1e-9:
            return f'<td style="background:#d4edda">{s}</td>'
        return f'<td>{s}</td>'

    for i, m_ in enumerate(mp_metas):
        da_val = m_.get('DA', float('nan'))
        da_fmt = _fmt(da_val, 'DA') if not np.isnan(da_val) else '<td>N/A</td>'
        html += f'<tr><td><span class="model-badge" style="background:{m_["Color"]}">{m_["Modelo"]}</span></td>{_fmt(m_["MSE"], "MSE")}{_fmt(m_["RMSE"], "RMSE")}{_fmt(m_["MAE"], "MAE")}{_fmt(m_["R2"], "R2")}{da_fmt}</tr>\n'
        
    html += """    </tbody></table></div>
"""

    # ===== PARTE 4: Comparación de Meta Modelos — Ensamble Actual =====
    meta_compare_keys = ['MT', 'SA', 'WA', 'RD', 'LS', 'EN']
    meta_compare_rows = []
    for km in meta_compare_keys:
        pred = preds_p.get(km)
        if pred is None:
            meta_compare_rows.append({'key': km, 'valid': False})
            continue
        mask = (~np.isnan(pred)) & (~np.isnan(pr_r))
        if mask.sum() > 0 and met_fn is not None and da_fn is not None:
            m = met_fn(pr_r[mask], pred[mask])
            da_val = da_fn(pr_r[mask], pred[mask]) if mask.sum() >= 2 else float('nan')
            meta_compare_rows.append({
                'key': km, 'valid': True,
                'MSE': m['MSE'], 'RMSE': m['RMSE'], 'MAE': m['MAE'], 'R2': m['R2'],
                'DA': round(da_val, 2)
            })
        else:
            meta_compare_rows.append({'key': km, 'valid': False})

    # Best per column
    mc_best = {}
    mc_valid = [r for r in meta_compare_rows if r['valid']]
    if mc_valid:
        for mn in ['MSE', 'RMSE', 'MAE']:
            vals = [r[mn] for r in mc_valid if not np.isnan(r[mn])]
            if vals: mc_best[mn] = min(vals)
        for mn in ['R2', 'DA']:
            vals = [r[mn] for r in mc_valid if not np.isnan(r[mn])]
            if vals: mc_best[mn] = max(vals)

    html += '  <div class="card"><h2>Comparación de Meta Modelos — Ensamble Actual</h2>\n'
    html += '    <table class="metrics-table"><thead><tr><th>MODELO</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R2</th><th>DA (%)</th></tr></thead>\n'
    html += '    <tbody>\n'
    for row in meta_compare_rows:
        color = MDL[row['key']][0] if row['key'] in MDL else '#888'
        name = MDL[row['key']][1] if row['key'] in MDL else row['key']
        badge = f'<span class="model-badge" style="background:{color}">{name}</span>'
        if not row['valid']:
            html += f'    <tr><td>{badge}</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>\n'
            continue
        cells = ''
        for mn in ['MSE', 'RMSE', 'MAE', 'R2']:
            val = row[mn]
            hi = ' style="background:#d4edda"' if mn in mc_best and abs(val - mc_best[mn]) < 1e-9 else ''
            cells += f'<td{hi}>{val:.6f}</td>'
        da_val = row['DA']
        da_hi = ' style="background:#d4edda"' if 'DA' in mc_best and abs(da_val - mc_best['DA']) < 1e-9 else ''
        cells += f'<td{da_hi}>{da_val:.2f}%</td>'
        html += f'    <tr><td>{badge}</td>{cells}</tr>\n'
    html += '    </tbody></table></div>\n'

    # ===== SECCIÓN: Métricas por Ensamble (Test Set — Escala USD) =====
    ensemble_groups = [
        ('Ensamble Actual', [
            ('MT',  'Meta LSTM (Ensamble Actual)', True),
            ('LGB', 'LightGBM', False),
            ('CB',  'CatBoost', False),
            ('TX',  'TimeXer', False),
            ('MO',  'Moirai-MoE', False),
        ]),
        ('Ours (Sin TimeXer)', [
            ('AB',  'Ours (Ensamble Actual Sin TimeXer)', True),
            ('LGB', 'LightGBM', False),
            ('CB',  'CatBoost', False),
            ('MO',  'Moirai-MoE', False),
        ]),
        ('Yu et al. [44] 2025', [
            ('SM',  'Yu et al. [44] 2025', True),
            ('LGB', 'LightGBM', False),
            ('CB',  'CatBoost', False),
            ('BL',  'Base LSTM', False),
        ]),
        ('Parker et al. 2025', [
            ('XGB_META_EXT', 'Parker et al. 2025', True),
            ('LSTM_EXT',     'LSTM (Externo)', False),
            ('GRU_EXT',      'GRU (Externo)', False),
            ('ARIMA_EXT',    'ARIMA (Externo)', False),
            ('RF_EXT',       'Random Forest (Externo)', False),
            ('TRANS_EXT',    'Transformer (Externo)', False),
        ]),
    ]

    html += '  <div class="card"><h2>M\u00e9tricas por Ensamble (Test Set &mdash; Escala USD)</h2>\n'

    for ens_name, models in ensemble_groups:
        html += f'    <h3 style="margin-top:20px;margin-bottom:10px;font-size:1.1rem;color:#333">{ens_name}</h3>\n'
        html += '    <table class="metrics-table" style="margin-bottom:20px">\n'
        html += '      <thead><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R2</th><th>DA (%)</th></tr></thead>\n'
        html += '      <tbody>\n'

        # Compute metrics for each model in this ensemble
        ens_rows = []
        for (key, label, is_meta) in models:
            pred = preds_p.get(key)
            if pred is None:
                ens_rows.append({'key': key, 'label': label, 'is_meta': is_meta, 'valid': False})
                continue
            mask = (~np.isnan(pred)) & (~np.isnan(pr_r))
            if mask.sum() > 0 and met_fn is not None and da_fn is not None:
                m = met_fn(pr_r[mask], pred[mask])
                da_val = da_fn(pr_r[mask], pred[mask]) if mask.sum() >= 2 else float('nan')
                ens_rows.append({'key': key, 'label': label, 'is_meta': is_meta, 'valid': True,
                                 'MSE': m['MSE'], 'RMSE': m['RMSE'], 'MAE': m['MAE'], 'R2': m['R2'], 'DA': round(da_val, 2)})
            else:
                ens_rows.append({'key': key, 'label': label, 'is_meta': is_meta, 'valid': False})

        # Find best per column among valid rows
        ens_best = {}
        valid_rows = [r for r in ens_rows if r['valid']]
        if valid_rows:
            for mn in ['MSE', 'RMSE', 'MAE']:
                vals = [r[mn] for r in valid_rows if not np.isnan(r[mn])]
                if vals: ens_best[mn] = min(vals)
            for mn in ['R2', 'DA']:
                vals = [r[mn] for r in valid_rows if not np.isnan(r[mn])]
                if vals: ens_best[mn] = max(vals)

        # Render rows
        for row in ens_rows:
            border_style = ' style="border-top:2px solid #333"' if row['is_meta'] else ''
            color = MDL[row['key']][0] if row['key'] in MDL else '#888'
            badge = f'<span class="model-badge" style="background:{color}">{row["label"]}</span>'

            if not row['valid']:
                html += f'      <tr{border_style}><td>{badge}</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>\n'
                continue

            cells = ''
            for mn in ['MSE', 'RMSE', 'MAE', 'R2']:
                val = row[mn]
                hi = ' style="background:#d4edda"' if mn in ens_best and abs(val - ens_best[mn]) < 1e-9 else ''
                cells += f'<td{hi}>{val:.6f}</td>'
            da_val = row['DA']
            da_hi = ' style="background:#d4edda"' if 'DA' in ens_best and abs(da_val - ens_best['DA']) < 1e-9 else ''
            cells += f'<td{da_hi}>{da_val:.2f}%</td>'

            html += f'      <tr{border_style}><td>{badge}</td>{cells}</tr>\n'

        html += '      </tbody>\n    </table>\n'

    html += '  </div>\n'

    html += """  <div class="card"><h2>Prediccion sobre LogReturn_MinMax (Variable Objetivo)</h2>
    <div class="charts-grid">
      <div id="lr-actual"></div>
      <div id="lr-ablation"></div>
      <div id="lr-sota"></div>
      <div id="lr-parker"></div>
    </div>
  </div>
  <div class="card"><h2>Comparacion Visual de Metricas</h2>
    <div class="metrics-grid" style="grid-template-columns:1fr 1fr;grid-template-rows:auto auto auto;">
      <div id="chart-mse"></div><div id="chart-rmse"></div>
      <div id="chart-mae"></div><div id="chart-r2"></div>
      <div id="chart-da" style="grid-column:1/3;max-width:50%;margin:0 auto;"></div>
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
                    # Para MT usar nombre "Ensamble Actual (Meta)" en leyenda LR
                    lr_name = 'Ensamble Actual (Meta)' if km == 'MT' else MDL[km][1]
                    lr_data[km] = {
                        'idx': [int(i) for i in np.where(m_valid)[0]],
                        'y': [float(p) for p in p_raw[m_valid]],
                        'name': lr_name,
                        'color': MDL[km][0]
                    }
        # Agregar predicciones base (raw LogReturn_MinMax) para cada modelo
        if base_raw_preds is not None:
            for km in ['LGB', 'CB', 'TX', 'MO', 'BL', 'LSTM_EXT', 'GRU_EXT', 'ARIMA_EXT', 'RF_EXT', 'TRANS_EXT']:
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

    html += f"const closeX={json.dumps(close_x)};\nconst closeY={json.dumps(close_y)};\n"
    html += f"const zoomModels={json.dumps(zoom_models)};\n"
    html += f"const metricsData={json.dumps(mp_metas)};\n"
    html += f"const lrData={json.dumps(lr_data)};\n"
    
    html += """const dL={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#333',family:'Segoe UI,system-ui,sans-serif'},xaxis:{gridcolor:'#eee',linecolor:'#ccc'},yaxis:{gridcolor:'#eee',linecolor:'#ccc'},margin:{t:40,r:30,b:50,l:60},legend:{bgcolor:'rgba(0,0,0,0)',font:{size:11},orientation:'h',y:-0.2}};

// Wrapper function para graficar
function drawChart(divId, titleTxt, keys, metaKey) {
    // zorder: base models (fondo), meta model (segundo), Close (USD) al frente
    const data = [];
    // 1) Modelos base al fondo
    keys.forEach(k => {
        if(k === metaKey) return;
        if(zoomModels[k] && zoomModels[k].x.length > 0) {
            data.push({
                x: zoomModels[k].x, y: zoomModels[k].y,
                type: 'scatter', mode: 'lines',
                name: zoomModels[k].name,
                line: {color: zoomModels[k].color, width: 1.2, dash: 'dot'},
                marker: {size: 2, color: zoomModels[k].color}
            });
        }
    });
    // 2) Meta modelo en segundo plano
    if(zoomModels[metaKey] && zoomModels[metaKey].x.length > 0) {
        data.push({
            x: zoomModels[metaKey].x, y: zoomModels[metaKey].y,
            type: 'scatter', mode: 'lines+markers',
            name: zoomModels[metaKey].name,
            line: {color: zoomModels[metaKey].color, width: 2.5},
            marker: {size: 4, color: zoomModels[metaKey].color}
        });
    }
    // 3) Close (USD) siempre al frente (zorder máximo) — usa closeX/closeY alineado con el test set
    data.push({x:closeX, y:closeY, type:'scatter', mode:'lines', name:'Close (USD)', line:{color:'#000', width:2}});
    Plotly.newPlot(divId, data, {...dL, title: {text: titleTxt, font: {size: 14, color: '#333'}}, xaxis: {...dL.xaxis, title: 'Indice'}, yaxis: {...dL.yaxis, title: 'USD'}, hovermode: 'x unified'}, {responsive: true});
}

drawChart('zoom-chart-actual', 'Precio Close vs Actual', ['LGB','CB','TX','MO','MT'], 'MT');
drawChart('zoom-chart-ablation', 'Precio Close vs Ours', ['LGB','CB','MO','AB'], 'AB');
drawChart('zoom-chart-sota', 'Precio Close vs Yu et al. 2025', ['LGB','CB','BL','SM'], 'SM');
drawChart('zoom-chart-parker', 'Precio Close vs Parker et al. 2025', ['LSTM_EXT','GRU_EXT','ARIMA_EXT','RF_EXT','TRANS_EXT','XGB_META_EXT'], 'XGB_META_EXT');

function drawLR(divId, titleTxt, metaKey, baseKeys) {
    if(!lrData['_real']) return;
    const real = lrData['_real'];
    // zorder: base models (fondo), meta model (segundo), Real al frente
    const data = [];
    // 1) Modelos base al fondo
    baseKeys.forEach(bk => {
        if(lrData[bk]) {
            data.push({x: lrData[bk].idx, y: lrData[bk].y, type:'scatter', mode:'lines', name: lrData[bk].name, line:{color: lrData[bk].color, width:1.2, dash:'dot'}});
        }
    });
    // 2) Meta-learner en segundo plano
    if(lrData[metaKey]) {
        const pred = lrData[metaKey];
        data.push({x: pred.idx, y: pred.y, type:'scatter', mode:'lines+markers', name: pred.name, line:{color: pred.color, width:2.5}, marker:{size:3, color: pred.color}});
    }
    // 3) Real (LogReturn_MinMax) siempre al frente (zorder máximo)
    data.push({x: real.idx, y: real.y, type:'scatter', mode:'lines', name:'Real (LogReturn_MinMax)', line:{color:'#000', width:2}});
    Plotly.newPlot(divId, data, {...dL, title: {text: titleTxt, font: {size: 14, color: '#333'}}, xaxis: {...dL.xaxis, title: 'Indice Test'}, yaxis: {...dL.yaxis, title: 'LogReturn_MinMax'}, hovermode: 'x unified'}, {responsive: true});
}

drawLR('lr-actual', 'LogReturn_MinMax: Ensamble Actual', 'MT', ['LGB','CB','TX','MO']);
drawLR('lr-ablation', 'LogReturn_MinMax: Ours (Sin TimeXer)', 'AB', ['LGB','CB','MO']);
drawLR('lr-sota', 'LogReturn_MinMax: Yu et al. 2025', 'SM', ['LGB','CB','BL']);
drawLR('lr-parker', 'LogReturn_MinMax: Parker et al. 2025', 'XGB_META_EXT', ['LSTM_EXT','GRU_EXT','ARIMA_EXT','RF_EXT','TRANS_EXT']);

// --- Gráficas de Métricas con DA y resaltado ---
// Orden fijo de modelos para las barras
const metaOrder = ['Ensamble Actual', 'Ours (Ensamble Actual Sin TimeXer)', 'Yu et al. [44] 2025', 'Parker et al. 2025'];
const orderedMetrics = metaOrder.map(name => metricsData.find(m => m.Modelo === name)).filter(m => m);

function drawMetricBar(divId, mn, titleTxt, bestFn, fmtFn) {
    const vals = orderedMetrics.map(m => m[mn]);
    const bestVal = bestFn(vals);
    const borderColors = vals.map(v => v === bestVal ? '#000' : 'rgba(0,0,0,0)');
    const borderWidths = vals.map(v => v === bestVal ? 2.5 : 0);
    Plotly.newPlot(divId, [{
        x: orderedMetrics.map(m => m.Modelo),
        y: vals,
        type: 'bar',
        marker: {color: orderedMetrics.map(m => m.Color), opacity: 0.85, line: {color: borderColors, width: borderWidths}},
        text: vals.map(v => fmtFn(v)),
        textposition: 'outside',
        textfont: {color: '#333', size: 11}
    }], {...dL, title: {text: titleTxt, font: {size: 14, color: '#333'}}, xaxis: {...dL.xaxis, tickangle: 15}, showlegend: false, margin: {t: 50, r: 20, b: 150, l: 60}}, {responsive: true, displayModeBar: false});
}

const minFn = arr => Math.min(...arr);
const maxFn = arr => Math.max(...arr);
const fmt4 = v => v.toFixed(4);
const fmtDA = v => v.toFixed(2) + '%';

drawMetricBar('chart-mse', 'MSE', 'MSE', minFn, fmt4);
drawMetricBar('chart-rmse', 'RMSE', 'RMSE', minFn, fmt4);
drawMetricBar('chart-mae', 'MAE', 'MAE', minFn, fmt4);
drawMetricBar('chart-r2', 'R2', 'R2', maxFn, fmt4);
drawMetricBar('chart-da', 'DA', 'DA (%)', maxFn, fmtDA);
</script></body></html>"""
    # Verificaciones de sanidad antes de escribir
    if 'Ensamble Actual' not in html:
        logging.warning("[WARN] Fila 'Ensamble Actual' no fue insertada en la tabla de métricas")
    if 'DA (%)' not in html:
        logging.warning("[WARN] Columna DA (%) no fue insertada en la tabla")

    safe_token = token.replace('/', '-').replace('^', '').replace('=', '-')
    out_html = os.path.join(out_dir, f'report_compare_{safe_token}.html')
    with open(out_html, 'w', encoding='utf-8') as fh: fh.write(html)
    print(f'[REPORTE] Guardado en: {out_html}')

# Carpeta de reportes (se crea si no existe)
reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(reports_dir, exist_ok=True)

generate_compare_report(TOKEN, cp, gi_v, pr_r, preds_p, mp, MDL, zs, ze, reports_dir, ye_vals=yv, meta_raw_preds=meta_raw_preds, base_raw_preds=base_raw_preds, met_fn=met, da_fn=directional_accuracy)

# ===== INYECTAR DA Y CORREGIR TABLA DE META LEARNERS + METRICAS BAR CHARTS =====
safe_token = TOKEN.replace('/', '-').replace('^', '').replace('=', '-')
html_path_meta = os.path.join(reports_dir, f'report_compare_{safe_token}.html')
if os.path.exists(html_path_meta):
    with open(html_path_meta, 'r', encoding='utf-8') as f:
        soup_meta = BeautifulSoup(f.read(), 'html.parser')

    # Buscar la tabla "Metricas Generales de los Meta Learners"
    meta_table = None
    for h2 in soup_meta.find_all('h2'):
        if 'Metricas Generales' in h2.text:
            meta_table = h2.find_next('table', class_='metrics-table')
            break

    if meta_table:
        # 1) Reconstruir header con DA
        thead_tr = meta_table.find('thead').find('tr')
        thead_tr.clear()
        for col_name in ['Modelo', 'MSE', 'RMSE', 'MAE', 'R2', 'DA (%)']:
            th = soup_meta.new_tag('th')
            th.string = col_name
            thead_tr.append(th)

        # 2) Construir datos de los 4 meta modelos con DA
        meta_order = ['MT', 'AB', 'SM', 'XGB_META_EXT']  # Orden deseado de filas
        meta_rows_data = []
        for km in meta_order:
            if km not in preds_p:
                raise KeyError(f'Modelo "{MDL[km][1]}" (key={km}) no encontrado en preds_p')
            v = preds_p[km]
            m_valid = ~np.isnan(v)
            if not m_valid.any():
                continue
            v_valid = v[m_valid]
            y_valid = pr_r[m_valid]
            if len(v_valid) < 2:
                continue
            metrics = met(y_valid, v_valid)
            da_val = directional_accuracy(y_valid, v_valid)
            meta_rows_data.append({
                'key': km,
                'name': MDL[km][1],
                'color': MDL[km][0],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'DA': round(da_val, 2)
            })

        # 3) Encontrar mejores valores para cada métrica
        best_vals = {}
        if meta_rows_data:
            best_vals['MSE'] = min(r['MSE'] for r in meta_rows_data)
            best_vals['RMSE'] = min(r['RMSE'] for r in meta_rows_data)
            best_vals['MAE'] = min(r['MAE'] for r in meta_rows_data)
            best_vals['R2'] = max(r['R2'] for r in meta_rows_data)
            best_vals['DA'] = max(r['DA'] for r in meta_rows_data)

        # 4) Reconstruir tbody con filas en orden correcto y resaltado verde
        tbody = meta_table.find('tbody')
        tbody.clear()
        for row_data in meta_rows_data:
            tr = soup_meta.new_tag('tr')
            # Columna Modelo (badge)
            td_model = soup_meta.new_tag('td')
            badge = soup_meta.new_tag('span', attrs={'class': 'model-badge', 'style': f'background:{row_data["color"]}'})
            badge.string = row_data['name']
            td_model.append(badge)
            tr.append(td_model)
            # Columnas MSE, RMSE, MAE, R2
            for mn in ['MSE', 'RMSE', 'MAE', 'R2']:
                td = soup_meta.new_tag('td')
                td.string = f'{row_data[mn]:.6f}'
                if mn in best_vals and row_data[mn] == best_vals[mn]:
                    td['style'] = 'background:#d4edda'
                tr.append(td)
            # Columna DA (%)
            td_da = soup_meta.new_tag('td')
            td_da.string = f'{row_data["DA"]:.2f}%'
            if 'DA' in best_vals and row_data['DA'] == best_vals['DA']:
                td_da['style'] = 'background:#d4edda'
            tr.append(td_da)
            tbody.append(tr)

    # 5) Actualizar metricsData en el script para incluir DA y orden correcto
    script_tag = soup_meta.find('script', string=lambda s: s and 'metricsData' in s)
    if script_tag and meta_rows_data:
        js_content = script_tag.string
        # Construir metricsData ordenada con DA
        ordered_metrics_js = []
        for rd in meta_rows_data:
            ordered_metrics_js.append({
                'Modelo': rd['name'],
                'Color': rd['color'],
                'MSE': rd['MSE'],
                'RMSE': rd['RMSE'],
                'MAE': rd['MAE'],
                'R2': rd['R2'],
                'DA': rd['DA']
            })
        new_metrics_json = json.dumps(ordered_metrics_js)
        # Reemplazar la línea const metricsData=... en el JS
        import re
        js_content = re.sub(
            r'const metricsData=.*?;',
            f'const metricsData={new_metrics_json};',
            js_content,
            count=1
        )
        script_tag.string = js_content

    with open(html_path_meta, 'w', encoding='utf-8') as f:
        f.write(str(soup_meta))
    print(f'[META TABLE + CHARTS] Tabla y gráficas de métricas actualizadas en {html_path_meta}')

# ===== PRUEBA DE DIEBOLD-MARIANO (ESCALA USD) =====
def check_dm_assumptions(d: np.ndarray, name: str) -> None:
    """Verifica estacionariedad y autocorrelación del diferencial de pérdida."""
    # 1. ADF (Estacionariedad)
    adf_res = adfuller(d, autolag='AIC')
    if adf_res[1] > 0.05:
        logging.warning(f"[DM:{name}] d_t no estacionaria (ADF p={adf_res[1]:.3f})")
    # 2. Ljung-Box (Autocorrelación)
    h = int(np.floor(len(d)**(1/3)))
    lb_res = acorr_ljungbox(d, lags=[h], return_df=True)
    if lb_res['lb_pvalue'].iloc[0] < 0.05:
        logging.info(f"[DM:{name}] Autocorrelación detectada (Ljung-Box p={lb_res['lb_pvalue'].iloc[0]:.3f}). Uso de HAC confirmado.")

def dm_test(d: np.ndarray) -> tuple[float, float]:
    """Prueba DM con corrección HAC (Heteroskedasticity and Autocorrelation Consistent)."""
    T = len(d)
    h = int(np.floor(T ** (1/3)))
    d_bar = d.mean()
    # OLS sobre constante para obtener varianza HAC
    model = OLS(d, np.ones(T)).fit()
    hac_var = cov_hac(model, nlags=h).item()
    dm_stat = d_bar / np.sqrt(hac_var / T)
    p_value = 2 * (1 - t_dist.cdf(abs(dm_stat), df=T-1))
    return dm_stat, p_value

logging.info("[DM] Iniciando pruebas Diebold-Mariano sobre 9 meta modelos (36 pares)...")

# Validar que MT existe en preds_p
if 'MT' not in preds_p:
    raise KeyError('Modelo "Meta LSTM (Ensamble Actual)" (key=MT) no encontrado en preds_p')

# 5A: target_metas ampliado con nuevos keys
target_metas = ['MT', 'AB', 'SM', 'XGB_META_EXT',
                'SA', 'WA', 'RD', 'LS', 'EN']

# Generar todos los pares únicos C(9,2) = 36
dm_all_pairs = list(combinations(target_metas, 2))

dm_results = []
for (ki, kj) in dm_all_pairs:
    if ki in preds_p and kj in preds_p:
        pi, pj = preds_p[ki], preds_p[kj]
        # Alinear por índices no nulos comunes
        mask = (~np.isnan(pi)) & (~np.isnan(pj)) & (~np.isnan(pr_r))
        if mask.sum() > 30:  # Mínimo de muestras para validez estadística
            d = (pr_r[mask] - pi[mask])**2 - (pr_r[mask] - pj[mask])**2
            check_dm_assumptions(d, f"{ki} vs {kj}")
            stat, pval = dm_test(d)
            if pval < 0.05:
                better = MDL[kj][1] if stat > 0 else MDL[ki][1]
            else:
                better = '—'   # Sin diferencia significativa
            dm_results.append({
                'key_a': ki, 'key_b': kj,
                'model_a': MDL[ki][1], 'model_b': MDL[kj][1],
                'stat': stat, 'p_value': pval, 'sig': pval < 0.05,
                'better': better
            })

# 5B: Organizar en bloques
actual_keys = {'MT', 'SA', 'WA', 'RD', 'LS', 'EN'}
bloque1 = []  # Ensamble Actual vs Ensamble Actual
bloque2 = []  # Ensamble Actual vs AB
bloque3 = []  # Ensamble Actual vs SM
bloque4 = []  # Ensamble Actual vs XGB_META_EXT (Parker)
bloque5 = []  # Comparaciones cruzadas restantes

for r in dm_results:
    ka, kb = r['key_a'], r['key_b']
    if ka in actual_keys and kb in actual_keys:
        bloque1.append(r)
    elif (ka in actual_keys and kb == 'AB') or (ka == 'AB' and kb in actual_keys):
        bloque2.append(r)
    elif (ka in actual_keys and kb == 'SM') or (ka == 'SM' and kb in actual_keys):
        bloque3.append(r)
    elif (ka in actual_keys and kb == 'XGB_META_EXT') or (ka == 'XGB_META_EXT' and kb in actual_keys):
        bloque4.append(r)
    else:
        bloque5.append(r)

bloques = [
    ('Bloque 1: Ensamble Actual vs Ensamble Actual', bloque1),
    ('Bloque 2: Ensamble Actual vs Ours Sin TimeXer', bloque2),
    ('Bloque 3: Ensamble Actual vs Yu et al.', bloque3),
    ('Bloque 4: Ensamble Actual vs Parker et al.', bloque4),
    ('Bloque 5: Comparaciones cruzadas restantes', bloque5),
]

# Inyectar/reemplazar tabla DM en el HTML existente
safe_token = TOKEN.replace('/', '-').replace('^', '').replace('=', '-')
html_path = os.path.join(reports_dir, f'report_compare_{safe_token}.html')
if os.path.exists(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Eliminar tabla DM existente si ya fue inyectada previamente
    for h2 in soup.find_all('h2'):
        if 'Diebold-Mariano' in h2.text:
            dm_card = h2.find_parent('div', class_='card')
            if dm_card:
                dm_card.decompose()
            break

    container = soup.find('div', class_='container')
    if container:
        dm_html = """
        <div class="card">
            <h2>Prueba de Diebold-Mariano (Escala USD: Errores Cuadráticos)</h2>
            <p style="color:#666; font-size:0.9rem; margin-bottom:12px;">H0: Los modelos tienen la misma precisión predictiva. p-valor &lt; 0.05 indica diferencia significativa. C(9,2) = 36 pares.</p>
            <table class="metrics-table">
                <thead><tr>
                    <th>Modelo A</th><th>Modelo B</th>
                    <th>Estadístico DM</th><th>p-valor</th>
                    <th>Significativo</th><th>Mejor Modelo</th>
                </tr></thead>
                <tbody>"""

        for bloque_name, bloque_rows in bloques:
            if not bloque_rows:
                continue
            # Separador de bloque
            dm_html += f'<tr style="background:#f0f0f0"><td colspan="6" style="font-weight:600;font-size:0.9rem;padding:10px 18px;color:#333">{bloque_name}</td></tr>'
            for r in bloque_rows:
                st = f'<strong style="color:#000">{r["stat"]:.4f}</strong>' if r['sig'] else f'{r["stat"]:.4f}'
                pv = f'<strong style="color:#000">{r["p_value"]:.4f}</strong>' if r['sig'] else f'{r["p_value"]:.4f}'
                better_cell = (
                    f'<td><strong>{r["better"]}</strong></td>'
                    if r['sig']
                    else '<td style="color:#999">Sin diferencia significativa</td>'
                )
                dm_html += (
                    f'<tr>'
                    f'<td>{r["model_a"]}</td>'
                    f'<td>{r["model_b"]}</td>'
                    f'<td>{st}</td>'
                    f'<td>{pv}</td>'
                    f'<td>{"SÍ" if r["sig"] else "No"}</td>'
                    f'{better_cell}'
                    f'</tr>'
                )
        dm_html += "</tbody></table></div>"
        container.append(BeautifulSoup(dm_html, 'html.parser'))

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    print(f"[DM] {len(dm_results)} pares DM (de 36 posibles) inyectados exitosamente en {html_path}")

