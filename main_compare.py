import pandas as pd; import numpy as np; import optuna; import warnings; import os; import logging
from model.meta_model.evaluation import run_pairwise_dm_tests
from reports.report_utils import generate_compare_report, enhance_report_with_metrics, inject_dm_results
from data.yfinance_data import download_yf; from data.ccxt_data import download_cx; from features.macroeconomics import macroeconomicos

# Modelos actuales
from model.bases_models.ligthGBM_model import objective_global, train_final_and_predict_test as lgb_predict_test
from model.bases_models.catboost_model import objective_catboost_global, train_final_and_predict_test as cb_predict_test
from model.bases_models.timexer_model import objective_timexer_global, train_final_and_predict_test as tx_predict_test
from model.bases_models.moraiMOE_model import objective_moirai_moe_global, preload_moirai_module, train_final_and_predict_test as moirai_predict_test
from model.meta_model.lstm_model import optimize_lstm_meta, get_average_weights
from model.meta_model.simple_avg import train_and_predict as sa_predict
from model.meta_model.weighted_avg import train_and_predict as wa_predict
from model.meta_model.ridge_model import train_and_predict as rd_predict
from model.meta_model.lasso_model import train_and_predict as ls_predict
from model.meta_model.elasticnet_model import train_and_predict as en_predict
from model.meta_model.mlp_model import train_and_predict as mlp_predict
from model.meta_model.gru_model import train_and_predict as gru_predict
from model.meta_model.transformer_model import train_and_predict as trans_predict
from model.meta_model.lgbm_meta import train_and_predict as lgbm_meta_predict
from model.meta_model.rf_meta import train_and_predict as rf_meta_predict
from preprocessing.oof_generators import build_oof_dataframe, collect_oof_predictions, build_oof_dataframe_refit


# Modelos SOTA
from model.sota.stacking_ensemble import (
    objective_base_lstm_global, train_final_base_lstm,
    build_oof_dataframe_sota, optimize_stacking_meta
)

from preprocessing.walk_forward import wfrw; from features.tecnical_indicators import TA; from features.top_n import top_k
from sklearn.preprocessing import MinMaxScaler; import torch
from sklearn.metrics import mean_squared_error

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
    'NC':  ('#e7298a', 'Ours (Sin CatBoost)'),
    'SM':  ('#d62728', 'Yu et al. [44] 2025'),
    'LSTM_EXT': ('#ff1493', 'LSTM (Externo)'),
    'GRU_EXT': ('#00ced1', 'GRU (Externo)'),
    'ARIMA_EXT': ('#ffd700', 'ARIMA (Externo)'),
    'RF_EXT': ('#32cd32', 'Random Forest (Externo)'),
    'TRANS_EXT': ('#8a2be2', 'Transformer (Externo)'),
    'XGB_META_EXT': ('#ff4500', 'Parker et al. 2025'),
    'LGB_META':   ('#1a9850', 'LightGBM Meta (Ensamble Actual)'),
    'RF_META':    ('#74add1', 'Random Forest (Ensamble Actual)'),
    'MLP_META':   ('#f46d43', 'MLP (Ensamble Actual)'),
    'GRU_META':   ('#542788', 'GRU (Ensamble Actual)'),
    'TRANS_META': ('#bf812d', 'Transformer (Ensamble Actual)'),
}

# ===== CONFIG =====
TOKEN = '^GSPC'
# TOKEN = 'BTC/USDT'
N_LGB, N_CB = 10, 10
N_TX, N_MO = 10, 10
N_BL = 10 
N_MT, N_AB, N_NC, N_SM = 10 , 10, 10, 10
N_LGB_META  = 10   # trials Optuna para LightGBM meta
N_RF_META   = 10   # trials Optuna para Random Forest meta
N_MLP_META  = 10   # trials Optuna para MLP meta
N_GRU_META  = 10   # trials Optuna para GRU meta
N_TRANS_META = 10  # trials Optuna para Transformer meta

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
sp = wfrw(yt, fh_val=30, window_ratio=0.65, step_length=30)

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
print(f'[6/11] Generando OOF con refitting + cross-boundary context...')
oof_df_actual = build_oof_dataframe_refit(
    bp_lgb=bp_l,
    bp_cb=bp_c,
    bp_tx=bp_t,
    bp_moirai=bp_m,
    Xt=Xt, yt=yt, sp=sp,
    device=device,
    lgb_predict_fn=lgb_predict_test,
    cb_predict_fn=cb_predict_test,
    tx_predict_fn=tx_predict_test,
    moirai_predict_fn=moirai_predict_test,
    seq_len_tx=96,
    pred_len_tx=1,
    model_size_mo='small',
    freq_mo='D'
)
print(f'  OOF matrix shape: {oof_df_actual.shape}')
print(f'  OOF columns: {oof_df_actual.columns.tolist()}')
print(f'  OOF NaN count: {oof_df_actual.isna().sum().to_dict()}')
if len(oof_df_actual) < 20:
    print(f'  [ERROR] OOF dataframe too small ({len(oof_df_actual)} samples).')

print(f'  Entrenando Meta LSTM (Ensamble Actual)...')
meta_model_actual, _, _, bp_mt, _ = optimize_lstm_meta(
    oof_df_actual, device, n_trials=N_MT
)
ws_meta_actual = bp_mt.get('window_size', 10) if meta_model_actual is not None else 10
if meta_model_actual is None:
    print(f'  [ERROR] meta_model_actual is None. Training failed.')

# Las ablaciones derivan directamente de oof_df_actual (ya completo)
print(f'[7/11] Entrenando Meta LSTM Ours (Sin TimeXer)...')
oof_df_ablation = oof_df_actual.drop(columns=['timexer'])
print(f'  OOF matrix shape: {oof_df_ablation.shape}')
meta_model_ablation, _, _, bp_ab, _ = optimize_lstm_meta(
    oof_df_ablation, device, n_trials=N_AB
)
ws_meta_ablation = bp_ab.get('window_size', 10) if meta_model_ablation is not None else 10

print(f'[7.5/11] Entrenando Meta LSTM Ours (Sin CatBoost)...')
oof_df_no_cb = oof_df_actual.drop(columns=['catboost'])
print(f'  OOF matrix shape: {oof_df_no_cb.shape}')
meta_model_no_cb, _, _, bp_nc, _ = optimize_lstm_meta(
    oof_df_no_cb, device, n_trials=N_NC
)
ws_meta_no_cb = bp_nc.get('window_size', 10) if meta_model_no_cb is not None else 10

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

X_test_bases = np.column_stack([pl, pc, pt, pm])

pmt_actual   = np.full(len(ye), np.nan)
pmt_ablation = np.full(len(ye), np.nan)
pmt_no_cb    = np.full(len(ye), np.nan)
pmt_sota     = np.full(len(ye), np.nan)

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

if meta_model_no_cb is not None:
    test_matrix_no_cb = np.column_stack([pl, pt, pm]).astype(np.float32)
    meta_model_no_cb.eval()
    with torch.no_grad():
        for i in range(ws_meta_no_cb - 1, len(ye)):
            window = test_matrix_no_cb[i - ws_meta_no_cb + 1:i + 1]
            if not np.isnan(window).any():
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                pmt_no_cb[i] = meta_model_no_cb(x_t).cpu().item()
    print(f'  [NC] {(~np.isnan(pmt_no_cb)).sum()}/{len(ye)} válidas')

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

# ── Paso 2: Primera pasada GRU y Transformer para obtener window_size ──
print('  > GRU Meta - primera pasada...')
_, info_gru_tmp   = gru_predict(oof_df_actual, X_test_bases, n_trials=N_GRU_META, device=device)
ws_gru_meta       = info_gru_tmp.get('window_size', 1)

print('  > Transformer Meta - primera pasada...')
_, info_trans_tmp = trans_predict(oof_df_actual, X_test_bases, n_trials=N_TRANS_META, device=device)
ws_trans_meta     = info_trans_tmp.get('window_size', 1)

# ── Paso 3: Calcular ws_max y construir oof_df_cut (Opción A) ──
ws_max = max(
    ws_meta_actual   if meta_model_actual   is not None else 1,
    ws_meta_ablation if meta_model_ablation is not None else 1,
    ws_meta_no_cb    if meta_model_no_cb    is not None else 1,
    ws_meta_sota     if meta_model_sota     is not None else 1,
    ws_gru_meta,
    ws_trans_meta,
)
n_ventanas_validas = len(oof_df_actual) - (ws_max - 1)
tr_sz_lstm         = int(n_ventanas_validas * 0.8)
oof_df_cut         = oof_df_actual.iloc[
    ws_max - 1 : ws_max - 1 + tr_sz_lstm
].reset_index(drop=True)

print(f'\n  [META ALIGN] ws_lstm={ws_meta_actual}, '
      f'ws_gru={ws_gru_meta}, ws_trans={ws_trans_meta}')
print(f'  [META ALIGN] ws_max={ws_max}')
print(f'  [META ALIGN] OOF total:           {len(oof_df_actual)}')
print(f'  [META ALIGN] Ventanas válidas:    {n_ventanas_validas}')
print(f'  [META ALIGN] 80% train LSTM:      {tr_sz_lstm}')
print(f'  [META ALIGN] oof_df_cut shape:    {oof_df_cut.shape}')

# ── Paso 4: Entrenar todos los meta-learners con oof_df_cut ──
print('\n  > Meta modelos lineales y promedio (Ensamble Actual)...')
oof_cols = ['lgb', 'catboost', 'timexer', 'moirai']
assert set(oof_cols).issubset(oof_df_cut.columns), \
    f"[ERROR] oof_df_cut no tiene las columnas esperadas: {oof_cols}"

pmt_simple_avg,   _           = sa_predict(oof_df_cut, X_test_bases, n_trials=N_MT, device=device)
pmt_weighted_avg, info_wa     = wa_predict(oof_df_cut, X_test_bases, n_trials=N_MT, device=device)
pmt_ridge,        info_rd     = rd_predict(oof_df_cut, X_test_bases, n_trials=N_MT, device=device)
pmt_lasso,        info_ls     = ls_predict(oof_df_cut, X_test_bases, n_trials=N_MT, device=device)
pmt_elasticnet,   info_en     = en_predict(oof_df_cut, X_test_bases, n_trials=N_MT, device=device)

print(f'  [WA] Pesos: {info_wa.get("weights", {})}')
print(f'  [RD] alpha={info_rd.get("alpha", "N/A")}')
print(f'  [LS] alpha={info_ls.get("alpha", "N/A")}')
print(f'  [EN] alpha={info_en.get("alpha", "N/A")}, '
      f'l1_ratio={info_en.get("l1_ratio", "N/A")}')

print('  > LightGBM Meta (Ensamble Actual)...')
pmt_lgb_meta,   info_lgb_meta = lgbm_meta_predict(oof_df_cut, X_test_bases, n_trials=N_LGB_META, device=device)
print('  > Random Forest Meta (Ensamble Actual)...')
pmt_rf_meta,    info_rf_meta  = rf_meta_predict(oof_df_cut, X_test_bases, n_trials=N_RF_META, device=device)
print(f'  [LGB_META] n_estimators={info_lgb_meta.get("n_estimators","N/A")}, '
      f'lr={info_lgb_meta.get("learning_rate","N/A")}')
print(f'  [RF_META]  n_estimators={info_rf_meta.get("n_estimators","N/A")}, '
      f'max_depth={info_rf_meta.get("max_depth","N/A")}')

print('  > MLP Meta (Ensamble Actual)...')
pmt_mlp_meta,   info_mlp      = mlp_predict(oof_df_cut, X_test_bases, n_trials=N_MLP_META, device=device)

print('  > GRU Meta - entrenamiento final (Ensamble Actual)...')
pmt_gru_meta,   info_gru      = gru_predict(oof_df_cut, X_test_bases, n_trials=N_GRU_META, device=device)

print('  > Transformer Meta - entrenamiento final (Ensamble Actual)...')
pmt_trans_meta, info_trans    = trans_predict(oof_df_cut, X_test_bases, n_trials=N_TRANS_META, device=device)

print(f'  [MLP]   hidden={info_mlp.get("hidden_size","N/A")}, '
      f'layers={info_mlp.get("n_layers","N/A")}')
print(f'  [GRU]   window={info_gru.get("window_size","N/A")}, '
      f'hidden={info_gru.get("hidden_size","N/A")}')
print(f'  [TRANS] window={info_trans.get("window_size","N/A")}, '
      f'd_model={info_trans.get("d_model","N/A")}, '
      f'nhead={info_trans.get("nhead","N/A")}')

# ── Paso 5: Alinear cobertura de test ──
effective_start = ws_max - 1
for arr in [
    pmt_actual, pmt_ablation, pmt_no_cb, pmt_sota,
    pmt_simple_avg, pmt_weighted_avg,
    pmt_ridge, pmt_lasso, pmt_elasticnet,
    pmt_lgb_meta, pmt_rf_meta,
    pmt_mlp_meta, pmt_gru_meta, pmt_trans_meta
]:
    arr[:effective_start] = np.nan

print(f'\n  [META ALIGN] effective_start={effective_start} | '
      f'observaciones comparables en test: '
      f'{len(ye) - effective_start}/{len(ye)}')


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
    'LGB_META':   safe_inv_recon(pmt_lgb_meta),
    'RF_META':    safe_inv_recon(pmt_rf_meta),
    'MLP_META':   safe_inv_recon(pmt_mlp_meta),
    'GRU_META':   safe_inv_recon(pmt_gru_meta),
    'TRANS_META': safe_inv_recon(pmt_trans_meta),
    'AB':  safe_inv_recon(pmt_ablation),
    'NC':  safe_inv_recon(pmt_no_cb),
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
            mp.append({'Modelo': MDL[km][1], 'Color': MDL[km][0], **met(y_valid, v_valid), 'DA': round(da_val, 2)})

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
                  'LGB_META': pmt_lgb_meta, 'RF_META': pmt_rf_meta,
                  'MLP_META': pmt_mlp_meta, 'GRU_META': pmt_gru_meta,
                  'TRANS_META': pmt_trans_meta,
                  'AB': pmt_ablation, 'NC': pmt_no_cb, 'SM': pmt_sota, 'XGB_META_EXT': pmt_parker}
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

# Carpeta de reportes (se crea si no existe)
reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(reports_dir, exist_ok=True)


# 5. Generación de Reporte y Evaluación Estadística
target_metas = ['MT', 'NC', 'AB', 'SM', 'XGB_META_EXT',
                'SA', 'WA', 'RD', 'LS', 'EN',
                'LGB_META', 'RF_META', 'MLP_META',
                'GRU_META', 'TRANS_META']

report_path = generate_compare_report(
    TOKEN, cp, gi_v, pr_r, preds_p, mp, MDL, zs, ze, reports_dir, 
    ye_vals=yv, meta_raw_preds=meta_raw_preds, 
    base_raw_preds=base_raw_preds, met_fn=met, da_fn=directional_accuracy
)

enhance_report_with_metrics(report_path, preds_p, MDL, pr_r, met, directional_accuracy)

dm_results, dm_blocks = run_pairwise_dm_tests(preds_p, MDL, pr_r, target_metas)
inject_dm_results(report_path, dm_blocks)

print(f'\n[PIPELINE] Reporte comparativo generado en: {report_path}')
print(f'[PIPELINE] Pruebas Diebold-Mariano inyectadas ({len(dm_results)} pares).')
