import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import warnings
import streamlit as st
import matplotlib.pyplot as plt
import optuna
from data.yfinance_data import download_yf
from data.ccxt_data import download_cx
from features.macroeconomics import macroeconomicos
from model.bases_models.ligthGBM_model import objective_global
from model.bases_models.catboost_model import objective_catboost_global
from model.bases_models.timexer_model import objective_timexer_global
from model.bases_models.moraiMOE_model import (
    objective_moirai_moe_global,
    preload_moirai_module,
)
from model.meta_model.lstm_model import optimize_lstm_meta, get_average_weights
from preprocessing.oof_generators import collect_oof_predictions, build_oof_dataframe
from preprocessing.walk_forward import wfrw
from features.tecnical_indicators import TA
from features.top_n import top_k
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import torch

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

with st.sidebar:
    start = "2020-01-01"
    end = "2025-10-31"

    @st.cache_data
    def load_data():
        tokens = ['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC', "GC=F", "CBOE"]
        dy = download_yf(tokens, start, end)
        cryptos = ["BTC/USDT", "ETH/USDT"]
        dc = download_cx(cryptos, start, end)
        return dy, dc

    load_data()
    token = st.selectbox(label="ACTIVO FINANCIERO: ", options=['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC', "BTC-USDT", "ETH-USDT"])

st.title('TT')

df = pd.read_csv(rf"C:\Users\hibra\Desktop\TT\data\tokens\{token}_2020-2025.csv")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Datos & Retornos", "Caracteristicas (TA/Macro)", "MICFS", "Walk Folward", "BaseModelsTrain"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CLOSE")
        st.line_chart(df["Close"])
        st.dataframe(df.tail())

    with col2:
        st.subheader("LOG RETURN")
        log_close = np.log(df["Close"] / df["Close"].shift(-1)).dropna()
        log_close_normalized = (log_close - log_close.min()) / (log_close.max() - log_close.min())
        st.line_chart(log_close)
        st.subheader("Normalizado")
        st.line_chart(log_close_normalized)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Indicadores Tecnicos")
        df_ta = TA(df)
        st.dataframe(df_ta.tail())
    with col2:
        st.subheader("Datos Macroeconomicos")
        df_ma = macroeconomicos(df["Date_final"])
        st.dataframe(df_ma.tail())

with tab3:
    df_ta = df_ta.reset_index(drop=True)
    df_ma = df_ma.reset_index(drop=True)
    st.subheader("DF_final")
    df_final = pd.concat([df_ta, df_ma], axis=1)
    df_final = df_final.iloc[1:]
    cols_to_drop = []
    for col in df_final.columns:
        if df_final[col].max() - df_final[col].min() < 1e-8:
            cols_to_drop.append(col)
            st.warning(f"Columna {col} eliminada (valores constantes)")
    df_final = df_final.drop(columns=cols_to_drop)
    scaler_features = MinMaxScaler()
    df_final = pd.DataFrame(
        scaler_features.fit_transform(df_final),
        columns=df_final.columns,
        index=df_final.index
    )
    log_close = log_close
    log_range = log_close.max() - log_close.min()
    if log_range < 1e-8:
        st.error("log_close tiene valores constantes")
        st.stop()
    scaler_target = MinMaxScaler()
    log_close_normalized = pd.Series(
        scaler_target.fit_transform(log_close.values.reshape(-1, 1)).flatten(),
        index=log_close.index,
        name='log_close'
    )
    df_final = df_final.replace([np.inf, -np.inf], 0.0)
    log_close_normalized = log_close_normalized.replace([np.inf, -np.inf], 0.0)
    st.dataframe(df_final.tail())
    st.subheader("MIC: top n caracteristicas")
    features, valores_mic = top_k(df_final, log_close_normalized, 15)
    df_importance = pd.DataFrame(list(valores_mic.items()), columns=['Feature', 'Score'])
    fig = px.bar(df_importance, x='Score', y='Feature', orientation='h', title='MIC')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, width='stretch')
    X = df_final[features].reset_index(drop=True)
    y = log_close_normalized.reset_index(drop=True)
    st.dataframe(X)

with tab4:
    n = len(log_close)
    train_size = int(n * 0.9)
    y_train = log_close.iloc[:train_size]
    y_test  = log_close.iloc[train_size:]
    splitter = wfrw(y_train, k=5, fh_val=30)
    #Grafica de train y test cesaron
    fig_tt = go.Figure()
    fig_tt.add_trace(go.Scatter(x=y_train.index, y=y_train, name='train', line_color='blue'))
    fig_tt.add_trace(go.Scatter(x=y_test.index, y=y_test, name='test', line_color='orange'))
    fig_tt.update_layout(title="Train y test", height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_tt, width='stretch')
    #Grafica de wf
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, (t_idx, v_idx) in enumerate(wfrw(y_train, k=5, fh_val=30).split(y_train)):
        fig.add_trace(go.Scatter(x=y_train.index[t_idx], y=y_train.iloc[t_idx], line_color='blue'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=y_train.index[v_idx], y=y_train.iloc[v_idx], line_color='red'), row=i+1, col=1)
    fig.update_layout(height=800, showlegend=False, title="Folds", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width='stretch')

with tab5:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Device: {device}")
    n_samples = len(y)
    # Inicializar storages para OOF
    oof_lgb, oof_cb, oof_tx, oof_moirai = {}, {}, {}, {}
    # ========================================
    # FASE 1: OPTIMIZACION DE HIPERPARAMETROS
    # ========================================
    st.subheader("Fase 1: Optimizacion de Hiperparametros")
    splitter = wfrw(y, k=5, fh_val=30)
    st.write("Optimizando LightGBM...")
    with st.spinner('Optimizando LGB'):
        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(lambda trial: objective_global(trial, X, y, splitter, oof_storage=oof_lgb), n_trials=30, n_jobs=-1)
    best_params_lgb = study_lgb.best_params
    st.json(best_params_lgb)
    st.write(f"Mejor MAE LGB: {study_lgb.best_value:.4f}")
    splitter = wfrw(y, k=5, fh_val=30)
    st.write("Optimizando CatBoost...")
    with st.spinner('Optimizando CatBoost'):
        study_cb = optuna.create_study(direction="minimize")
        study_cb.optimize(lambda trial: objective_catboost_global(trial, X, y, splitter, oof_storage=oof_cb), n_trials=30, n_jobs=1)
    best_params_cb = study_cb.best_params
    st.json(best_params_cb)
    st.write(f"Mejor MAE CatBoost: {study_cb.best_value:.4f}")
    splitter = wfrw(y, k=5, fh_val=30)
    st.write("Optimizando TimeXer...")
    with st.spinner('Optimizando TimeXer'):
        study_tx = optuna.create_study(direction="minimize")
        study_tx.optimize(lambda trial: objective_timexer_global(trial, X, y, splitter, device=device, seq_len=96, pred_len=30, features='MS', oof_storage=oof_tx), n_trials=30, n_jobs=1)
    best_params_tx = study_tx.best_params
    st.json(best_params_tx)
    st.write(f"Mejor MAE TimeXer: {study_tx.best_value:.4f}")
    splitter = wfrw(y, k=5, fh_val=30)
    st.write("Optimizando Moirai-MoE...")
    with st.spinner('Optimizando Moirai-MoE'):
        preload_moirai_module(model_size='small')
        study_moirai = optuna.create_study(direction="minimize")
        study_moirai.optimize(lambda trial: objective_moirai_moe_global(trial, X, y, splitter, device=device, pred_len=30, model_size='small', freq='D', use_full_train=True, oof_storage=oof_moirai), n_trials=30, n_jobs=1)
    best_params_moirai = study_moirai.best_params
    st.json(best_params_moirai)
    st.write(f"Mejor MAE Moirai: {study_moirai.best_value:.4f}")
    # ========================================
    # FASE 2: CONSTRUIR MATRIZ OOF
    # ========================================
    st.subheader("Fase 2: Matriz OOF")
    # Mostrar predicciones recolectadas
    preds_lgb, idx_lgb, _ = collect_oof_predictions(oof_lgb)
    preds_cb, idx_cb, _ = collect_oof_predictions(oof_cb)
    preds_tx, idx_tx, _ = collect_oof_predictions(oof_tx)
    preds_moirai, idx_moirai, _ = collect_oof_predictions(oof_moirai)
    st.write(f"Predicciones recolectadas:")
    st.write(f"  - LGB: {len(preds_lgb)}")
    st.write(f"  - CatBoost: {len(preds_cb)}")
    st.write(f"  - TimeXer: {len(preds_tx)}")
    st.write(f"  - Moirai: {len(preds_moirai)}")
    # Construir matriz OOF
    oof_df = build_oof_dataframe(oof_lgb, oof_cb, oof_tx, oof_moirai, y)
    st.write(f"Matriz OOF (inner join) shape: {oof_df.shape}")
    st.write(f"Columnas: {list(oof_df.columns)}")
    st.dataframe(oof_df.head(30))
    # ========================================
    # FASE 3: META-MODELO LSTM CON OPTUNA
    # ========================================
    st.subheader("Fase 3: Meta-Modelo LSTM")
    if len(oof_df) < 50:
        st.warning(f"Solo hay {len(oof_df)} filas en la matriz OOF. Se recomienda al menos 50.")
    n_trials_lstm = st.slider("Trials para optimizar LSTM", min_value=10, max_value=100, value=30)
    if st.button("Optimizar y Entrenar Meta-Modelo"):
        with st.spinner('Optimizando Meta-Modelo LSTM con Optuna...'):
            meta_model, mae_meta, results, best_params_lstm, study_lstm = optimize_lstm_meta(oof_df, device, n_trials=n_trials_lstm)
        if meta_model is not None:
            st.success("Meta-Modelo LSTM entrenado exitosamente")
            st.write("Mejores hiperparametros LSTM:")
            st.json(best_params_lstm)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE Meta-Modelo", f"{results['mae']:.4f}")
            with col2:
                st.metric("RMSE Meta-Modelo", f"{results['rmse']:.4f}")
            with col3:
                st.metric("Mejor Epoch", results['best_epoch'])
            # Curvas de entrenamiento
            st.subheader("Curvas de Entrenamiento")
            loss_df = pd.DataFrame({
                'Epoch': range(1, len(results['train_losses'])+1),
                'Train Loss': results['train_losses'],
                'Val Loss': results['val_losses']
            })
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=loss_df['Epoch'], y=loss_df['Train Loss'], name='Train', line=dict(color='blue')))
            fig_loss.add_trace(go.Scatter(x=loss_df['Epoch'], y=loss_df['Val Loss'], name='Validation', line=dict(color='orange')))
            fig_loss.update_layout(title="Loss durante entrenamiento", xaxis_title="Epoch", yaxis_title="MSE Loss", height=300)
            st.plotly_chart(fig_loss, key="loss_chart")
            # Pesos promedio
            st.subheader("Pesos Promedio por Modelo Base")
            weights_df = get_average_weights(results['weights'], results['model_names'])
            fig_weights = px.bar(weights_df, x='Modelo', y='Peso_Promedio', title='Peso promedio asignado por LSTM')
            st.plotly_chart(fig_weights, key="weights_chart")
            # Evolucion de pesos
            st.subheader("Evolucion de Pesos en el Tiempo")
            weights_evolution = pd.DataFrame(results['weights'], columns=results['model_names'])
            weights_evolution['index'] = results['valid_indices']
            fig_evo = go.Figure()
            for col in results['model_names']:
                fig_evo.add_trace(go.Scatter(x=weights_evolution['index'], y=weights_evolution[col], name=col, mode='lines'))
            fig_evo.update_layout(title="Pesos dinamicos alpha_t por modelo", xaxis_title="Indice temporal", yaxis_title="Peso (softmax)", height=400)
            st.plotly_chart(fig_evo, key="evo_chart")
            # Predicciones vs Real
            st.subheader("Predicciones vs Real")
            pred_df = pd.DataFrame({
                'Indice': results['valid_indices'],
                'Real': results['targets'],
                'Prediccion_Meta': results['predictions']
            })
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=pred_df['Indice'], y=pred_df['Real'], name='Real', line=dict(color='blue')))
            fig_pred.add_trace(go.Scatter(x=pred_df['Indice'], y=pred_df['Prediccion_Meta'], name='Meta LSTM', line=dict(color='red', dash='dash')))
            fig_pred.update_layout(title="Prediccion del Ensemble vs Valor Real", height=400)
            st.plotly_chart(fig_pred, key="pred_chart")
            # Comparacion MAE
            st.subheader("Comparacion MAE")
            comparison_df = pd.DataFrame({
                'Modelo': ['LGB', 'CatBoost', 'TimeXer', 'Moirai-MoE', 'Meta LSTM'],
                'MAE': [study_lgb.best_value, study_cb.best_value, study_tx.best_value, study_moirai.best_value, results['mae']]
            })
            fig_comparison = px.bar(comparison_df, x='Modelo', y='MAE', title='Comparacion de MAE entre modelos', color='Modelo')
            st.plotly_chart(fig_comparison, key="comparison_chart")
            best_base = min(study_lgb.best_value, study_cb.best_value, study_tx.best_value, study_moirai.best_value)
            if results['mae'] < best_base:
                st.success("El Meta-Modelo LSTM supera a todos los modelos base")
            else:
                st.info("El Meta-Modelo no supera al mejor modelo base")
        else:
            st.error("No hay suficientes datos para entrenar el meta-modelo. Se requieren al menos 20 muestras validas.")
            st.write("Intenta con mas trials o ajusta los parametros de los modelos base.")
        
    


 