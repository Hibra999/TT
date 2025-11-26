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
    predict_with_best_params,
    preload_moirai_module,
    clear_module_cache
)
from model.meta_model.lstm_model import collect_oof_predictions, build_oof_dataframe, train_lstm_meta_model, predict_with_meta_model, get_average_weights
from preprocessing.walk_forward import wfrw
from features.tecnical_indicators import TA
from features.top_n import top_k
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
    for col in df_final.columns:
        col_range = df_final[col].max() - df_final[col].min()
        if col_range > 1e-8:
            df_final[col] = (df_final[col] - df_final[col].min()) / col_range
        else:
            df_final = df_final.drop(columns=[col])
            st.warning(f"Columna {col} eliminada (valores constantes)")
    log_close = log_close.iloc[1:].reset_index(drop=True)
    log_range = log_close.max() - log_close.min()
    if log_range > 1e-8:
        log_close_normalized = (log_close - log_close.min()) / log_range
    else:
        st.error("log_close tiene valores constantes")
        st.stop()
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
    print(device)
    n_samples = len(y)
    oof_lgb, oof_cb, oof_tx, oof_moirai = {}, {}, {}, {}
    st.subheader("lgb")
    splitter = wfrw(y, k=5, fh_val=30)
    with st.spinner('optimizando lgb'):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_global(trial, X, y, splitter, oof_storage=oof_lgb), n_trials=300, n_jobs=-1)
    best_params = study.best_params
    st.json(best_params)
    st.write(f"Mejor MAE Promedio Global: {study.best_value:.4f}")
    st.subheader("CatBoost")
    with st.spinner('optimizando catboost'):
        study_cb = optuna.create_study(direction="minimize")
        study_cb.optimize(lambda trial: objective_catboost_global(trial, X, y, splitter, oof_storage=oof_cb), n_trials=150, n_jobs=1)
    best_params_cb = study_cb.best_params
    st.write("CatBoost hiper:")
    st.json(best_params_cb)
    st.write(f"Mejor MAE Promedio Global: {study_cb.best_value:.4f}")
    st.subheader("timexer")
    splitter = wfrw(y, k=5, fh_val=30)
    with st.spinner('optimizando TimeXer'):
        study_tx = optuna.create_study(direction="minimize")
        study_tx.optimize(lambda trial: objective_timexer_global(trial, X, y, splitter, device=device, seq_len=96, pred_len=30, features='MS', pretrained_path=None, freeze_backbone=False, oof_storage=oof_tx), n_trials=150, n_jobs=1)
    best_params_tx = study_tx.best_params
    st.json(best_params_tx)
    st.write(f"Mejor MAE Promedio Global TimeXer: {study_tx.best_value:.4f}")
    st.subheader("Moirai-MoE")
    with st.spinner('Optimizando Moirai-MoE...'):
        st.write(f"Usando dispositivo: {device}")
        preload_moirai_module(model_size='small')
        study_moirai = optuna.create_study(direction="minimize")
        study_moirai.optimize(lambda trial: objective_moirai_moe_global(trial, X, y, splitter, device=device, pred_len=30, model_size='small', freq='D', use_full_train=True, oof_storage=oof_moirai), n_trials=300,jobs=-1)
    best_params_moirai = study_moirai.best_params
    st.write("Mejores hiperparametros Moirai-MoE:")
    st.json(best_params_moirai)
    st.write(f"Mejor MAE: {study_moirai.best_value:.4f}")
    st.subheader("Meta-Modelo LSTM Stacking")
    oof_df = build_oof_dataframe(oof_lgb, oof_cb, oof_tx, oof_moirai, n_samples)
    st.write(f"OOF shape: {oof_df.shape}")
    st.write(f"Valores validos por modelo: lgb={oof_df['lgb'].notna().sum()}, catboost={oof_df['catboost'].notna().sum()}, timexer={oof_df['timexer'].notna().sum()}, moirai={oof_df['moirai'].notna().sum()}")
    col1, col2 = st.columns(2)
    with col1:
        window_size = st.slider("Ventana temporal (T)", min_value=5, max_value=30, value=10)
        hidden_size = st.slider("Hidden size LSTM", min_value=16, max_value=128, value=64)
    with col2:
        num_layers = st.slider("Num capas LSTM", min_value=1, max_value=4, value=2)
        meta_epochs = st.slider("Epochs meta-modelo", min_value=50, max_value=300, value=100)
    with st.spinner('Entrenando Meta-Modelo LSTM'):
        meta_model, mae_meta, results, device = train_lstm_meta_model(oof_df, y, window_size=window_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.1, lr=1e-3, epochs=meta_epochs, batch_size=32, patience=15, device=device)
    if meta_model is not None:
        st.success(f"Meta-Modelo LSTM entrenado exitosamente")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE Meta-Modelo", f"{results['mae']:.4f}")
        with col2:
            st.metric("RMSE Meta-Modelo", f"{results['rmse']:.4f}")
        with col3:
            st.metric("Mejor Epoch", results['best_epoch'])
        st.subheader("Curvas de Entrenamiento")
        loss_df = pd.DataFrame({'Epoch': range(1, len(results['train_losses'])+1), 'Train Loss': results['train_losses'], 'Val Loss': results['val_losses']})
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=loss_df['Epoch'], y=loss_df['Train Loss'], name='Train', line=dict(color='blue')))
        fig_loss.add_trace(go.Scatter(x=loss_df['Epoch'], y=loss_df['Val Loss'], name='Validation', line=dict(color='orange')))
        fig_loss.update_layout(title="Loss durante entrenamiento", xaxis_title="Epoch", yaxis_title="MSE Loss", height=300)
        st.plotly_chart(fig_loss, width='stretch')
        st.subheader("Pesos Promedio por Modelo Base")
        model_names = ['lgb', 'catboost', 'timexer', 'moirai']
        weights_df = get_average_weights(results['weights'], model_names)
        fig_weights = px.bar(weights_df, x='Modelo', y='Peso_Promedio', title='Peso promedio asignado por LSTM')
        st.plotly_chart(fig_weights, width='stretch')
        st.subheader("Evolucion de Pesos en el Tiempo")
        weights_evolution = pd.DataFrame(results['weights'], columns=model_names)
        weights_evolution['index'] = results['valid_indices']
        fig_evo = go.Figure()
        for col in model_names:
            fig_evo.add_trace(go.Scatter(x=weights_evolution['index'], y=weights_evolution[col], name=col, mode='lines'))
        fig_evo.update_layout(title="Pesos dinamicos alpha_t por modelo", xaxis_title="Indice temporal", yaxis_title="Peso (softmax)", height=400)
        st.plotly_chart(fig_evo, width='stretch')
        st.subheader("Predicciones vs Real")
        pred_df = pd.DataFrame({'Indice': results['valid_indices'], 'Real': results['targets'], 'Prediccion_Meta': results['predictions']})
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=pred_df['Indice'], y=pred_df['Real'], name='Real', line=dict(color='blue')))
        fig_pred.add_trace(go.Scatter(x=pred_df['Indice'], y=pred_df['Prediccion_Meta'], name='Meta LSTM', line=dict(color='red', dash='dash')))
        fig_pred.update_layout(title="Prediccion del Ensemble vs Valor Real", height=400)
        st.plotly_chart(fig_pred, width='stretch')
        st.subheader("Predicciones OOF de Modelos Base")
        st.dataframe(oof_df.head(50))
        st.subheader("Comparacion MAE")
        comparison_df = pd.DataFrame({'Modelo': ['LGB', 'CatBoost', 'TimeXer', 'Moirai-MoE', 'Meta LSTM'], 'MAE': [study.best_value, study_cb.best_value, study_tx.best_value, study_moirai.best_value, results['mae']]})
        fig_comparison = px.bar(comparison_df, x='Modelo', y='MAE', title='Comparacion de MAE entre modelos', color='Modelo')
        st.plotly_chart(fig_comparison, width='stretch')
        if results['mae'] < min(study.best_value, study_cb.best_value, study_tx.best_value, study_moirai.best_value):
            st.success("El Meta-Modelo LSTM supera a todos los modelos base individuales")
        else:
            st.info("El Meta-Modelo LSTM no supera al mejor modelo base individual")
    else:
        st.error("No hay suficientes datos validos para entrenar el meta-modelo LSTM. Se requieren al menos 20 muestras con predicciones de todos los modelos.")
        st.subheader("Predicciones OOF disponibles")
        st.dataframe(oof_df.head(50))
        
    


 