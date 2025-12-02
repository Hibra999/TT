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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    k = 5
    y_train = log_close.iloc[:train_size]
    y_test  = log_close.iloc[train_size:]
    splitter = wfrw(y_train, k=k, fh_val=30)
    fig_tt = go.Figure()
    fig_tt.add_trace(go.Scatter(x=y_train.index, y=y_train, name='train', line_color='blue'))
    fig_tt.add_trace(go.Scatter(x=y_test.index, y=y_test, name='test', line_color='orange'))
    fig_tt.update_layout(title="Train y test", height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_tt, width='stretch')
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, (t_idx, v_idx) in enumerate(wfrw(y_train, k=k, fh_val=30).split(y_train)):
        fig.add_trace(go.Scatter(x=y_train.index[t_idx], y=y_train.iloc[t_idx], line_color='blue'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=y_train.index[v_idx], y=y_train.iloc[v_idx], line_color='red'), row=i+1, col=1)
    fig.update_layout(height=800, showlegend=False, title="Folds", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width='stretch')
with tab5:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Device: {device}")
    n_samples = len(y)
    oof_lgb, oof_cb, oof_tx, oof_moirai = {}, {}, {}, {}
    st.subheader("Fase 1: Optimizacion de Hiperparametros")
    st.write("Optimizando LightGBM...")
    with st.spinner('Optimizando LGB'):
        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(lambda trial: objective_global(trial, X, y, splitter, oof_storage=oof_lgb), n_trials=5, n_jobs=-1)
    best_params_lgb = study_lgb.best_params
    st.json(best_params_lgb)
    st.write(f"Mejor MAE LGB: {study_lgb.best_value:.4f}")
    st.write("Optimizando CatBoost...")
    with st.spinner('Optimizando CatBoost'):
        study_cb = optuna.create_study(direction="minimize")
        study_cb.optimize(lambda trial: objective_catboost_global(trial, X, y, splitter, oof_storage=oof_cb), n_trials=5, n_jobs=1)
    best_params_cb = study_cb.best_params
    st.json(best_params_cb)
    st.write(f"Mejor MAE CatBoost: {study_cb.best_value:.4f}")
    st.write("Optimizando TimeXer...")
    with st.spinner('Optimizando TimeXer'):
        study_tx = optuna.create_study(direction="minimize")
        study_tx.optimize(lambda trial: objective_timexer_global(trial, X, y, splitter, device=device, seq_len=96, pred_len=30, features='MS', oof_storage=oof_tx), n_trials=5, n_jobs=1)
    best_params_tx = study_tx.best_params
    st.json(best_params_tx)
    st.write(f"Mejor MAE TimeXer: {study_tx.best_value:.4f}")
    st.write("Optimizando Moirai-MoE...")
    with st.spinner('Optimizando Moirai-MoE'):
        preload_moirai_module(model_size='small')
        study_moirai = optuna.create_study(direction="minimize")
        study_moirai.optimize(lambda trial: objective_moirai_moe_global(trial, X, y, splitter, device=device, pred_len=30, model_size='small', freq='D', use_full_train=True, oof_storage=oof_moirai), n_trials=5, n_jobs=1)
    best_params_moirai = study_moirai.best_params
    st.json(best_params_moirai)
    st.write(f"Mejor MAE Moirai: {study_moirai.best_value:.4f}")
    st.subheader("Fase 2: Matriz OOF")
    preds_lgb, idx_lgb, _ = collect_oof_predictions(oof_lgb)
    preds_cb, idx_cb, _ = collect_oof_predictions(oof_cb)
    preds_tx, idx_tx, _ = collect_oof_predictions(oof_tx)
    preds_moirai, idx_moirai, _ = collect_oof_predictions(oof_moirai)
    st.write(f"Predicciones recolectadas:")
    st.write(f"  - LGB: {len(preds_lgb)}")
    st.write(f"  - CatBoost: {len(preds_cb)}")
    st.write(f"  - TimeXer: {len(preds_tx)}")
    st.write(f"  - Moirai: {len(preds_moirai)}")
    oof_df = build_oof_dataframe(oof_lgb, oof_cb, oof_tx, oof_moirai, y)
    st.write(f"Matriz OOF (inner join) shape: {oof_df.shape}")
    st.write(f"Columnas: {list(oof_df.columns)}")
    st.dataframe(oof_df.head(30))
    def get_column_name(df, keywords):
        for col in df.columns:
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return col
        return None
    col_lgb = get_column_name(oof_df, ['lgb', 'lightgbm', 'light'])
    col_cb = get_column_name(oof_df, ['cb', 'cat', 'catboost'])
    col_tx = get_column_name(oof_df, ['tx', 'timex', 'timexer'])
    col_moirai = get_column_name(oof_df, ['moirai', 'moe'])
    col_target = get_column_name(oof_df, ['target', 'y', 'real'])
    st.write(f"Columnas detectadas: LGB={col_lgb}, CB={col_cb}, TX={col_tx}, Moirai={col_moirai}, Target={col_target}")
    st.subheader("Fase 3: Meta-Modelo LSTM")
    if len(oof_df) < 50:
        st.warning(f"Solo hay {len(oof_df)} filas en la matriz OOF. Se recomienda al menos 50.")
    n_trials_lstm = 3
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
            st.subheader("Curvas de Entrenamiento Completas")
            loss_df = pd.DataFrame({
                'Epoch': range(1, len(results['train_losses'])+1),
                'Train Loss': results['train_losses'],
                'Val Loss': results['val_losses']
            })
            fig_loss, ax_loss = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=loss_df, x='Epoch', y='Train Loss', ax=ax_loss, label='Train', color='blue')
            sns.lineplot(data=loss_df, x='Epoch', y='Val Loss', ax=ax_loss, label='Validation', color='orange')
            ax_loss.set_title('Loss durante entrenamiento')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('MSE Loss')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            st.pyplot(fig_loss)
            plt.close(fig_loss)
            st.subheader("Pesos Promedio por Modelo Base")
            weights_df = get_average_weights(results['weights'], results['model_names'])
            fig_weights, ax_weights = plt.subplots(figsize=(10, 5))
            sns.barplot(data=weights_df, x='Modelo', y='Peso_Promedio', ax=ax_weights, palette='viridis')
            ax_weights.set_title('Peso promedio asignado por LSTM')
            ax_weights.set_xlabel('Modelo')
            ax_weights.set_ylabel('Peso Promedio')
            ax_weights.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig_weights)
            plt.close(fig_weights)
            st.subheader("Evolucion de Pesos en el Tiempo")
            weights_evolution = pd.DataFrame(results['weights'], columns=results['model_names'])
            weights_evolution['index'] = results['valid_indices']
            fig_evo, ax_evo = plt.subplots(figsize=(12, 5))
            for col in results['model_names']:
                sns.lineplot(data=weights_evolution, x='index', y=col, ax=ax_evo, label=col)
            ax_evo.set_title('Pesos dinamicos alpha_t por modelo')
            ax_evo.set_xlabel('Indice temporal')
            ax_evo.set_ylabel('Peso (softmax)')
            ax_evo.legend()
            ax_evo.grid(True, alpha=0.3)
            st.pyplot(fig_evo)
            plt.close(fig_evo)
            st.subheader("Predicciones vs Real (Normalizado) - Todos los Modelos")
            valid_indices = results['valid_indices']
            oof_subset = oof_df.loc[oof_df.index.isin(valid_indices)].copy()
            oof_subset = oof_subset.loc[valid_indices]
            pred_df = pd.DataFrame({'Indice': valid_indices, 'Real': results['targets'], 'Meta_LSTM': results['predictions']})
            if col_lgb:
                pred_df['LGB'] = oof_subset[col_lgb].values
            if col_cb:
                pred_df['CatBoost'] = oof_subset[col_cb].values
            if col_tx:
                pred_df['TimeXer'] = oof_subset[col_tx].values
            if col_moirai:
                pred_df['Moirai'] = oof_subset[col_moirai].values
            fig_pred, ax_pred = plt.subplots(figsize=(14, 6))
            sns.lineplot(data=pred_df, x='Indice', y='Real', ax=ax_pred, label='Real', color='black', linewidth=2)
            sns.lineplot(data=pred_df, x='Indice', y='Meta_LSTM', ax=ax_pred, label='Meta LSTM', color='red', linestyle='--', linewidth=2)
            if 'LGB' in pred_df.columns:
                sns.lineplot(data=pred_df, x='Indice', y='LGB', ax=ax_pred, label='LGB', color='blue', alpha=0.6)
            if 'CatBoost' in pred_df.columns:
                sns.lineplot(data=pred_df, x='Indice', y='CatBoost', ax=ax_pred, label='CatBoost', color='green', alpha=0.6)
            if 'TimeXer' in pred_df.columns:
                sns.lineplot(data=pred_df, x='Indice', y='TimeXer', ax=ax_pred, label='TimeXer', color='purple', alpha=0.6)
            if 'Moirai' in pred_df.columns:
                sns.lineplot(data=pred_df, x='Indice', y='Moirai', ax=ax_pred, label='Moirai', color='orange', alpha=0.6)
            ax_pred.set_title('Predicciones de Todos los Modelos vs Valor Real (Normalizado)')
            ax_pred.set_xlabel('Indice')
            ax_pred.set_ylabel('Valor Normalizado')
            ax_pred.legend(loc='upper right')
            ax_pred.grid(True, alpha=0.3)
            st.pyplot(fig_pred)
            plt.close(fig_pred)
            st.subheader("Predicciones vs Precio Real (Escala Original) - Todos los Modelos")
            predictions_scaled = np.array(results['predictions']).reshape(-1, 1)
            targets_scaled = np.array(results['targets']).reshape(-1, 1)
            predictions_log = scaler_target.inverse_transform(predictions_scaled).flatten()
            targets_log = scaler_target.inverse_transform(targets_scaled).flatten()
            lgb_log, cb_log, tx_log, moirai_log = None, None, None, None
            if col_lgb:
                lgb_scaled = oof_subset[col_lgb].values.reshape(-1, 1)
                lgb_log = scaler_target.inverse_transform(lgb_scaled).flatten()
            if col_cb:
                cb_scaled = oof_subset[col_cb].values.reshape(-1, 1)
                cb_log = scaler_target.inverse_transform(cb_scaled).flatten()
            if col_tx:
                tx_scaled = oof_subset[col_tx].values.reshape(-1, 1)
                tx_log = scaler_target.inverse_transform(tx_scaled).flatten()
            if col_moirai:
                moirai_scaled = oof_subset[col_moirai].values.reshape(-1, 1)
                moirai_log = scaler_target.inverse_transform(moirai_scaled).flatten()
            close_prices = df['Close'].values
            precio_real, precio_meta, precio_lgb, precio_cb, precio_tx, precio_moirai = [], [], [], [], [], []
            indices_validos = []
            for i, idx in enumerate(valid_indices):
                if idx > 0 and idx < len(close_prices):
                    precio_anterior = close_prices[idx - 1]
                    precio_real.append(precio_anterior * np.exp(-targets_log[i]))
                    precio_meta.append(precio_anterior * np.exp(-predictions_log[i]))
                    if lgb_log is not None:
                        precio_lgb.append(precio_anterior * np.exp(-lgb_log[i]))
                    if cb_log is not None:
                        precio_cb.append(precio_anterior * np.exp(-cb_log[i]))
                    if tx_log is not None:
                        precio_tx.append(precio_anterior * np.exp(-tx_log[i]))
                    if moirai_log is not None:
                        precio_moirai.append(precio_anterior * np.exp(-moirai_log[i]))
                    indices_validos.append(idx)
            precio_df = pd.DataFrame({'Indice': indices_validos, 'Precio_Real': precio_real, 'Meta_LSTM': precio_meta})
            if precio_lgb:
                precio_df['LGB'] = precio_lgb
            if precio_cb:
                precio_df['CatBoost'] = precio_cb
            if precio_tx:
                precio_df['TimeXer'] = precio_tx
            if precio_moirai:
                precio_df['Moirai'] = precio_moirai
            fig_precio, ax_precio = plt.subplots(figsize=(14, 6))
            sns.lineplot(data=precio_df, x='Indice', y='Precio_Real', ax=ax_precio, label='Precio Real', color='black', linewidth=2)
            sns.lineplot(data=precio_df, x='Indice', y='Meta_LSTM', ax=ax_precio, label='Meta LSTM', color='red', linestyle='--', linewidth=2)
            if 'LGB' in precio_df.columns:
                sns.lineplot(data=precio_df, x='Indice', y='LGB', ax=ax_precio, label='LGB', color='blue', alpha=0.6)
            if 'CatBoost' in precio_df.columns:
                sns.lineplot(data=precio_df, x='Indice', y='CatBoost', ax=ax_precio, label='CatBoost', color='green', alpha=0.6)
            if 'TimeXer' in precio_df.columns:
                sns.lineplot(data=precio_df, x='Indice', y='TimeXer', ax=ax_precio, label='TimeXer', color='purple', alpha=0.6)
            if 'Moirai' in precio_df.columns:
                sns.lineplot(data=precio_df, x='Indice', y='Moirai', ax=ax_precio, label='Moirai', color='orange', alpha=0.6)
            ax_precio.set_title('Predicciones de Todos los Modelos en Escala de Precio Original')
            ax_precio.set_xlabel('Indice')
            ax_precio.set_ylabel('Precio')
            ax_precio.legend(loc='upper right')
            ax_precio.grid(True, alpha=0.3)
            st.pyplot(fig_precio)
            plt.close(fig_precio)
            st.subheader("Metricas sobre Precio Real (Escala Original)")
            precio_real_arr = np.array(precio_real)
            precio_meta_arr = np.array(precio_meta)
            def calcular_metricas(y_true, y_pred):
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                return mse, rmse, mae, r2
            metricas_lista = []
            mse_meta, rmse_meta, mae_meta, r2_meta = calcular_metricas(precio_real_arr, precio_meta_arr)
            metricas_lista.append({'Modelo': 'Meta LSTM', 'MSE': mse_meta, 'RMSE': rmse_meta, 'MAE': mae_meta, 'R2': r2_meta})
            if precio_lgb:
                mse_lgb, rmse_lgb, mae_lgb, r2_lgb = calcular_metricas(precio_real_arr, np.array(precio_lgb))
                metricas_lista.append({'Modelo': 'LightGBM', 'MSE': mse_lgb, 'RMSE': rmse_lgb, 'MAE': mae_lgb, 'R2': r2_lgb})
            if precio_cb:
                mse_cb, rmse_cb, mae_cb, r2_cb = calcular_metricas(precio_real_arr, np.array(precio_cb))
                metricas_lista.append({'Modelo': 'CatBoost', 'MSE': mse_cb, 'RMSE': rmse_cb, 'MAE': mae_cb, 'R2': r2_cb})
            if precio_tx:
                mse_tx, rmse_tx, mae_tx, r2_tx = calcular_metricas(precio_real_arr, np.array(precio_tx))
                metricas_lista.append({'Modelo': 'TimeXer', 'MSE': mse_tx, 'RMSE': rmse_tx, 'MAE': mae_tx, 'R2': r2_tx})
            if precio_moirai:
                mse_moirai, rmse_moirai, mae_moirai, r2_moirai = calcular_metricas(precio_real_arr, np.array(precio_moirai))
                metricas_lista.append({'Modelo': 'Moirai-MoE', 'MSE': mse_moirai, 'RMSE': rmse_moirai, 'MAE': mae_moirai, 'R2': r2_moirai})
            metricas_df = pd.DataFrame(metricas_lista)
            metricas_df = metricas_df.round(4)
            metricas_df = metricas_df.sort_values(by='MAE', ascending=True).reset_index(drop=True)
            st.dataframe(metricas_df, use_container_width=True)
            mejor_modelo = metricas_df.iloc[0]['Modelo']
            mejor_mae = metricas_df.iloc[0]['MAE']
            mejor_r2 = metricas_df.iloc[0]['R2']
            st.write(f"Mejor modelo por MAE: {mejor_modelo} (MAE={mejor_mae}, R2={mejor_r2})")
            st.subheader("Comparacion Visual de Metricas (Precio Real)")
            fig_metricas, axes = plt.subplots(2, 2, figsize=(12, 10))
            sns.barplot(data=metricas_df, x='Modelo', y='MSE', ax=axes[0, 0], palette='Blues_d')
            axes[0, 0].set_title('MSE por Modelo')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            sns.barplot(data=metricas_df, x='Modelo', y='RMSE', ax=axes[0, 1], palette='Greens_d')
            axes[0, 1].set_title('RMSE por Modelo')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            sns.barplot(data=metricas_df, x='Modelo', y='MAE', ax=axes[1, 0], palette='Oranges_d')
            axes[1, 0].set_title('MAE por Modelo')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            sns.barplot(data=metricas_df, x='Modelo', y='R2', ax=axes[1, 1], palette='Purples_d')
            axes[1, 1].set_title('R2 por Modelo')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig_metricas)
            plt.close(fig_metricas)
            if metricas_df.iloc[0]['Modelo'] == 'Meta LSTM':
                st.success("El Meta-Modelo LSTM es el mejor modelo en escala de precio real")
            else:
                st.info(f"El mejor modelo en escala de precio real es: {mejor_modelo}")
        else:
            st.error("No hay suficientes datos para entrenar el meta-modelo. Se requieren al menos 20 muestras validas.")
            st.write("Intenta con mas trials o ajusta los parametros de los modelos base.")