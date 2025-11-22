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
from preprocessing.walk_forward import wfrw
from features.tecnical_indicators import TA
from features.top_n import top_k
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
        st.line_chart(log_close)

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
    #st.write(df_final.describe())
    df_final = df_final.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    #st.write(df_final.describe())
    st.dataframe(df_final.tail())
    st.subheader("MIC: top n caracteristicas")
    df_final = df_final.iloc[1:]
    features, valores_mic = top_k(df_final, log_close, 15)
    df_importance = pd.DataFrame(list(valores_mic.items()), columns=['Feature', 'Score'])
    fig = px.bar(df_importance,x='Score',y='Feature', orientation='h',title='MIC')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    X = df_final[features] 
    y = log_close.iloc[1:].reset_index(drop=True)

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
    st.plotly_chart(fig_tt, use_container_width=True)
    #Grafica de wf
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, (t_idx, v_idx) in enumerate(wfrw(y_train, k=5, fh_val=30).split(y_train)):
        fig.add_trace(go.Scatter(x=y_train.index[t_idx], y=y_train.iloc[t_idx], line_color='blue'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=y_train.index[v_idx], y=y_train.iloc[v_idx], line_color='red'), row=i+1, col=1)
    fig.update_layout(height=800, showlegend=False, title="Folds", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab5:

    """
    st.subheader("lgb")
    splitter = wfrw(y, k=5, fh_val=30)
    with st.spinner('optimizando lgb'):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_global(trial, X, y, splitter), n_trials=300, n_jobs=-1)
    best_params = study.best_params
    st.json(best_params)
    st.write(f"Mejor MAE Promedio Global: {study.best_value:.4f}")
    st.subheader("CatBoost")
    st.subheader("catboost")

    with st.spinner('optimizando catboost'):
        study_cb = optuna.create_study(direction="minimize")
        study_cb.optimize(lambda trial: objective_catboost_global(trial, X, y, splitter), n_trials=300, n_jobs=1)
    best_params_cb = study_cb.best_params
    st.write("CatBoost hiper:")
    st.json(best_params_cb)
    st.write(f"Mejor MAE Promedio Global: {study_cb.best_value:.4f}")"""

    
    pass

