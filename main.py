import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import streamlit as st
from data.yfinance_data import download_yf
from data.ccxt_data import download_cx
from features.macroeconomics import macroeconomicos
from features.tecnical_indicators import TA
from features.top_n import top_k

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

with st.sidebar:
    start = "2020-01-01"
    end = "2025-11-01"

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

tab1, tab2, tab3 = st.tabs(["Datos & Retornos", "Indicadores (TA/Macro)", "Modelo (MIC)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CLOSE")
        st.line_chart(df["Close"])
        st.dataframe(df.head(31))
    with col2:
        st.subheader("LOG RETURN")
        log_df = np.log(df["Close"] / df["Close"].shift(-1)).dropna()
        st.line_chart(log_df)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Indicadores Tecnicos")
        df_ta = TA(df)
        st.dataframe(df_ta.head(31))
    with col2:
        st.subheader("Datos Macroeconomicos")
        df_ma = macroeconomicos(df["Date_final"])
        st.dataframe(df_ma.head(31))

with tab3:
    df_ta = df_ta.reset_index(drop=True)
    df_ma = df_ma.reset_index(drop=True)
    
    st.subheader("DF_final")
    df_final = pd.concat([df_ta, df_ma], axis=1)
    st.dataframe(df_final)

    st.subheader("MIC: top n caracteristicas")
    features = top_k(df_final.dropna(axis=0), log_df, 15)
    st.write(features)