import pandas as pd
import warnings
from data.yfinance_data import download_yf
from data.ccxt_data import download_cx
from features.tecnical_indicators import TA
warnings.filterwarnings("ignore")
import streamlit as st #Interfaz para facilitarnos el trabajo
st.title('TT')
#Extracción datos
df = pd.DataFrame()    
start = "2019-12-31" # 2020
end = "2025-11-01" # Hasta el 2025 del 31 de octubre

#Criptomonedas
@st.cache_data
def load_data():
    #Acciones y indices
    tokens  = ['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC', "GC=F", "CBOE"] 
    dy = download_yf(tokens, start, end)
    cryptos = ["BTC/USDT", "ETH/USDT"] # bitcoin y etherium
    dc = download_cx(cryptos, start, end)
    return dy, dc
load_data()

token = st.selectbox(label="ACTIVO FINANCIERO: ", options=['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC', "BTC-USDT", "ETH-USDT"])
df = pd.read_csv(rf"C:\Users\hibra\Desktop\TT\data\tokens\{token}_2020-2025.csv") #Hay que mejorar esto del directorio cesarin
st.dataframe(df.head(5))

st.subheader("Indicadores Tecnicos")
df_ta = TA(df)
st.dataframe(df_ta.tail(5))