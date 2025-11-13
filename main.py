import pandas as pd
import warnings
from data.yfinance_data import download_yf
from data.ccxt_data import download_cx
warnings.filterwarnings("ignore")
import streamlit as st #Interfaz para facilitarnos el trabajo

#Extracción datos
df = pd.DataFrame()    
start = "2019-12-31" # 2020
end = "2025-11-01" # Hasta el 2025 del 31 de octubre

#Acciones y indices
tokens  = ['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC'] # acciones y el indice que escojimos, coca cola, apple, nvidia, jyj, s&p500
download_yf(tokens, start, end)

#Criptomonedas
cryptos = ["BTC/USDT", "ETH/USDT"] # bitcoin y etherium
download_cx(cryptos, start, end)

#df = pd.read_csv(r"C:\Users\hibra\Desktop\TT\data\AAPL_2020-2025.csv")
st.title('TT')
df = pd.read_csv(r"C:\Users\hibra\Desktop\TT\data\tokens\AAPL_2020-2025.csv") #Hay que mejorar esto del directorio cesarin
st.dataframe(df.head(20))
