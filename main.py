import pandas as pd
import numpy as np
import warnings
import os
from data.yfinance_data import download_yf
warnings.filterwarnings("ignore")

#Extracción datos
df = pd.DataFrame()                              
tokens  = ['KO', 'AAPL', 'NVDA', 'JNJ', '^GSPC'] # acciones y el indice que escojimos, coca cola, apple, nvidia, jyj, s&p500
start = "2019-12-31" # 2020
end = "2025-11-01" # Hasta el 2025 del 31 de octubre

for token in tokens:
    df[token] = download_yf(token, start, end)

df = pd.read_csv(r"C:\Users\hibra\Desktop\TT\data\AAPL_2020-2025.csv")
print(df.head(20))