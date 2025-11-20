import yfinance as yf
import pandas as pd
import os

os.makedirs('data/tokens', exist_ok=True)

def download_yf(acciones, start, end):
    for accion in acciones:
        serie = yf.download(accion, start, end, multi_level_index=False)
        serie["Date_final"] = pd.to_datetime(serie.index)
        serie["Date_final"] = serie["Date_final"].dt.date
        serie.to_csv(f"data/tokens/{accion}_2020-2025.csv")
