import yfinance as yf
import pandas as pd
import os

os.makedirs('data/tokens', exist_ok=True)

def download_yf(acciones, start, end):
    for accion in acciones:
        serie = yf.download(accion, start, end, multi_level_index=False)
        serie.to_csv(f"data/tokens/{accion}_2020-2025.csv")
