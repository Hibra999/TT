import yfinance as yf
import pandas as pd
import os
os.makedirs('data', exist_ok=True)
def download_yf(accion, start, end):
    serie = yf.download(accion, start,  multi_level_index=False)
    serie.to_csv(f"data/{accion}_2020-2025.csv")
