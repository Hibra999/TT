import talib as ta
import numpy as np
import pandas as pd

def TA(df):
    print(df.columns)
    close = df["Close"]
    df_ta = pd.DataFrame()
    resagos = [i for i in range(30, 365)] # SMA hasta de uno año
    for r in resagos:
        df_ta[f"SMA_{r}"] = ta.SMA(close, r)
        df_ta[f"EMA_{r}"] = ta.EMA(close, r)
        df_ta[f"TEMA_{r}"] = ta.TEMA(close, r)
        df_ta[f"WMA_{r}"] = ta.WMA(close, r)
    return df_ta