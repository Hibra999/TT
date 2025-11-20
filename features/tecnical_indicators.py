import talib as ta
import numpy as np
import pandas as pd


def TA(df):
    print(df.columns)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    df_ta = pd.DataFrame()
    resagos = [i for i in range(30,365)]
    for r in resagos:
        # IT de Tendencia
        df_ta[f"SMA_{r}"] = ta.SMA(close, timeperiod = r)
        df_ta[f"EMA_{r}"] = ta.EMA(close, timeperiod = r)
        df_ta[f"TEMA_{r}"] = ta.TEMA(close, timeperiod = r)
        df_ta[f"WMA_{r}"] = ta.WMA(close, timeperiod = r)
    

        # IT de Volatilidad
        df_ta[f"upperband_{r}"], df_ta[f"middleband_{r}"], df_ta[f"lowerband_{r}"] = ta.BBANDS(close, timeperiod=r, nbdevup=2, nbdevdn=2, matype=0)
        df_ta[f"ATR_{r}"] = ta.ATR(high, low, close, timeperiod=r)

        # IT de Momentum
        df_ta[f"MOM_{r}"] = ta.MOM(close, timeperiod = r)
        df_ta[f"RSI_{r}"] = ta.RSI(close, timeperiod = r)
        df_ta[f"STOCHRSI_fastk_{r}"], df_ta[f"STOCHRSI_fastd_{r}"] = ta.STOCHRSI(close, timeperiod = r, fastk_period=5, fastd_period=3, fastd_matype=0)
        df_ta[f"KAMA_{r}"] = ta.KAMA(close, timeperiod=r)

        # Volumen con resagos
        df_ta[f"MFI_{r}"] = ta.MFI(high, low, close, volume, timeperiod=r)
    # IT de Volumen
    df_ta["AD"] = ta.AD(high, low, close, volume)
    df_ta["ADOSC"] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df_ta["OBV"] = ta.OBV(close, volume)

    return df_ta
