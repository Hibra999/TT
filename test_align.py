import pandas as pd
import numpy as np
import os
from data.yfinance_data import download_yf
from data.ccxt_data import download_cx
from features.macroeconomics import macroeconomicos
from features.tecnical_indicators import TA

TOKEN = '^GSPC'
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'tokens', f'{TOKEN.replace("/", "-")}_2020-2025.csv'))
print(f"Original df size: {len(df)}")
df_ta = TA(df)
df_ma = macroeconomicos(df['Date_final'])

target_series = np.log(df['Close'].shift(-1) / df['Close'])
df_dates = pd.to_datetime(df['Date_final']).dt.date
df_ma_aligned = df_ma.reindex(df_dates).ffill().bfill().reset_index(drop=True)

df_ta_r = df_ta.reset_index(drop=True)
target_series_r = target_series.reset_index(drop=True)

df_combined = pd.concat([df_ta_r, df_ma_aligned], axis=1)
df_combined['target_lc'] = target_series_r
df_combined['orig_idx'] = df_combined.index

df_combined = df_combined.dropna().reset_index(drop=True)
orig_idx_array = df_combined['orig_idx'].values

print(f"Final combined size: {len(df_combined)}")
print(f"First valid index: {orig_idx_array[0]}")
print(f"Last valid index: {orig_idx_array[-1]}")
print(f"Test size roughly: {int(len(df_combined)*0.1)}")
