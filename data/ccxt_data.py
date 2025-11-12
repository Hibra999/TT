import pandas as pd
from datetime import datetime
import os
import ccxt

os.makedirs('data/tokens', exist_ok=True)
exchange = ccxt.binance()

def download_cx(cryptos, start, end):
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
    for cyt in cryptos:
        all_data = []
        current_ts = start_ts
        while current_ts < end_ts:
            ohlcv = exchange.fetch_ohlcv(cyt, '1d', since=current_ts, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 86400000 
        df_temp = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_temp['date'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
        df_temp.set_index('date', inplace=True)
        filename = cyt.replace('/', '-')
        df_temp.to_csv(f"data/tokens/{filename}_2020-2025.csv")
