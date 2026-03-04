import pandas as pd
import warnings
from datetime import datetime
import pandas_datareader.data as web
import time
warnings.filterwarnings("ignore")

start = datetime(2020,1,1)
end = datetime(2025, 10, 31)

def macroeconomicos(tiempo):
    max_retries = 5
    macro = None
    for attempt in range(max_retries):
        try:
            macro = web.DataReader(["GDP","CPIAUCSL","FEDFUNDS","DGS10","SOFR","UNRATE"],"fred", start=start, end=end)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [Advertencia] Error conectando a FRED ({e}). Reintentando {attempt+1}/{max_retries} en 5s...")
                time.sleep(5)
            else:
                print("  [Error] No se pudo obtener la información de FRED después de múltiples intentos.")
                raise e
    macro = macro.reset_index()
    macro["Date"] = pd.to_datetime(macro["DATE"]).dt.date
    macro = macro.drop("DATE", axis=1)
    macro = macro.ffill()
    macro = macro[macro["Date"].isin(pd.to_datetime(tiempo).dt.date)]
    macro = macro.dropna()
    macro = macro.set_index("Date")
    return macro