import pandas as pd
import warnings
from datetime import datetime
import pandas_datareader.data as web
warnings.filterwarnings("ignore")

start = datetime(2020,1,1)
end = datetime(2025, 10, 31)

def macroeconomicos(tiempo):
    macro = web.DataReader(["GDP","CPIAUCSL","FEDFUNDS","DGS10","SOFR","UNRATE"],"fred", start=start, end=end)
    macro = macro.reset_index()
    macro["Date"] = pd.to_datetime(macro["DATE"]).dt.date
    macro = macro.drop("DATE", axis=1)
    macro = macro.ffill()
    macro = macro[macro["Date"].isin(pd.to_datetime(tiempo).dt.date)]
    macro = macro.dropna()
    macro = macro.set_index("Date")
    return macro