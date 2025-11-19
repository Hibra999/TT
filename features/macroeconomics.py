import pandas as pd
import warnings
from datetime import datetime
import pandas_datareader.data as web
warnings.filterwarnings("ignore")

start = datetime(2020,1,1)
end = datetime(2025, 10, 31)

def macroeconomicos():
    macro = web.DataReader(["GDP","CPIAUCSL","FEDFUNDS","DGS10","SOFR","UNRATE"],"fred", start=start, end=end)
    return macro