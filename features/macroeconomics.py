import pandas as pd
import warnings
from datetime import datetime
import time
warnings.filterwarnings("ignore")

start = datetime(2020,1,1)
end = datetime(2025, 10, 31)

def macroeconomicos(tiempo):
    # FRED deshabilitado temporalmente por problemas de conexión
    print("  [Info] FRED deshabilitado - usando datos sin variables macroeconómicas")
    macro = pd.DataFrame({"Date": pd.to_datetime(tiempo).dt.date})
    macro = macro.drop_duplicates().set_index("Date")
    return macro