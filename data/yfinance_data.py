import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class extract_yf_data():
    def __init__(self, start, end, accion):
        self.start = start
        self.end = end
        self.accion = accion
    def download(self, start, end, accion):
        self.data = yf.download(self.accion, self.start, self.end)
        return self.data
