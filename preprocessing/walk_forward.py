import numpy as np
from sktime.forecasting.model_selection import SlidingWindowSplitter

def wfrw(y_data, k=5, fh_val=30):
    n_samples = len(y_data)
    fh = np.arange(1, fh_val + 1)
    ventana = int(n_samples * 0.5)
    oof = n_samples - ventana - fh_val
    pasos = max(1, int(oof // (k - 1)))
    splitter = SlidingWindowSplitter(fh=fh, window_length=ventana, step_length=pasos)
    print(f"folds: {splitter.get_n_splits(y_data)}")
    return splitter
    
