import numpy as np
from sktime.forecasting.model_selection import SlidingWindowSplitter

def wfrw(y_data, k=5, fh_val=30):
    n_samples = len(y_data)
    fh = np.arange(1, fh_val + 1)
    window_length = int(n_samples * 0.5)
    usable_samples = n_samples - window_length - fh_val
    step_length = max(1, int(usable_samples // (k - 1)))
    splitter = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)
    print(f"folds: {splitter.get_n_splits(y_data)}")
    return splitter
    
