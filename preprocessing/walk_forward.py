import numpy as np
from sktime.forecasting.model_selection import SlidingWindowSplitter

def wfrw(y_data, k=5, fh_val=30, window_ratio=0.75, step_length=None):
    n_samples = len(y_data)
    fh = np.arange(1, fh_val + 1)
    ventana = int(n_samples * window_ratio)
    max_ventana = n_samples - fh_val - 1
    if ventana > max_ventana:
        print(f"  [WARN] window_length={ventana} + fh={fh_val} >= n_samples={n_samples}. "
              f"Reduciendo window_length a {max_ventana}")
        ventana = max_ventana

    if step_length is not None:
        pasos = step_length
    else:
        oof = n_samples - ventana - fh_val
        pasos = max(1, int(oof // (k - 1)))

    splitter = SlidingWindowSplitter(fh=fh, window_length=ventana, step_length=pasos)
    n_real = splitter.get_n_splits(y_data)
    print(f"folds: {n_real}")
    print(f"  window_length: {ventana} ({window_ratio*100:.0f}% de {n_samples})")
    print(f"  step_length:   {pasos}")
    print(f"  OOF estimado:  ~{n_real * fh_val} observaciones")
    return splitter
    
