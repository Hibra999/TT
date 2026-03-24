import numpy as np
from sktime.forecasting.model_selection import SlidingWindowSplitter

def wfrw(y_data, k=5, fh_val=30, window_ratio=0.5):
    """
    Crea un splitter para validación walk-forward.
    
    Args:
        y_data: Serie temporal de target
        k: Número de folds deseado
        fh_val: Horizonte de predicción
        window_ratio: Fracción de datos para training (0.5 = 50%, 0.7 = 70%)
    """
    n_samples = len(y_data)
    fh = np.arange(1, fh_val + 1)
    ventana = int(n_samples * window_ratio)
    # Asegurar que window_length + fh_val < n_samples (requisito de sktime)
    max_ventana = n_samples - fh_val - 1
    if ventana > max_ventana:
        print(f"  [WARN] window_length={ventana} + fh={fh_val} >= n_samples={n_samples}. "
              f"Reduciendo window_length a {max_ventana}")
        ventana = max_ventana
    oof = n_samples - ventana - fh_val
    pasos = max(1, int(oof // (k - 1)))
    splitter = SlidingWindowSplitter(fh=fh, window_length=ventana, step_length=pasos)
    print(f"folds: {splitter.get_n_splits(y_data)}")
    print(f"  window_length: {ventana} ({window_ratio*100:.0f}% de {n_samples})")
    print(f"  step_length: {pasos}")
    return splitter
    
