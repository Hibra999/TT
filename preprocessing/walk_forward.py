import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import SlidingWindowSplitter

def wfrw(y_data, k=5, fh_val=30, window_ratio=0.5):
    n_samples = len(y_data)
    fh = np.arange(1, fh_val + 1)
    ventana = int(n_samples * window_ratio)
    oof = n_samples - ventana - fh_val
    pasos = max(1, int(oof // (k - 1)))
    splitter = SlidingWindowSplitter(fh=fh, window_length=ventana, step_length=pasos)
    print(f"folds: {splitter.get_n_splits(y_data)}")
    return splitter


def experiment_wfrw(
    y_data,
    window_ratios=[0.3, 0.4, 0.5, 0.6, 0.7],
    k_values=[3, 5, 7],
    fh_values=[15, 30, 45, 60],
):
    """
    Prueba distintas configuraciones de Walk-Forward y devuelve
    un DataFrame con los resultados de cada combinacion.

    Parametros
    ----------
    y_data        : Serie temporal (target).
    window_ratios : Proporciones de ventana de entrenamiento a probar.
    k_values      : Cantidades de folds a probar.
    fh_values     : Horizontes de prediccion a probar.

    Retorna
    -------
    pd.DataFrame con columnas:
        window_ratio, window_size, k, fh, step_length,
        n_folds, train_pct, val_pct, oof_space
    """
    n = len(y_data)
    resultados = []

    for wr in window_ratios:
        for k in k_values:
            for fh_val in fh_values:
                ventana = int(n * wr)
                oof = n - ventana - fh_val

                if oof <= 0:
                    continue

                pasos = max(1, int(oof // (k - 1)))
                fh = np.arange(1, fh_val + 1)
                splitter = SlidingWindowSplitter(
                    fh=fh, window_length=ventana, step_length=pasos
                )
                n_folds = splitter.get_n_splits(y_data)

                resultados.append({
                    "window_ratio": wr,
                    "window_size": ventana,
                    "k": k,
                    "fh": fh_val,
                    "step_length": pasos,
                    "n_folds": n_folds,
                    "train_pct": round(ventana / n * 100, 1),
                    "val_pct": round(fh_val / n * 100, 1),
                    "oof_space": oof,
                })

    df = pd.DataFrame(resultados)
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Walk-Forward Configurations")
    print(f"  Total samples: {n} | Configs tested: {len(df)}")
    print(f"{'='*60}\n")
    print(df.to_string(index=False))
    return df

