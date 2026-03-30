import numpy as np
import pandas as pd
import logging


def train_and_predict(
    oof_df: pd.DataFrame,
    X_test: np.ndarray,
    n_trials: int = 10,
    device=None,
    random_state: int = 42
) -> tuple[np.ndarray, dict]:
    """
    Simple Average meta model. Promedia las predicciones de todos los
    modelos base sin entrenamiento.

    Parámetros
    ----------
    oof_df      : DataFrame con columnas OOF + 'target' (no usado para entrenar)
    X_test      : matriz (n_test, n_bases) con predicciones base en test set
    n_trials    : ignorado
    device      : ignorado
    random_state: ignorado

    Retorna
    -------
    predictions : np.ndarray shape (n_test,)
    meta_info   : dict
    """
    try:
        cols = [c for c in oof_df.columns if c not in ('idx', 'target')]
        n_bases = len(cols)

        n_test = len(X_test)
        predictions = np.full(n_test, np.nan)
        mask = ~np.any(np.isnan(X_test), axis=1)
        predictions[mask] = np.mean(X_test[mask], axis=1)

        n_valid = int((~np.isnan(predictions)).sum())
        print(f'  [SA] {n_valid}/{n_test} válidas')

        return predictions, {'method': 'simple_average', 'n_bases': n_bases}
    except Exception as e:
        logging.warning(f'[SA] Simple Average falló: {e}')
        return np.full(len(X_test), np.nan), {}
