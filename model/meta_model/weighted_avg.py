import numpy as np
import pandas as pd
import optuna
import logging
from sklearn.metrics import mean_squared_error


def train_and_predict(
    oof_df: pd.DataFrame,
    X_test: np.ndarray,
    n_trials: int = 10,
    device=None,
    random_state: int = 42
) -> tuple[np.ndarray, dict]:
    """
    Weighted Average meta model. Optimiza pesos por Optuna sobre OOF.

    Parámetros
    ----------
    oof_df      : DataFrame con columnas OOF + 'target'
    X_test      : matriz (n_test, n_bases) con predicciones base en test set
    n_trials    : número de trials Optuna
    device      : ignorado
    random_state: ignorado

    Retorna
    -------
    predictions : np.ndarray shape (n_test,)
    meta_info   : dict con pesos óptimos y oof_mse
    """
    try:
        cols = [c for c in oof_df.columns if c not in ('idx', 'target')]
        X_oof = oof_df[cols].values
        y_oof = oof_df['target'].values
        n_test = len(X_test)

        def objective(trial):
            w = np.array([
                trial.suggest_float(f'w_{c}', 0.0, 1.0) for c in cols
            ])
            w = w / w.sum()
            pred = (X_oof * w).sum(axis=1)
            return mean_squared_error(y_oof, pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_w = np.array([study.best_params[f'w_{c}'] for c in cols])
        best_w = best_w / best_w.sum()
        weights_dict = {c: round(float(w), 6) for c, w in zip(cols, best_w)}

        print(f'  [WA] Pesos óptimos: {weights_dict}')

        predictions = np.full(n_test, np.nan)
        mask = ~np.any(np.isnan(X_test), axis=1)
        predictions[mask] = (X_test[mask] * best_w).sum(axis=1)

        n_valid = int((~np.isnan(predictions)).sum())
        print(f'  [WA] {n_valid}/{n_test} válidas')

        return predictions, {'weights': weights_dict, 'oof_mse': study.best_value}
    except Exception as e:
        logging.warning(f'[WA] Weighted Average falló: {e}')
        return np.full(len(X_test), np.nan), {}
