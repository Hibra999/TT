import numpy as np
import pandas as pd
import optuna
import logging
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def train_and_predict(
    oof_df: pd.DataFrame,
    X_test: np.ndarray,
    n_trials: int = 10,
    device=None,
    random_state: int = 42
) -> tuple[np.ndarray, dict]:
    """
    Ridge meta model. Optimiza alpha con Optuna + 5-fold CV sobre OOF.

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
    meta_info   : dict con alpha óptimo y oof_mse
    """
    try:
        cols = [c for c in oof_df.columns if c not in ('idx', 'target')]
        X_oof = oof_df[cols].values
        y_oof = oof_df['target'].values
        n_test = len(X_test)

        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
            kf = KFold(n_splits=5, shuffle=False)
            scores = []
            for tr_idx, va_idx in kf.split(X_oof):
                m = Ridge(alpha=alpha)
                m.fit(X_oof[tr_idx], y_oof[tr_idx])
                scores.append(mean_squared_error(y_oof[va_idx], m.predict(X_oof[va_idx])))
            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_alpha = study.best_params['alpha']
        best_model = Ridge(alpha=best_alpha)
        best_model.fit(X_oof, y_oof)

        predictions = np.full(n_test, np.nan)
        mask = ~np.any(np.isnan(X_test), axis=1)
        predictions[mask] = best_model.predict(X_test[mask])

        n_valid = int((~np.isnan(predictions)).sum())
        print(f'  [RD] alpha={best_alpha:.6f}, {n_valid}/{n_test} válidas')

        return predictions, {'alpha': best_alpha, 'oof_mse': study.best_value}
    except Exception as e:
        logging.warning(f'[RD] Ridge falló: {e}')
        return np.full(len(X_test), np.nan), {}
