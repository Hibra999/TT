import numpy as np
import pandas as pd
import optuna
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def train_and_predict(
    oof_df: pd.DataFrame,
    X_test: np.ndarray,
    n_trials: int = 10,
    device=None,
    random_state: int = 42
) -> tuple[np.ndarray, dict]:
    """
    Random Forest meta model. Optimiza hiperparámetros con Optuna
    + 5-fold cross_val_score sobre OOF.

    Parámetros
    ----------
    oof_df      : DataFrame con columnas OOF + 'target'
    X_test      : matriz (n_test, n_bases) con predicciones base en test set
    n_trials    : número de trials Optuna
    device      : ignorado
    random_state: semilla de reproducibilidad

    Retorna
    -------
    predictions : np.ndarray shape (n_test,)
    meta_info   : dict con hiperparámetros óptimos y oof_mse
    """
    try:
        feat_cols = [c for c in oof_df.columns if c not in ('idx', 'target')]
        X_oof = oof_df[feat_cols].values
        y_oof = oof_df['target'].values
        n_test = len(X_test)

        def objective(trial):
            params = {
                'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
                'max_depth':         trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features':      trial.suggest_categorical('max_features',
                                     ['sqrt', 'log2', None]),
                'random_state': random_state, 'n_jobs': -1
            }
            model_ = RandomForestRegressor(**params)
            scores = cross_val_score(model_, X_oof, y_oof, cv=5,
                                     scoring='neg_mean_squared_error')
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        bp = dict(study.best_params)
        bp['random_state'] = random_state
        bp['n_jobs'] = -1

        best_model = RandomForestRegressor(**bp)
        best_model.fit(X_oof, y_oof)

        predictions = np.full(n_test, np.nan)
        mask = ~np.any(np.isnan(X_test), axis=1)
        predictions[mask] = best_model.predict(X_test[mask])

        n_valid = int((~np.isnan(predictions)).sum())
        print(f'  [RF_META] {n_valid}/{n_test} válidas')

        meta_info = {
            'n_estimators': bp.get('n_estimators'),
            'max_depth': bp.get('max_depth'),
            'max_features': bp.get('max_features'),
            'oof_mse': study.best_value
        }
        return predictions, meta_info
    except Exception as e:
        logging.warning(f'[RF_META] Random Forest Meta falló: {e}')
        return np.full(len(X_test), np.nan), {}
