import numpy as np
import pandas as pd
import optuna
import logging
import lightgbm as lgb_pkg
from sklearn.model_selection import cross_val_score


def train_and_predict(
    oof_df: pd.DataFrame,
    X_test: np.ndarray,
    n_trials: int = 10,
    device=None,
    random_state: int = 42
) -> tuple[np.ndarray, dict]:
    """
    LightGBM meta model. Optimiza hiperparámetros con Optuna
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
                'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
                'learning_rate':    trial.suggest_float('lr', 1e-3, 0.3, log=True),
                'num_leaves':       trial.suggest_int('num_leaves', 15, 127),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'verbosity': -1, 'random_state': random_state
            }
            model_ = lgb_pkg.LGBMRegressor(**params)
            scores = cross_val_score(model_, X_oof, y_oof, cv=5,
                                     scoring='neg_mean_squared_error')
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        bp = dict(study.best_params)
        bp['verbosity'] = -1
        bp['random_state'] = random_state
        # Rename 'lr' key to 'learning_rate' for LGBMRegressor
        if 'lr' in bp:
            bp['learning_rate'] = bp.pop('lr')

        best_model = lgb_pkg.LGBMRegressor(**bp)
        best_model.fit(X_oof, y_oof)

        predictions = np.full(n_test, np.nan)
        mask = ~np.any(np.isnan(X_test), axis=1)
        predictions[mask] = best_model.predict(X_test[mask])

        n_valid = int((~np.isnan(predictions)).sum())
        print(f'  [LGB_META] {n_valid}/{n_test} válidas')

        meta_info = {
            'n_estimators': bp.get('n_estimators'),
            'learning_rate': bp.get('learning_rate'),
            'num_leaves': bp.get('num_leaves'),
            'max_depth': bp.get('max_depth'),
            'oof_mse': study.best_value
        }
        return predictions, meta_info
    except Exception as e:
        logging.warning(f'[LGB_META] LightGBM Meta falló: {e}')
        return np.full(len(X_test), np.nan), {}
