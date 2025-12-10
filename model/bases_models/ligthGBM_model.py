import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import numpy as np

def objective_global(trial, X, y, splitter, oof_storage=None):
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "rf"])
    param = {
        "boosting_type": boosting_type,
        "linear_tree": False,
        "num_leaves": trial.suggest_int("num_leaves", 31, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "device": "gpu",
        "verbose": -1
    }
    if param["boosting_type"] == "rf":
        param["subsample"] = max(param["subsample"], 0.7)
        param["subsample_freq"] = 1
    
    fold_scores = []
    for t_idx, v_idx in splitter.split(y):
        X_train, y_train = X.iloc[t_idx], y.iloc[t_idx]
        X_val, y_val = X.iloc[v_idx], y.iloc[v_idx]
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        fold_scores.append(mean_absolute_error(y_val, y_pred))
    
    mean_score = np.mean(fold_scores)
    
    # Guardar mejores parámetros
    if oof_storage is not None:
        if 'best_score' not in oof_storage or mean_score < oof_storage['best_score']:
            oof_storage['best_score'] = mean_score
            oof_storage['params'] = param.copy()
    
    return mean_score


def train_final_and_predict_test(X_train, y_train, X_test, best_params):
    """
    Entrena modelo final con todo el train y predice en test.
    """
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model