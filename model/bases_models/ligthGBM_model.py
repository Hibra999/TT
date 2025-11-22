import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna

import optuna
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error

def objective_global(trial, X, y, splitter):
    param = {
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "rf"]),
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
        "verbose": -1,
        "n_jobs": -1
    }
    if param["boosting_type"] == "rf":
        param["subsample"] = max(param["subsample"], 0.7) 
        param["subsample_freq"] = 1
    fold_scores = []
    for t_idx, v_idx in splitter.split(y):
        X_train, y_train = X.iloc[t_idx], y.iloc[t_idx]
        X_test, y_test = X.iloc[v_idx], y.iloc[v_idx]
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        fold_scores.append(mae)
    return np.mean(fold_scores)