from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

def objective_catboost_global(trial, X, y, splitter):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
    param = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": 42,              
        "allow_writing_files": False,
        "bootstrap_type": bootstrap_type,
        "task_type": "GPU",
        "thread_count": 1,
    }
    if bootstrap_type == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    else:
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.01, 10.0, log=True)
    fold_scores = []
    for t_idx, v_idx in splitter.split(y):
        X_train, y_train = X.iloc[t_idx], y.iloc[t_idx]
        X_test, y_test = X.iloc[v_idx], y.iloc[v_idx]
        model = CatBoostRegressor(**param)
        model.fit(X_train,y_train,eval_set=(X_test, y_test),early_stopping_rounds=5,verbose=False)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        fold_scores.append(mae)
    return float(np.mean(fold_scores))
