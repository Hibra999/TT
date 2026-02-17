from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

def objective_catboost_global(trial, X, y, splitter, oof_storage=None):
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
        "task_type": "CPU"
    }
    if bootstrap_type == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    else:
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.01, 10.0, log=True)
    
    fold_scores = []
    fold_preds = []
    fold_indices = []

    for t_idx, v_idx in splitter.split(y):
        X_train, y_train = X.iloc[t_idx], y.iloc[t_idx]
        X_val, y_val = X.iloc[v_idx], y.iloc[v_idx]
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5, verbose=False)
        y_pred = model.predict(X_val)
        fold_scores.append(mean_absolute_error(y_val, y_pred))
        fold_preds.append(y_pred)
        fold_indices.append(v_idx)
    
    mean_score = float(np.mean(fold_scores))
    
    if oof_storage is not None:
        if 'best_score' not in oof_storage or mean_score < oof_storage['best_score']:
            oof_storage['best_score'] = mean_score
            oof_storage['params'] = param.copy()
            oof_storage['preds'] = fold_preds
            oof_storage['indices'] = fold_indices
    
    return mean_score


def train_final_and_predict_test(X_train, y_train, X_test, best_params):
    """
    Entrena modelo final con todo el train y predice en test.
    """
    model = CatBoostRegressor(**best_params)
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)
    return predictions, model