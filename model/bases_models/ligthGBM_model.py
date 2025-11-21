import lightgbm as lgb
from sklearn.metrics import r2_score
import optuna



def train_fold(y_train, y_test):

    def objective(trial):
        param = {
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "rf"]),
            "num_leaves": trial.suggest_int("num_leaves", 31, 500),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 100, 200, 300, 500]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
            "subsample_for_bin": trial.suggest_int("subsample_for_bin", 200000, 2000000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": 42,
            "verbose": -1
        }
        model = lgb.LGBMRegressor()
        model.fit(y_train)
        y_pred = model.predict(y_test)
        r2 = r2_score(y_test, y_pred)
        return r2
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500, n_jobs=-1)
