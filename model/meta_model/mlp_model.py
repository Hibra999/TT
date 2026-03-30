import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class MLPMeta(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_predict(
    oof_df: pd.DataFrame,
    X_test: np.ndarray,
    n_trials: int = 10,
    device: torch.device | None = None,
    random_state: int = 42
) -> tuple[np.ndarray, dict]:
    """
    MLP meta model. Optimiza hiperparámetros con Optuna + 5-fold CV
    sobre OOF. No requiere ventana deslizante (predicción pointwise).

    Parámetros
    ----------
    oof_df      : DataFrame con columnas OOF + 'target'
    X_test      : matriz (n_test, n_bases) con predicciones base en test set
    n_trials    : número de trials Optuna
    device      : torch.device para entrenamiento
    random_state: ignorado

    Retorna
    -------
    predictions : np.ndarray shape (n_test,)
    meta_info   : dict con hiperparámetros óptimos y oof_mse
    """
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        feat_cols = [c for c in oof_df.columns if c not in ('idx', 'target')]
        n_features = len(feat_cols)
        X_oof = oof_df[feat_cols].values
        y_oof = oof_df['target'].values
        n_test = len(X_test)

        scaler = StandardScaler()
        X_oof_scaled = scaler.fit_transform(X_oof)

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_size = trial.suggest_int('hidden_size', 16, 256)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            epochs = trial.suggest_int('epochs', 20, 100)
            # 5-fold CV
            kf = KFold(n_splits=5, shuffle=False)
            fold_scores = []
            for tr_i, va_i in kf.split(X_oof_scaled):
                model_ = MLPMeta(n_features, hidden_size, n_layers, dropout).to(device)
                opt_ = torch.optim.Adam(model_.parameters(), lr=lr)
                crit_ = nn.MSELoss()
                X_tr_t = torch.tensor(X_oof_scaled[tr_i], dtype=torch.float32).to(device)
                y_tr_t = torch.tensor(y_oof[tr_i], dtype=torch.float32).to(device)
                X_va_t = torch.tensor(X_oof_scaled[va_i], dtype=torch.float32).to(device)
                ds_tr = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
                dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
                model_.train()
                for _ in range(epochs):
                    for xb, yb in dl_tr:
                        opt_.zero_grad()
                        loss = crit_(model_(xb), yb)
                        loss.backward()
                        opt_.step()
                model_.eval()
                with torch.no_grad():
                    va_pred = model_(X_va_t).cpu().numpy()
                fold_scores.append(mean_squared_error(y_oof[va_i], va_pred))
            return np.mean(fold_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        bp = study.best_params

        # Train final model on all OOF
        final_model = MLPMeta(n_features, bp['hidden_size'], bp['n_layers'], bp['dropout']).to(device)
        opt = torch.optim.Adam(final_model.parameters(), lr=bp['lr'])
        crit = nn.MSELoss()
        X_oof_t = torch.tensor(X_oof_scaled, dtype=torch.float32).to(device)
        y_oof_t = torch.tensor(y_oof, dtype=torch.float32).to(device)
        ds_full = torch.utils.data.TensorDataset(X_oof_t, y_oof_t)
        dl_full = DataLoader(ds_full, batch_size=bp['batch_size'], shuffle=True)
        final_model.train()
        for _ in range(bp['epochs']):
            for xb, yb in dl_full:
                opt.zero_grad()
                loss = crit(final_model(xb), yb)
                loss.backward()
                opt.step()
        final_model.eval()

        # Predict on test set
        predictions = np.full(n_test, np.nan)
        mask = ~np.any(np.isnan(X_test), axis=1)
        X_test_scaled = scaler.transform(X_test[mask])
        with torch.no_grad():
            predictions[mask] = final_model(
                torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            ).cpu().numpy()

        n_valid = int((~np.isnan(predictions)).sum())
        print(f'  [MLP_META] {n_valid}/{n_test} válidas')

        meta_info = {
            'n_layers': bp['n_layers'], 'hidden_size': bp['hidden_size'],
            'dropout': bp['dropout'], 'lr': bp['lr'],
            'epochs': bp['epochs'], 'oof_mse': study.best_value
        }
        return predictions, meta_info
    except Exception as e:
        logging.warning(f'[MLP_META] MLP Meta falló: {e}')
        return np.full(len(X_test), np.nan), {}
