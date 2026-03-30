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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerMeta(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


def _build_windows(X, y, ws):
    """Construye ventanas deslizantes para entrenamiento secuencial."""
    Xw, yw = [], []
    for i in range(ws - 1, len(y)):
        Xw.append(X[i - ws + 1:i + 1])
        yw.append(y[i])
    return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.float32)


def train_and_predict(
    oof_df: pd.DataFrame,
    X_test: np.ndarray,
    n_trials: int = 10,
    device: torch.device | None = None,
    random_state: int = 42
) -> tuple[np.ndarray, dict]:
    """
    Transformer meta model. Optimiza hiperparámetros con Optuna + 5-fold CV
    sobre OOF con ventana deslizante. Incluye positional encoding y
    proyección de n_features → d_model.

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
            d_model = trial.suggest_categorical('d_model', [16, 32, 64])
            nhead = trial.suggest_categorical('nhead', [1, 2, 4])
            if d_model % nhead != 0:
                return float('inf')
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.0, 0.4)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            epochs = trial.suggest_int('epochs', 20, 100)
            window_size = trial.suggest_int('window_size', 3, 20)
            Xw, yw = _build_windows(X_oof_scaled, y_oof, window_size)
            if len(Xw) < 20:
                return float('inf')
            kf = KFold(n_splits=5, shuffle=False)
            fold_scores = []
            for tr_i, va_i in kf.split(Xw):
                model_ = TransformerMeta(n_features, d_model, nhead, num_layers, dropout).to(device)
                opt_ = torch.optim.Adam(model_.parameters(), lr=lr)
                crit_ = nn.MSELoss()
                ds_tr = torch.utils.data.TensorDataset(
                    torch.tensor(Xw[tr_i]).to(device), torch.tensor(yw[tr_i]).to(device))
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
                    va_pred = model_(torch.tensor(Xw[va_i]).to(device)).cpu().numpy()
                fold_scores.append(mean_squared_error(yw[va_i], va_pred))
            return np.mean(fold_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        bp = study.best_params
        ws = bp['window_size']

        # Train final model on all OOF
        Xw_full, yw_full = _build_windows(X_oof_scaled, y_oof, ws)
        final_model = TransformerMeta(n_features, bp['d_model'], bp['nhead'],
                                      bp['num_layers'], bp['dropout']).to(device)
        opt = torch.optim.Adam(final_model.parameters(), lr=bp['lr'])
        crit = nn.MSELoss()
        ds_full = torch.utils.data.TensorDataset(
            torch.tensor(Xw_full).to(device), torch.tensor(yw_full).to(device))
        dl_full = DataLoader(ds_full, batch_size=bp['batch_size'], shuffle=True)
        final_model.train()
        for _ in range(bp['epochs']):
            for xb, yb in dl_full:
                opt.zero_grad()
                loss = crit(final_model(xb), yb)
                loss.backward()
                opt.step()
        final_model.eval()

        # Predict on test set using sliding window
        predictions = np.full(n_test, np.nan)
        X_test_scaled = scaler.transform(X_test)
        with torch.no_grad():
            for i in range(ws - 1, n_test):
                window = X_test_scaled[i - ws + 1:i + 1]
                if not np.isnan(window).any():
                    x_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                    predictions[i] = final_model(x_t).cpu().item()

        n_valid = int((~np.isnan(predictions)).sum())
        print(f'  [TRANS_META] ws={ws}, d_model={bp["d_model"]}, nhead={bp["nhead"]}, '
              f'{n_valid}/{n_test} válidas')

        meta_info = {
            'd_model': bp['d_model'], 'nhead': bp['nhead'],
            'num_layers': bp['num_layers'], 'window_size': ws,
            'dropout': bp['dropout'], 'lr': bp['lr'],
            'epochs': bp['epochs'], 'oof_mse': study.best_value
        }
        return predictions, meta_info
    except Exception as e:
        logging.warning(f'[TRANS_META] Transformer Meta falló: {e}')
        return np.full(len(X_test), np.nan), {}
