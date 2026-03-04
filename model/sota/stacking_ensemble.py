"""
SOTA Stacking Ensemble: CatBoost + LightGBM + XGBoost + BaseLSTM → Meta LSTM
Implementación basada en el paper de referencia con ponderación dinámica convexa.
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F; import optuna; import xgboost as xgb
from torch.utils.data import Dataset, DataLoader; from sklearn.metrics import mean_absolute_error; from numba import njit
@njit(cache=True)
def _fast_valid_indices(oof_matrix, y_true, window_size):
    return [t for t in range(window_size - 1, len(y_true)) if not np.isnan(oof_matrix[t - window_size + 1:t + 1]).any() and not np.isnan(y_true[t])]

# ═══════════════════════════════════════════════════════════════════
#  XGBoost Base Learner
# ═══════════════════════════════════════════════════════════════════

def objective_xgboost_global(trial, X, y, splitter, oof_storage=None):
    param = {
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": 1,
        "verbosity": 0,
    }

    fold_scores, fold_preds, fold_indices = [], [], []

    for t_idx, v_idx in splitter.split(y):
        X_train, y_train = X.iloc[t_idx], y.iloc[t_idx]
        X_val, y_val = X.iloc[v_idx], y.iloc[v_idx]
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
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


def train_final_xgb(X_train, y_train, X_test, best_params):
    """Entrena XGBoost final con todo el train y predice en test."""
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)
    return predictions, model


# ═══════════════════════════════════════════════════════════════════
#  Base LSTM
# ═══════════════════════════════════════════════════════════════════

class BaseLSTMModel(nn.Module):
    """LSTM base de 2 capas para capturar dependencias temporales."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if 'weight_ih' in n: nn.init.xavier_uniform_(p)
            elif 'weight_hh' in n: nn.init.orthogonal_(p)
            elif 'bias' in n: nn.init.zeros_(p)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class BaseLSTMDataset(Dataset):
    """Dataset para ventanas deslizantes del BaseLSTM."""
    def __init__(self, X, y, window_size):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.window_size = window_size
        self.n_samples = len(X) - window_size

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        x_window = self.X[idx:idx + self.window_size]
        target = self.y[idx + self.window_size - 1]
        return torch.from_numpy(x_window), torch.tensor(target)


def _train_base_lstm(X_np, y_np, window_size, hidden_size, num_layers,
                     dropout, lr, weight_decay, epochs, batch_size,
                     patience, device):
    """Entrenamiento interno del BaseLSTM con early stopping."""
    input_size = X_np.shape[1]
    ds = BaseLSTMDataset(X_np, y_np, window_size)
    if len(ds) < 20:
        return None, float('inf')

    tr_sz = int(len(ds) * 0.8)
    vl_sz = len(ds) - tr_sz
    if vl_sz < 5:
        return None, float('inf')

    tr_loader = DataLoader(
        torch.utils.data.Subset(ds, list(range(tr_sz))),
        batch_size=batch_size, shuffle=True
    )
    vl_loader = DataLoader(
        torch.utils.data.Subset(ds, list(range(tr_sz, len(ds)))),
        batch_size=batch_size, shuffle=False
    )

    model = BaseLSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2, min_lr=1e-6
    )

    best_vl, pat_cnt, best_st = float('inf'), 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        ep_vl = 0.0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.to(device)
                ep_vl += criterion(model(xb), yb).item() * xb.size(0)
        ep_vl /= vl_sz
        scheduler.step(ep_vl)

        if ep_vl < best_vl:
            best_vl = ep_vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    if best_st:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})

    return model, best_vl


def objective_base_lstm_global(trial, X, y, splitter, device=None,
                               oof_storage=None):
    """Optuna objective para BaseLSTM con Walk-Forward + OOF."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ws = trial.suggest_int('window_size', 5, 30)
    hs = trial.suggest_categorical('hidden_size', [32, 64, 128])
    nl = trial.suggest_int('num_layers', 1, 3)
    do = trial.suggest_float('dropout', 0.0, 0.4)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    bs = trial.suggest_categorical('batch_size', [16, 32, 64])

    X_v = np.nan_to_num(
        (X.values if hasattr(X, 'values') else np.array(X)).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    y_v = np.nan_to_num(
        (y.values if hasattr(y, 'values') else np.array(y)).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    fold_scores, fold_preds, fold_indices = [], [], []

    for fold_num, (t_idx, v_idx) in enumerate(splitter.split(y)):
        ts, te = int(t_idx[0]), int(t_idx[-1]) + 1
        if te - ts < ws + 10:
            continue

        X_tr, y_tr = X_v[ts:te], y_v[ts:te]
        input_size = X_tr.shape[1]

        # Train model on this fold
        model = BaseLSTMModel(input_size, hs, nl, do).to(device)
        ds = BaseLSTMDataset(X_tr, y_tr, ws)
        if len(ds) < 10:
            continue

        loader = DataLoader(ds, batch_size=bs, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        best_loss, pat_cnt = float('inf'), 0
        for _ in range(50):
            model.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                if torch.isnan(loss):
                    return float('inf')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            ml = np.mean(losses)
            if ml < best_loss:
                best_loss, pat_cnt = ml, 0
            else:
                pat_cnt += 1
                if pat_cnt >= 8:
                    break

        # Predict on validation indices
        model.eval()
        vp, vi = [], []
        with torch.no_grad():
            for ti in v_idx:
                ti = int(ti)
                w_start = ti - ws + 1
                if w_start < 0 or ti >= len(X_v):
                    continue
                x_window = X_v[w_start:ti + 1]
                if len(x_window) != ws:
                    continue
                xt = torch.from_numpy(x_window).unsqueeze(0).to(device)
                vp.append(float(model(xt).cpu().item()))
                vi.append(ti)

        if not vp:
            continue

        vp, vi = np.array(vp), np.array(vi)
        vt = y_v[vi]
        fm = mean_absolute_error(vt, vp)
        if np.isnan(fm):
            continue
        fold_scores.append(fm)
        fold_preds.append(vp)
        fold_indices.append(vi)

    if not fold_scores:
        return float('inf')

    ms = float(np.mean(fold_scores))

    if oof_storage is not None:
        if 'best_score' not in oof_storage or ms < oof_storage['best_score']:
            oof_storage['best_score'] = ms
            oof_storage['params'] = {
                'window_size': ws, 'hidden_size': hs, 'num_layers': nl,
                'dropout': do, 'lr': lr, 'weight_decay': wd, 'batch_size': bs
            }
            oof_storage['preds'] = fold_preds
            oof_storage['indices'] = fold_indices

    return ms


def train_final_base_lstm(X_train, y_train, X_test, best_params, device):
    """Entrena BaseLSTM final con todo el train y predice en test."""
    ws = best_params.get('window_size', 10)
    hs = best_params.get('hidden_size', 64)
    nl = best_params.get('num_layers', 2)
    do = best_params.get('dropout', 0.1)
    lr = best_params.get('lr', 1e-3)
    wd = best_params.get('weight_decay', 1e-5)
    bs = best_params.get('batch_size', 32)

    X_tr_v = np.nan_to_num(
        (X_train.values if hasattr(X_train, 'values') else np.array(X_train)).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    y_tr_v = np.nan_to_num(
        (y_train.values if hasattr(y_train, 'values') else np.array(y_train)).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    X_te_v = np.nan_to_num(
        (X_test.values if hasattr(X_test, 'values') else np.array(X_test)).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    model, _ = _train_base_lstm(
        X_tr_v, y_tr_v, ws, hs, nl, do, lr, wd,
        epochs=100, batch_size=bs, patience=15, device=device
    )

    if model is None:
        return np.full(len(X_te_v), np.nan), None

    # Concatenar train + test features para ventanas rolling
    full_X = np.concatenate([X_tr_v, X_te_v], axis=0)
    train_len = len(X_tr_v)

    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X_te_v)):
            global_idx = train_len + i
            w_start = global_idx - ws + 1
            if w_start < 0:
                predictions.append(np.nan)
                continue
            x_window = full_X[w_start:global_idx + 1]
            if len(x_window) != ws:
                predictions.append(np.nan)
                continue
            xt = torch.from_numpy(x_window).unsqueeze(0).to(device)
            predictions.append(float(model(xt).cpu().item()))

    return np.array(predictions), model


# ═══════════════════════════════════════════════════════════════════
#  Stacking Meta-Learner LSTM (2 capas, pesos convexos dinámicos)
# ═══════════════════════════════════════════════════════════════════

class StackingMetaLSTM(nn.Module):
    """
    Meta-Learner LSTM de 2 capas con ponderación dinámica convexa.
    Produce pesos α_t, β_t, γ_t, δ_t via softmax (suman 1, ≥0).
    ŷ_stacked = α_t·ŷ_Cat + β_t·ŷ_LGB + γ_t·ŷ_XGB + δ_t·ŷ_LSTM_base
    """
    def __init__(self, num_models=4, hidden_size=64, dropout=0.1, temperature=1.0):
        super().__init__()
        self.num_models = num_models
        self.temperature = max(0.1, temperature)
        # Estrictamente 2 capas LSTM según el paper
        self.lstm = nn.LSTM(
            input_size=num_models, hidden_size=hidden_size,
            num_layers=2, batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, num_models)
        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if 'weight_ih' in n: nn.init.xavier_uniform_(p)
            elif 'weight_hh' in n: nn.init.orthogonal_(p)
            elif 'bias' in n: nn.init.zeros_(p)
            elif 'fc.weight' in n: nn.init.xavier_uniform_(p)

    def _compute_weights(self, z_t):
        """Softmax puro → combinación convexa (suman 1, ≥0)."""
        return F.softmax(z_t / self.temperature, dim=-1)

    def forward(self, x, return_weights=False):
        # x: (batch, window, num_models) — predicciones de los 4 modelos base
        p_t = x[:, -1, :]           # última predicción de cada modelo base
        lstm_out, _ = self.lstm(x)
        h_t = lstm_out[:, -1, :]    # hidden state del último paso
        alpha_t = self._compute_weights(self.fc(h_t))  # pesos convexos
        y_hat = torch.sum(alpha_t * p_t, dim=-1)       # combinación ponderada
        return (y_hat, alpha_t) if return_weights else y_hat


class StackingMetaDataset(Dataset):
    """Dataset para el meta-learner: ventanas de predicciones OOF."""
    def __init__(self, oof_matrix, y_true, window_size, noise_std=0.0, training=True):
        self.oof_matrix = oof_matrix.astype(np.float32)
        self.y_true = y_true.astype(np.float32)
        self.window_size = window_size
        self.noise_std = noise_std
        self.training = training
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        return _fast_valid_indices(self.oof_matrix, self.y_true, self.window_size)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        X_t = self.oof_matrix[t - self.window_size + 1:t + 1].copy()
        if self.training and self.noise_std > 0:
            X_t += np.random.normal(0, self.noise_std, X_t.shape).astype(np.float32)
        return torch.from_numpy(X_t), torch.tensor(self.y_true[t])


def build_oof_dataframe_sota(oof_lgb, oof_cb, oof_xgb, oof_lstm, y):
    """Construye DataFrame OOF alineando predicciones de los 4 modelos SOTA."""
    from preprocessing.oof_generators import collect_oof_predictions

    preds_lgb, idx_lgb, _ = collect_oof_predictions(oof_lgb)
    preds_cb, idx_cb, _ = collect_oof_predictions(oof_cb)
    preds_xgb, idx_xgb, _ = collect_oof_predictions(oof_xgb)
    preds_lstm, idx_lstm, _ = collect_oof_predictions(oof_lstm)

    df_lgb = pd.DataFrame({'idx': idx_lgb, 'lgb': preds_lgb})
    df_cb = pd.DataFrame({'idx': idx_cb, 'catboost': preds_cb})
    df_xgb = pd.DataFrame({'idx': idx_xgb, 'xgboost': preds_xgb})
    df_lstm = pd.DataFrame({'idx': idx_lstm, 'base_lstm': preds_lstm})

    y_array = y.values if isinstance(y, pd.Series) else np.array(y)

    merged = df_lgb.merge(df_cb, on='idx', how='inner')
    merged = merged.merge(df_xgb, on='idx', how='inner')
    merged = merged.merge(df_lstm, on='idx', how='inner')
    merged['target'] = merged['idx'].apply(
        lambda i: y_array[int(i)] if int(i) < len(y_array) else np.nan
    )
    merged = merged.dropna().sort_values('idx').reset_index(drop=True)
    return merged


def _train_stacking_meta(oof_df, window_size=10, hidden_size=64, dropout=0.1,
                         lr=1e-3, weight_decay=1e-5, epochs=100, batch_size=32,
                         patience=15, device=None, noise_std=0.0, temperature=2.0):
    """Entrenamiento del StackingMetaLSTM."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cols = [c for c in oof_df.columns if c not in ['idx', 'target']]
    oof_matrix = oof_df[model_cols].values
    y_array = oof_df['target'].values
    num_models = len(model_cols)

    full_ds = StackingMetaDataset(oof_matrix, y_array, window_size, noise_std=0.0, training=False)
    if len(full_ds) < 20:
        print(f"  Dataset muy pequeño: {len(full_ds)} muestras")
        return None, None, None, None

    tr_sz = int(len(full_ds) * 0.8)
    vl_sz = len(full_ds) - tr_sz
    if vl_sz < 5:
        print(f"  Validation set muy pequeño: {vl_sz}")
        return None, None, None, None

    tr_idx = list(range(tr_sz))
    vl_idx = list(range(tr_sz, len(full_ds)))

    tr_ds = StackingMetaDataset(oof_matrix, y_array, window_size, noise_std=noise_std, training=True)
    vl_ds = StackingMetaDataset(oof_matrix, y_array, window_size, noise_std=0.0, training=False)
    tr_loader = DataLoader(torch.utils.data.Subset(tr_ds, tr_idx), batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(torch.utils.data.Subset(vl_ds, vl_idx), batch_size=batch_size, shuffle=False)

    model = StackingMetaLSTM(
        num_models=num_models, hidden_size=hidden_size,
        dropout=dropout, temperature=temperature
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2, min_lr=1e-6
    )

    best_vl, pat_cnt, best_st, best_ep = float('inf'), 0, None, 1
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        ep_vl = 0.0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.to(device)
                ep_vl += criterion(model(xb), yb).item() * xb.size(0)
        ep_vl /= vl_sz
        scheduler.step(ep_vl)

        if ep_vl < best_vl:
            best_vl = ep_vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt, best_ep = 0, ep + 1
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    if best_st:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})

    model.eval()
    all_p, all_t, all_w = [], [], []
    with torch.no_grad():
        for xb, yb in DataLoader(full_ds, batch_size=batch_size, shuffle=False):
            yp, wt = model(xb.to(device), return_weights=True)
            all_p.extend(yp.cpu().numpy())
            all_t.extend(yb.numpy())
            all_w.extend(wt.cpu().numpy())

    all_p, all_t, all_w = np.array(all_p), np.array(all_t), np.array(all_w)
    mae_m = mean_absolute_error(all_t, all_p)

    results = {
        'best_epoch': best_ep, 'mae': mae_m,
        'predictions': all_p, 'targets': all_t,
        'weights': all_w, 'valid_indices': full_ds.valid_indices,
        'window_size': window_size, 'model_names': model_cols,
        'weights_mean_per_model': all_w.mean(axis=0)
    }

    print(f"  Pesos promedio (convexos): {dict(zip(model_cols, all_w.mean(axis=0).round(4)))}")
    print(f"  Suma pesos promedio: {all_w.mean(axis=0).sum():.4f}")
    return model, mae_m, results, device


def _objective_stacking_meta(trial, oof_df, device):
    """Optuna objective para el StackingMetaLSTM."""
    ws = trial.suggest_int('window_size', 5, 30)
    hs = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    do = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    bs = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    ns = trial.suggest_float('noise_std', 0.0, 0.05)
    tp = trial.suggest_float('temperature', 0.5, 5.0)

    model, mae, res, _ = _train_stacking_meta(
        oof_df=oof_df, window_size=ws, hidden_size=hs, dropout=do,
        lr=lr, weight_decay=wd, epochs=100, batch_size=bs,
        patience=15, device=device, noise_std=ns, temperature=tp
    )
    if model is None or mae is None:
        return float('inf')
    return mae


def optimize_stacking_meta(oof_df, device, n_trials=50):
    """Optimiza y entrena el StackingMetaLSTM con Optuna."""
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3))
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda t: _objective_stacking_meta(t, oof_df, device),
        n_trials=n_trials, n_jobs=1, show_progress_bar=True
    )

    bp = study.best_params
    print("\n=== Mejores Hiperparametros (Stacking Meta) ===")
    for k, v in bp.items():
        print(f"  {k}: {v}")

    model, mae, res, device = _train_stacking_meta(
        oof_df=oof_df,
        window_size=bp.get('window_size', 10),
        hidden_size=bp.get('hidden_size', 64),
        dropout=bp.get('dropout', 0.2),
        lr=bp.get('lr', 1e-3),
        weight_decay=bp.get('weight_decay', 1e-4),
        epochs=200,
        batch_size=bp.get('batch_size', 32),
        patience=25,
        device=device,
        noise_std=bp.get('noise_std', 0.01),
        temperature=bp.get('temperature', 2.0)
    )
    return model, mae, res, bp, study
