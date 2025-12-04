import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error

class LSTMMetaLearner(nn.Module):
    def __init__(self, num_models, hidden_size=64, num_layers=2, dropout=0.1, min_weight=0.05, temperature=1.0):
        super(LSTMMetaLearner, self).__init__()
        self.num_models = num_models
        self.hidden_size = hidden_size
        self.temperature = max(0.1, temperature)
        max_min_weight = (1.0 / num_models) - 0.01
        self.min_weight = max(0.01, min(min_weight, max_min_weight))
        self.lstm = nn.LSTM(input_size=num_models, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, num_models)
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.zeros_(param)
            elif 'fc.weight' in name: nn.init.xavier_uniform_(param)
    
    def _compute_weights(self, z_t):
        # Temperature scaling para suavizar distribucion
        z_scaled = z_t / self.temperature
        # Softmax suavizado
        softmax_weights = F.softmax(z_scaled, dim=-1)
        # Garantizar minimo peso para cada modelo
        scale_factor = 1.0 - self.num_models * self.min_weight
        alpha_t = self.min_weight + scale_factor * softmax_weights
        return alpha_t
    
    def forward(self, x, return_weights=False):
        batch_size = x.size(0)
        p_t = x[:, -1, :]
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_t = lstm_out[:, -1, :]
        z_t = self.fc(h_t)
        alpha_t = self._compute_weights(z_t)
        y_hat = torch.sum(alpha_t * p_t, dim=-1)
        if return_weights: return y_hat, alpha_t
        return y_hat

class MetaDataset(Dataset):
    def __init__(self, oof_matrix, y_true, window_size, noise_std=0.0, training=True):
        self.oof_matrix = oof_matrix.astype(np.float32)
        self.y_true = y_true.astype(np.float32)
        self.window_size = window_size
        self.noise_std = noise_std
        self.training = training
        self.valid_indices = self._get_valid_indices()
    
    def _get_valid_indices(self):
        valid = []
        n = len(self.y_true)
        for t in range(self.window_size - 1, n):
            start_idx = t - self.window_size + 1
            window_data = self.oof_matrix[start_idx:t+1]
            if not np.isnan(window_data).any() and not np.isnan(self.y_true[t]): valid.append(t)
        return valid
    
    def __len__(self): return len(self.valid_indices)
    
    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        start_idx = t - self.window_size + 1
        X_t = self.oof_matrix[start_idx:t+1].copy()
        y_t = self.y_true[t]
        if self.training and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, X_t.shape).astype(np.float32)
            X_t = X_t + noise
        return torch.from_numpy(X_t), torch.tensor(y_t)

def train_lstm_meta_model(oof_df, window_size=10, hidden_size=64, num_layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-5, epochs=100, batch_size=32, patience=15, device=None, noise_std=0.0, min_weight=0.05, temperature=2.0):
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cols = [c for c in oof_df.columns if c not in ['idx', 'target']]
    oof_matrix = oof_df[model_cols].values
    y_array = oof_df['target'].values
    num_models = len(model_cols)
    full_dataset = MetaDataset(oof_matrix, y_array, window_size, noise_std=0.0, training=False)
    if len(full_dataset) < 20:
        print(f"Dataset muy pequeno: {len(full_dataset)} muestras")
        return None, None, None, None
    train_size = int(len(full_dataset) * 0.8)
    val_size = len(full_dataset) - train_size
    if val_size < 5:
        print(f"Validation set muy pequeno: {val_size}")
        return None, None, None, None
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    train_dataset = MetaDataset(oof_matrix, y_array, window_size, noise_std=noise_std, training=True)
    val_dataset = MetaDataset(oof_matrix, y_array, window_size, noise_std=0.0, training=False)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    model = LSTMMetaLearner(num_models=num_models, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, min_weight=min_weight, temperature=temperature).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2, min_lr=1e-6)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []
    best_epoch = 1
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss /= len(train_subset)
        train_losses.append(epoch_train_loss)
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss /= len(val_subset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience: break
    if best_state is not None: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    all_preds, all_targets, all_weights = [], [], []
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X_batch, y_batch in full_loader:
            X_batch = X_batch.to(device)
            y_pred, weights = model(X_batch, return_weights=True)
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            all_weights.extend(weights.cpu().numpy())
    weights_array = np.array(all_weights)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae_meta = mean_absolute_error(all_targets, all_preds)
    mse_meta = np.mean((all_targets - all_preds)**2)
    results = {'train_losses': train_losses, 'val_losses': val_losses, 'best_epoch': best_epoch, 'mae': mae_meta, 'mse': mse_meta, 'rmse': np.sqrt(mse_meta), 'predictions': all_preds, 'targets': all_targets, 'weights': weights_array, 'valid_indices': full_dataset.valid_indices, 'window_size': window_size, 'model_names': model_cols, 'weights_min': weights_array.min(), 'weights_max': weights_array.max(), 'weights_mean_per_model': weights_array.mean(axis=0)}
    print(f"Pesos promedio por modelo: {dict(zip(model_cols, weights_array.mean(axis=0).round(4)))}")
    print(f"Min peso: {weights_array.min():.4f}, Max peso: {weights_array.max():.4f}")
    return model, mae_meta, results, device

def objective_lstm_meta(trial, oof_df, device):
    window_size = trial.suggest_int('window_size', 5, 30)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    noise_std = trial.suggest_float('noise_std', 0.0, 0.05)
    min_weight = trial.suggest_float('min_weight', 0.02, 0.20)
    temperature = trial.suggest_float('temperature', 1.0, 5.0)
    model, mae, results, _ = train_lstm_meta_model(oof_df=oof_df, window_size=window_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, lr=lr, weight_decay=weight_decay, epochs=100, batch_size=batch_size, patience=15, device=device, noise_std=noise_std, min_weight=min_weight, temperature=temperature)
    if model is None or mae is None: return float('inf')
    weights = results.get('weights', np.array([]))
    if len(weights) > 0:
        weight_std = np.std(weights.mean(axis=0))
        if weight_std < 0.01: return float('inf')
    return mae

def optimize_lstm_meta(oof_df, device, n_trials=50):
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3))
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective_lstm_meta(trial, oof_df, device), n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    best_params = study.best_params
    print("\n=== Mejores Hiperparametros ===")
    for k, v in best_params.items(): print(f"  {k}: {v}")
    model, mae, results, device = train_lstm_meta_model(oof_df=oof_df, window_size=best_params.get('window_size', 10), hidden_size=best_params.get('hidden_size', 64), num_layers=best_params.get('num_layers', 2), dropout=best_params.get('dropout', 0.2), lr=best_params.get('lr', 1e-3), weight_decay=best_params.get('weight_decay', 1e-4), epochs=200, batch_size=best_params.get('batch_size', 32), patience=25, device=device, noise_std=best_params.get('noise_std', 0.01), min_weight=best_params.get('min_weight', 0.05), temperature=best_params.get('temperature', 2.0))
    return model, mae, results, best_params, study

def get_average_weights(weights_history, model_names):
    if len(weights_history) == 0: return pd.DataFrame({'Modelo': model_names, 'Peso_Promedio': [np.nan] * len(model_names)})
    valid_weights = weights_history[~np.isnan(weights_history).any(axis=1)]
    if len(valid_weights) == 0: return pd.DataFrame({'Modelo': model_names, 'Peso_Promedio': [np.nan] * len(model_names)})
    return pd.DataFrame({'Modelo': model_names, 'Peso_Promedio': np.mean(valid_weights, axis=0), 'Peso_Std': np.std(valid_weights, axis=0), 'Peso_Min': np.min(valid_weights, axis=0), 'Peso_Max': np.max(valid_weights, axis=0)})

def collect_oof_predictions(oof_storage):
    if not oof_storage or 'preds' not in oof_storage or 'indices' not in oof_storage: return np.array([]), np.array([]), 0.0
    all_preds, all_indices = [], []
    for preds, indices in zip(oof_storage['preds'], oof_storage['indices']):
        preds_flat = np.array(preds).flatten()
        indices_flat = np.array(indices).flatten()
        min_len = min(len(preds_flat), len(indices_flat))
        all_preds.extend(preds_flat[:min_len])
        all_indices.extend(indices_flat[:min_len])
    return np.array(all_preds), np.array(all_indices), oof_storage.get('best_score', 0.0)

def build_oof_dataframe(oof_lgb, oof_cb, oof_tx, oof_moirai, y_train):
    y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    preds_lgb, idx_lgb, _ = collect_oof_predictions(oof_lgb)
    preds_cb, idx_cb, _ = collect_oof_predictions(oof_cb)
    preds_tx, idx_tx, _ = collect_oof_predictions(oof_tx)
    preds_moirai, idx_moirai, _ = collect_oof_predictions(oof_moirai)
    dfs = []
    if len(preds_lgb) > 0: dfs.append(pd.DataFrame({'idx': idx_lgb.astype(int), 'pred_lgb': preds_lgb}))
    if len(preds_cb) > 0: dfs.append(pd.DataFrame({'idx': idx_cb.astype(int), 'pred_catboost': preds_cb}))
    if len(preds_tx) > 0: dfs.append(pd.DataFrame({'idx': idx_tx.astype(int), 'pred_timexer': preds_tx}))
    if len(preds_moirai) > 0: dfs.append(pd.DataFrame({'idx': idx_moirai.astype(int), 'pred_moirai': preds_moirai}))
    if not dfs: return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]: result = pd.merge(result, df, on='idx', how='inner')
    result['target'] = result['idx'].apply(lambda i: y_array[i] if i < len(y_array) else np.nan)
    result = result.dropna().reset_index(drop=True)
    return result