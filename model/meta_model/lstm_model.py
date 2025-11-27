
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error

# ============================================
# MODELO LSTM META-LEARNER
# ============================================

class LSTMMetaLearner(nn.Module):
    def __init__(self, num_models, hidden_size=64, num_layers=2, dropout=0.1):
        super(LSTMMetaLearner, self).__init__()
        self.num_models = num_models
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=num_models,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_models)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_weights=False):
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_t = lstm_out[:, -1, :]
        z_t = self.fc(h_t)
        alpha_t = self.softmax(z_t)
        p_t = x[:, -1, :]
        y_hat = torch.sum(alpha_t * p_t, dim=-1)
        if return_weights:
            return y_hat, alpha_t
        return y_hat

# ============================================
# DATASET PARA META-MODELO
# ============================================

class MetaDataset(Dataset):
    def __init__(self, oof_matrix, y_true, window_size):
        self.oof_matrix = oof_matrix
        self.y_true = y_true
        self.window_size = window_size
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        valid = []
        n = len(self.y_true)
        for t in range(self.window_size - 1, n):
            start_idx = t - self.window_size + 1
            window_data = self.oof_matrix[start_idx:t+1]
            if not np.isnan(window_data).any() and not np.isnan(self.y_true[t]):
                valid.append(t)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        start_idx = t - self.window_size + 1
        X_t = self.oof_matrix[start_idx:t+1].astype(np.float32)
        y_t = np.float32(self.y_true[t])
        return torch.from_numpy(X_t), torch.tensor(y_t)

# ============================================
# FUNCION DE ENTRENAMIENTO
# ============================================

def train_lstm_meta_model(oof_df, window_size=10, hidden_size=64, num_layers=2, dropout=0.1, lr=1e-3, epochs=100, batch_size=32, patience=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Extraer matriz OOF y targets
    model_cols = [c for c in oof_df.columns if c not in ['idx', 'target']]
    oof_matrix = oof_df[model_cols].values
    y_array = oof_df['target'].values
    num_models = len(model_cols)
    dataset = MetaDataset(oof_matrix, y_array, window_size)
    if len(dataset) < 20:
        return None, None, None, None
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    if val_size < 1:
        return None, None, None, None
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = LSTMMetaLearner(
        num_models=num_models,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss /= len(val_dataset)
        val_losses.append(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets, all_weights = [], [], []
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X_batch, y_batch in full_loader:
            X_batch = X_batch.to(device)
            y_pred, weights = model(X_batch, return_weights=True)
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            all_weights.extend(weights.cpu().numpy())
    mae_meta = mean_absolute_error(all_targets, all_preds)
    mse_meta = np.mean((np.array(all_targets) - np.array(all_preds))**2)
    results = {
        'mae': mae_meta,
        'mse': mse_meta,
        'rmse': np.sqrt(mse_meta),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch if 'best_epoch' in dir() else len(train_losses),
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
        'weights': np.array(all_weights),
        'valid_indices': dataset.valid_indices,
        'window_size': window_size,
        'model_names': model_cols
    }
    return model, mae_meta, results, device

# ============================================
# OPTIMIZACION CON OPTUNA
# ============================================

def objective_lstm_meta(trial, oof_df, device):
    """Objetivo de Optuna para optimizar hiperparametros del LSTM"""
    window_size = trial.suggest_int('window_size', 5, 30)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    model, mae, results, _ = train_lstm_meta_model(
        oof_df=oof_df,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        epochs=150,
        batch_size=batch_size,
        patience=15,
        device=device
    )
    if model is None or mae is None:
        return float('inf')
    return mae

def optimize_lstm_meta(oof_df, device, n_trials=50):
    """Ejecuta optimizacion con Optuna y retorna mejor modelo"""
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective_lstm_meta(trial, oof_df, device),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )
    best_params = study.best_params
    # Reentrenar con mejores parametros y mas epochs
    model, mae, results, device = train_lstm_meta_model(
        oof_df=oof_df,
        window_size=best_params['window_size'],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        lr=best_params['lr'],
        epochs=300,
        batch_size=best_params['batch_size'],
        patience=30,
        device=device
    )
    return model, mae, results, best_params, study

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def predict_with_meta_model(model, oof_df, window_size, device):
    model.eval()
    model_cols = [c for c in oof_df.columns if c not in ['idx', 'target']]
    oof_matrix = oof_df[model_cols].values
    n = len(oof_matrix)
    predictions, weights_history = [], []
    with torch.no_grad():
        for t in range(window_size - 1, n):
            start_idx = t - window_size + 1
            window_data = oof_matrix[start_idx:t+1]
            if np.isnan(window_data).any():
                predictions.append(np.nan)
                weights_history.append(np.full(len(model_cols), np.nan))
                continue
            X_t = torch.from_numpy(window_data.astype(np.float32)).unsqueeze(0).to(device)
            y_pred, alpha = model(X_t, return_weights=True)
            predictions.append(y_pred.cpu().item())
            weights_history.append(alpha.cpu().numpy().flatten())
    pred_array = np.full(n, np.nan)
    pred_array[window_size-1:] = predictions
    return pred_array, np.array(weights_history)

def get_average_weights(weights_history, model_names):
    if len(weights_history) == 0:
        return pd.DataFrame({'Modelo': model_names, 'Peso_Promedio': [np.nan]*len(model_names)})
    valid_weights = weights_history[~np.isnan(weights_history).any(axis=1)]
    if len(valid_weights) == 0:
        return pd.DataFrame({'Modelo': model_names, 'Peso_Promedio': [np.nan]*len(model_names)})
    avg_weights = np.mean(valid_weights, axis=0)
    return pd.DataFrame({'Modelo': model_names, 'Peso_Promedio': avg_weights})

def get_oof_stats(oof_df):
    """Retorna estadisticas de la matriz OOF"""
    model_cols = [c for c in oof_df.columns if c not in ['idx', 'target']]
    stats = {
        'total_rows': len(oof_df),
        'valid_rows': len(oof_df.dropna()),
        'models': model_cols,
        'per_model': {}
    }
    for col in model_cols:
        stats['per_model'][col] = {
            'valid': oof_df[col].notna().sum(),
            'mean': oof_df[col].mean(),
            'std': oof_df[col].std()
        }
    return stats