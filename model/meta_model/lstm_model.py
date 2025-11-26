import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error

def collect_oof_predictions(oof_storage, n_samples):
    if oof_storage is None or 'preds' not in oof_storage:
        return np.full(n_samples, np.nan)
    oof_array = np.full(n_samples, np.nan)
    for preds, indices in zip(oof_storage['preds'], oof_storage['indices']):
        preds_flat = np.array(preds).flatten()
        indices_flat = np.array(indices).flatten()
        min_len = min(len(preds_flat), len(indices_flat))
        for i in range(min_len):
            idx = indices_flat[i]
            if 0 <= idx < n_samples:
                if np.isnan(oof_array[idx]):
                    oof_array[idx] = preds_flat[i]
                else:
                    oof_array[idx] = (oof_array[idx] + preds_flat[i]) / 2
    return oof_array

def build_oof_dataframe(oof_lgb, oof_cb, oof_tx, oof_moirai, n_samples):
    return pd.DataFrame({'lgb': collect_oof_predictions(oof_lgb, n_samples), 'catboost': collect_oof_predictions(oof_cb, n_samples), 'timexer': collect_oof_predictions(oof_tx, n_samples), 'moirai': collect_oof_predictions(oof_moirai, n_samples)})

class LSTMMetaLearner(nn.Module):
    def __init__(self, num_models, hidden_size=64, num_layers=2, dropout=0.1):
        super(LSTMMetaLearner, self).__init__()
        self.num_models = num_models
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=num_models, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
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

def train_lstm_meta_model(oof_df, y, window_size=10, hidden_size=64, num_layers=2, dropout=0.1, lr=1e-3, epochs=100, batch_size=32, patience=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oof_matrix = oof_df.values
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    num_models = oof_matrix.shape[1]
    dataset = MetaDataset(oof_matrix, y_array, window_size)
    if len(dataset) < 20:
        return None, None, None, None
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = LSTMMetaLearner(num_models=num_models, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
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
    results = {'mae': mae_meta, 'mse': mse_meta, 'rmse': np.sqrt(mse_meta), 'train_losses': train_losses, 'val_losses': val_losses, 'best_epoch': len(train_losses) - patience if patience_counter >= patience else len(train_losses), 'predictions': np.array(all_preds), 'targets': np.array(all_targets), 'weights': np.array(all_weights), 'valid_indices': dataset.valid_indices, 'window_size': window_size}
    return model, mae_meta, results, device

def predict_with_meta_model(model, oof_df, window_size, device):
    model.eval()
    oof_matrix = oof_df.values
    n = len(oof_matrix)
    predictions, weights_history = [], []
    with torch.no_grad():
        for t in range(window_size - 1, n):
            start_idx = t - window_size + 1
            window_data = oof_matrix[start_idx:t+1]
            if np.isnan(window_data).any():
                predictions.append(np.nan)
                weights_history.append(np.full(oof_df.shape[1], np.nan))
                continue
            X_t = torch.from_numpy(window_data.astype(np.float32)).unsqueeze(0).to(device)
            y_pred, alpha = model(X_t, return_weights=True)
            predictions.append(y_pred.cpu().item())
            weights_history.append(alpha.cpu().numpy().flatten())
    pred_array = np.full(n, np.nan)
    pred_array[window_size-1:] = predictions
    return pred_array, np.array(weights_history)

def get_average_weights(weights_history, model_names):
    valid_weights = weights_history[~np.isnan(weights_history).any(axis=1)]
    if len(valid_weights) == 0:
        return pd.DataFrame({'Modelo': model_names, 'Peso_Promedio': [np.nan]*len(model_names)})
    avg_weights = np.mean(valid_weights, axis=0)
    return pd.DataFrame({'Modelo': model_names, 'Peso_Promedio': avg_weights})

def adaptive_forecast(meta_model, base_models_params, X_test, y_test, oof_df_train, y_train, window_size, device, splitter_func, seq_len=96, pred_len=30, model_size='small', freq='D', adaptation_lr=1e-4, adaptation_steps=1):
    meta_model.eval()
    n_test = len(y_test)
    n_train = len(y_train)
    model_names = ['lgb', 'catboost', 'timexer', 'moirai']
    num_models = len(model_names)
    oof_history = oof_df_train.values.copy()
    y_history = y_train.values.copy() if isinstance(y_train, pd.Series) else y_train.copy()
    predictions_meta = []
    predictions_base = {name: [] for name in model_names}
    weights_history = []
    adaptation_losses = []
    lgb_model = lgb.LGBMRegressor(**base_models_params['lgb'])
    lgb_model.fit(X_test.iloc[:1], y_test.iloc[:1]) if len(X_test) > 0 else None
    cb_params = base_models_params['catboost'].copy()
    cb_params['iterations'] = min(cb_params.get('iterations', 100), 100)
    cb_model = CatBoostRegressor(**cb_params)
    X_train_full = pd.concat([X_test.iloc[:0]], axis=0) if len(X_test) > 0 else None
    from model.bases_models.ligthGBM_model import lgb
    from catboost import CatBoostRegressor
    lgb_params = base_models_params['lgb'].copy()
    lgb_params['verbose'] = -1
    cb_params = base_models_params['catboost'].copy()
    cb_params['verbose'] = 0
    cb_params['allow_writing_files'] = False
    optimizer_adapt = torch.optim.Adam(meta_model.parameters(), lr=adaptation_lr)
    criterion_adapt = nn.MSELoss()
    test_start_idx = n_train
    X_full = pd.concat([X_test.iloc[:0].reindex_like(X_test), X_test], axis=0).reset_index(drop=True) if hasattr(X_test, 'iloc') else X_test
    for t in range(n_test):
        current_idx = n_train + t
        X_current = X_test.iloc[[t]] if hasattr(X_test, 'iloc') else X_test[t:t+1]
        base_preds_t = np.zeros(num_models)
        train_end_idx = n_train + t
        X_train_current = pd.concat([X_test.iloc[:t]], axis=0) if t > 0 else pd.DataFrame()
        y_train_current = y_test.iloc[:t] if t > 0 else pd.Series(dtype=float)
        if t >= 30:
            X_lgb_train = X_test.iloc[:t]
            y_lgb_train = y_test.iloc[:t]
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_lgb_train, y_lgb_train)
            base_preds_t[0] = lgb_model.predict(X_current)[0]
            cb_model = CatBoostRegressor(**cb_params)
            cb_model.fit(X_lgb_train, y_lgb_train, verbose=False)
            base_preds_t[1] = cb_model.predict(X_current)[0]
        else:
            base_preds_t[0] = oof_history[-1, 0] if len(oof_history) > 0 else 0.0
            base_preds_t[1] = oof_history[-1, 1] if len(oof_history) > 0 else 0.0
        base_preds_t[2] = oof_history[-1, 2] if len(oof_history) > 0 else 0.0
        base_preds_t[3] = oof_history[-1, 3] if len(oof_history) > 0 else 0.0
        for i, name in enumerate(model_names):
            predictions_base[name].append(base_preds_t[i])
        oof_history = np.vstack([oof_history, base_preds_t])
        if len(oof_history) >= window_size:
            window_data = oof_history[-window_size:]
            X_meta = torch.from_numpy(window_data.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                y_pred_meta, alpha_t = meta_model(X_meta, return_weights=True)
            pred_meta = y_pred_meta.cpu().item()
            weights_t = alpha_t.cpu().numpy().flatten()
        else:
            pred_meta = np.mean(base_preds_t)
            weights_t = np.ones(num_models) / num_models
        predictions_meta.append(pred_meta)
        weights_history.append(weights_t)
        y_true_t = y_test.iloc[t] if isinstance(y_test, pd.Series) else y_test[t]
        y_history = np.append(y_history, y_true_t)
        if len(oof_history) >= window_size:
            meta_model.train()
            for _ in range(adaptation_steps):
                window_data = oof_history[-window_size:]
                X_adapt = torch.from_numpy(window_data.astype(np.float32)).unsqueeze(0).to(device)
                y_adapt = torch.tensor([y_true_t], dtype=torch.float32).to(device)
                optimizer_adapt.zero_grad()
                y_pred_adapt = meta_model(X_adapt)
                loss_adapt = criterion_adapt(y_pred_adapt, y_adapt)
                loss_adapt.backward()
                torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
                optimizer_adapt.step()
            adaptation_losses.append(loss_adapt.item())
            meta_model.eval()
        else:
            adaptation_losses.append(np.nan)
    predictions_meta = np.array(predictions_meta)
    weights_history = np.array(weights_history)
    y_test_array = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)
    mae_meta = mean_absolute_error(y_test_array, predictions_meta)
    mse_meta = np.mean((y_test_array - predictions_meta)**2)
    rmse_meta = np.sqrt(mse_meta)
    mae_base = {}
    for name in model_names:
        preds = np.array(predictions_base[name])
        mae_base[name] = mean_absolute_error(y_test_array, preds)
    results = {'predictions_meta': predictions_meta, 'predictions_base': predictions_base, 'weights_history': weights_history, 'adaptation_losses': adaptation_losses, 'y_test': y_test_array, 'mae_meta': mae_meta, 'mse_meta': mse_meta, 'rmse_meta': rmse_meta, 'mae_base': mae_base, 'model_names': model_names}
    return results

def retrain_base_model_incremental(model_type, params, X_history, y_history, X_new, y_new):
    X_combined = pd.concat([X_history, X_new], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_history, y_new], axis=0).reset_index(drop=True) if isinstance(y_history, pd.Series) else np.concatenate([y_history, y_new])
    if model_type == 'lgb':
        model = lgb.LGBMRegressor(**params)
        model.fit(X_combined, y_combined)
    elif model_type == 'catboost':
        model = CatBoostRegressor(**params)
        model.fit(X_combined, y_combined, verbose=False)
    return model

def generate_base_predictions_for_test(X_train, y_train, X_test, base_models_params, device, seq_len=96, pred_len=30, model_size='small', freq='D'):
    from model.bases_models.ligthGBM_model import lgb
    from catboost import CatBoostRegressor
    n_test = len(X_test)
    model_names = ['lgb', 'catboost', 'timexer', 'moirai']
    predictions = {name: np.zeros(n_test) for name in model_names}
    lgb_params = base_models_params['lgb'].copy()
    lgb_params['verbose'] = -1
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    predictions['lgb'] = lgb_model.predict(X_test)
    cb_params = base_models_params['catboost'].copy()
    cb_params['verbose'] = 0
    cb_params['allow_writing_files'] = False
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(X_train, y_train, verbose=False)
    predictions['catboost'] = cb_model.predict(X_test)
    predictions['timexer'] = np.full(n_test, np.mean(y_train))
    predictions['moirai'] = np.full(n_test, np.mean(y_train))
    return predictions

def adaptive_forecast_v2(meta_model, base_models_params, X_train, y_train, X_test, y_test, oof_df_train, window_size, device, adaptation_lr=1e-4, adaptation_steps=1, retrain_interval=30):
    from model.bases_models.ligthGBM_model import lgb
    from catboost import CatBoostRegressor
    meta_model_adapt = LSTMMetaLearner(num_models=4, hidden_size=meta_model.hidden_size, num_layers=meta_model.num_layers).to(device)
    meta_model_adapt.load_state_dict(meta_model.state_dict())
    meta_model_adapt.eval()
    n_test = len(y_test)
    model_names = ['lgb', 'catboost', 'timexer', 'moirai']
    num_models = len(model_names)
    oof_history = oof_df_train.values.copy()
    y_history_list = y_train.values.tolist() if isinstance(y_train, pd.Series) else list(y_train)
    X_history = X_train.copy()
    lgb_params = base_models_params['lgb'].copy()
    lgb_params['verbose'] = -1
    cb_params = base_models_params['catboost'].copy()
    cb_params['verbose'] = 0
    cb_params['allow_writing_files'] = False
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(X_train, y_train, verbose=False)
    predictions_meta = []
    predictions_base = {name: [] for name in model_names}
    weights_history = []
    adaptation_losses = []
    errors_meta = []
    optimizer_adapt = torch.optim.Adam(meta_model_adapt.parameters(), lr=adaptation_lr)
    criterion_adapt = nn.MSELoss()
    for t in range(n_test):
        X_current = X_test.iloc[[t]] if hasattr(X_test, 'iloc') else X_test[t:t+1]
        base_preds_t = np.zeros(num_models)
        base_preds_t[0] = lgb_model.predict(X_current)[0]
        base_preds_t[1] = cb_model.predict(X_current)[0]
        base_preds_t[2] = oof_history[-1, 2] if len(oof_history) > 0 else np.mean(y_history_list)
        base_preds_t[3] = oof_history[-1, 3] if len(oof_history) > 0 else np.mean(y_history_list)
        for i, name in enumerate(model_names):
            predictions_base[name].append(base_preds_t[i])
        oof_history = np.vstack([oof_history, base_preds_t])
        if len(oof_history) >= window_size:
            window_data = oof_history[-window_size:]
            X_meta = torch.from_numpy(window_data.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                y_pred_meta, alpha_t = meta_model_adapt(X_meta, return_weights=True)
            pred_meta = y_pred_meta.cpu().item()
            weights_t = alpha_t.cpu().numpy().flatten()
        else:
            pred_meta = np.mean(base_preds_t)
            weights_t = np.ones(num_models) / num_models
        predictions_meta.append(pred_meta)
        weights_history.append(weights_t)
        y_true_t = y_test.iloc[t] if isinstance(y_test, pd.Series) else y_test[t]
        error_t = abs(pred_meta - y_true_t)
        errors_meta.append(error_t)
        y_history_list.append(y_true_t)
        X_history = pd.concat([X_history, X_current], axis=0).reset_index(drop=True)
        if len(oof_history) >= window_size:
            meta_model_adapt.train()
            for _ in range(adaptation_steps):
                window_data_adapt = oof_history[-window_size:]
                X_adapt = torch.from_numpy(window_data_adapt.astype(np.float32)).unsqueeze(0).to(device)
                y_adapt = torch.tensor([y_true_t], dtype=torch.float32).to(device)
                optimizer_adapt.zero_grad()
                y_pred_adapt = meta_model_adapt(X_adapt)
                loss_adapt = criterion_adapt(y_pred_adapt, y_adapt)
                loss_adapt.backward()
                torch.nn.utils.clip_grad_norm_(meta_model_adapt.parameters(), 1.0)
                optimizer_adapt.step()
            adaptation_losses.append(loss_adapt.item())
            meta_model_adapt.eval()
        else:
            adaptation_losses.append(np.nan)
        if (t + 1) % retrain_interval == 0 and t > 0:
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_history, np.array(y_history_list))
            cb_model = CatBoostRegressor(**cb_params)
            cb_model.fit(X_history, np.array(y_history_list), verbose=False)
    predictions_meta = np.array(predictions_meta)
    weights_history = np.array(weights_history)
    y_test_array = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)
    mae_meta = mean_absolute_error(y_test_array, predictions_meta)
    mse_meta = np.mean((y_test_array - predictions_meta)**2)
    rmse_meta = np.sqrt(mse_meta)
    mae_base = {}
    for name in model_names:
        preds = np.array(predictions_base[name])
        mae_base[name] = mean_absolute_error(y_test_array, preds)
    results = {'predictions_meta': predictions_meta, 'predictions_base': predictions_base, 'weights_history': weights_history, 'adaptation_losses': adaptation_losses, 'errors_meta': errors_meta, 'y_test': y_test_array, 'mae_meta': mae_meta, 'mse_meta': mse_meta, 'rmse_meta': rmse_meta, 'mae_base': mae_base, 'model_names': model_names, 'meta_model_adapted': meta_model_adapt}
    return results