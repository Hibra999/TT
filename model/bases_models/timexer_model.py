import math
from math import sqrt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
        else:
            return None


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = (B, 1, L, L)
        mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)
        self._mask = mask

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len, pred_len):
        assert len(X) == len(y)
        self.seq_len = seq_len
        self.pred_len = pred_len
        X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1)
        self.data = np.concatenate([X, y], axis=1)
        self.n_samples = len(self.data) - self.seq_len - self.pred_len + 1
        if self.n_samples <= 0:
            raise ValueError(f"Datos insuficientes: {len(self.data)} para seq_len={seq_len}, pred_len={pred_len}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len, :]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, -1]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_timexer_config(trial, enc_in, seq_len, pred_len, features='MS'):
    patch_len = trial.suggest_categorical("patch_len", [4, 8, 12, 16, 24])
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    e_layers = trial.suggest_int("e_layers", 1, 4)
    d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    factor = trial.suggest_int("factor", 1, 5)
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])
    configs = SimpleNamespace(
        task_name='long_term_forecast',
        features=features,
        seq_len=seq_len,
        pred_len=pred_len,
        use_norm=False,
        patch_len=patch_len,
        d_model=d_model,
        dropout=dropout,
        embed='fixed',
        freq='h',
        factor=factor,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        activation=activation,
        enc_in=enc_in,
    )
    return configs


def build_model_from_trial(trial, enc_in, seq_len, pred_len, device, features='MS', pretrained_path=None, freeze_backbone=False):
    configs = build_timexer_config(trial, enc_in, seq_len, pred_len, features)
    model = Model(configs).to(device)
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)
        if freeze_backbone:
            for name, p in model.named_parameters():
                if "head" not in name:
                    p.requires_grad = False
    return model


def objective_timexer_global(trial, X, y, splitter, device=None, seq_len=96, pred_len=30, features='MS', pretrained_path=None, freeze_backbone=False, oof_storage=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    enc_in = X.shape[1] + 1
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    max_epochs = trial.suggest_int("max_epochs", 10, 50)
    patience_val = trial.suggest_int("patience", 5, 15)
    
    X_values = X.values if hasattr(X, 'values') else np.array(X)
    y_values = y.values if hasattr(y, 'values') else np.array(y)
    
    X_values = np.nan_to_num(X_values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_values = np.nan_to_num(y_values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1)
    full_data = np.concatenate([X_values, y_values], axis=1)
    
    fold_scores, fold_preds, fold_indices = [], [], []
    
    for fold_num, (t_idx, v_idx) in enumerate(splitter.split(y)):
        train_start = int(t_idx[0])
        train_end = int(t_idx[-1]) + 1
        
        if train_end - train_start < seq_len + pred_len + 10:
            continue
        
        train_data = full_data[train_start:train_end]
        
        class TrainDataset(Dataset):
            def __init__(self, data, seq_len, pred_len):
                self.data = data
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.n_samples = len(data) - seq_len - pred_len + 1
            
            def __len__(self):
                return max(0, self.n_samples)
            
            def __getitem__(self, idx):
                x = self.data[idx:idx + self.seq_len]
                y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, -1]
                return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
        
        train_ds = TrainDataset(train_data, seq_len, pred_len)
        if len(train_ds) <= 0:
            continue
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        if len(train_loader) == 0:
            continue
        
        model = build_model_from_trial(
            trial, enc_in=enc_in, seq_len=seq_len, pred_len=pred_len,
            device=device, features=features, pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone
        )
        
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        
        best_train_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch, None, None, None).squeeze(-1)
                loss = criterion(out, y_batch)
                if torch.isnan(loss):
                    return float("inf")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
            
            mean_train_loss = np.mean(epoch_losses)
            if mean_train_loss < best_train_loss:
                best_train_loss = mean_train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_val:
                    break
        
        model.eval()
        val_preds = []
        val_indices = []
        
        with torch.no_grad():
            for target_idx in v_idx:
                target_idx = int(target_idx)
                window_end = target_idx
                window_start = window_end - seq_len
                
                if window_start < 0:
                    continue
                
                x_window = full_data[window_start:window_end]
                
                if len(x_window) != seq_len:
                    continue
                
                x_tensor = torch.from_numpy(x_window.astype(np.float32)).unsqueeze(0).to(device)
                out = model(x_tensor, None, None, None).squeeze(-1)
                pred_value = out[0, 0].cpu().numpy()
                
                val_preds.append(float(pred_value))
                val_indices.append(target_idx)
        
        if len(val_preds) == 0:
            continue
        
        val_preds = np.array(val_preds)
        val_indices = np.array(val_indices)
        val_targets = y_values[val_indices].flatten()
        
        fold_mae = np.mean(np.abs(val_preds - val_targets))
        
        if np.isnan(fold_mae):
            continue
        
        fold_scores.append(fold_mae)
        fold_preds.append(val_preds)
        fold_indices.append(val_indices)
        
        trial.report(fold_mae, fold_num)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    if not fold_scores:
        return float("inf")
    
    mean_score = float(np.mean(fold_scores))
    
    if oof_storage is not None:
        if 'best_score' not in oof_storage or mean_score < oof_storage['best_score']:
            oof_storage['best_score'] = mean_score
            oof_storage['preds'] = fold_preds
            oof_storage['indices'] = fold_indices
    
    return mean_score