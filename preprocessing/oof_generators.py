import numpy as np
import pandas as pd

def collect_oof_predictions(oof_storage):
    """
    Recolecta TODAS las predicciones OOF de todos los folds.
    Retorna: (predicciones, indices, targets)
    """
    if oof_storage is None or 'preds' not in oof_storage:
        return np.array([]), np.array([]), np.array([])
    all_preds = []
    all_indices = []
    all_targets = []
    for fold_idx, (preds, indices) in enumerate(zip(oof_storage['preds'], oof_storage['indices'])):
        preds_flat = np.array(preds).flatten()
        indices_flat = np.array(indices).flatten()
        min_len = min(len(preds_flat), len(indices_flat))
        all_preds.extend(preds_flat[:min_len].tolist())
        all_indices.extend(indices_flat[:min_len].tolist())
        # Si hay targets guardados
        if 'targets' in oof_storage and fold_idx < len(oof_storage['targets']):
            targets_flat = np.array(oof_storage['targets'][fold_idx]).flatten()
            all_targets.extend(targets_flat[:min_len].tolist())
    return np.array(all_preds), np.array(all_indices), np.array(all_targets)

def build_oof_dataframe(oof_lgb, oof_cb, oof_tx, oof_moirai, y):
    """
    Construye DataFrame OOF alineando predicciones por indice.
    Solo incluye filas donde TODOS los modelos tienen prediccion.
    """
    # Recolectar predicciones de cada modelo
    preds_lgb, idx_lgb, _ = collect_oof_predictions(oof_lgb)
    preds_cb, idx_cb, _ = collect_oof_predictions(oof_cb)
    preds_tx, idx_tx, _ = collect_oof_predictions(oof_tx)
    preds_moirai, idx_moirai, _ = collect_oof_predictions(oof_moirai)
    # Crear DataFrames individuales
    df_lgb = pd.DataFrame({'idx': idx_lgb, 'lgb': preds_lgb})
    df_cb = pd.DataFrame({'idx': idx_cb, 'catboost': preds_cb})
    df_tx = pd.DataFrame({'idx': idx_tx, 'timexer': preds_tx})
    df_moirai = pd.DataFrame({'idx': idx_moirai, 'moirai': preds_moirai})
    # Agregar targets desde y
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    # Merge por indice (inner join = solo donde todos tienen prediccion)
    merged = df_lgb.merge(df_cb, on='idx', how='inner')
    merged = merged.merge(df_tx, on='idx', how='inner')
    merged = merged.merge(df_moirai, on='idx', how='inner')
    # Agregar target
    merged['target'] = merged['idx'].apply(lambda i: y_array[int(i)] if int(i) < len(y_array) else np.nan)
    # Eliminar filas con NaN
    merged = merged.dropna()
    merged = merged.sort_values('idx').reset_index(drop=True)
    return merged