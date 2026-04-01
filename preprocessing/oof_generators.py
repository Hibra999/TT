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
    
    print(f"[DEBUG OOF] Base sizes: LGB={len(idx_lgb)}, CB={len(idx_cb)}, TX={len(idx_tx)}, MO={len(idx_moirai)}")
    
    # Crear DataFrames individuales
    df_lgb = pd.DataFrame({'idx': idx_lgb, 'lgb': preds_lgb})
    df_cb = pd.DataFrame({'idx': idx_cb, 'catboost': preds_cb})
    df_tx = pd.DataFrame({'idx': idx_tx, 'timexer': preds_tx})
    df_moirai = pd.DataFrame({'idx': idx_moirai, 'moirai': preds_moirai})
    # Agregar targets desde y
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    # Merge por indice (inner join = solo donde todos tienen prediccion)
    merged = df_lgb.merge(df_cb, on='idx', how='inner')
    print(f"[DEBUG OOF] After CB merge: {len(merged)}")
    merged = merged.merge(df_tx, on='idx', how='inner')
    print(f"[DEBUG OOF] After TX merge: {len(merged)}")
    merged = merged.merge(df_moirai, on='idx', how='inner')
    print(f"[DEBUG OOF] After MO merge: {len(merged)}")
    # Agregar target
    merged['target'] = merged['idx'].apply(lambda i: y_array[int(i)] if int(i) < len(y_array) else np.nan)
    # Eliminar filas con NaN
    merged = merged.dropna()
    print(f"[DEBUG OOF] Final merged: {len(merged)}")
    if len(merged) > 0:
        print(f"[DEBUG OOF] Missing TX indices in merged? Total unique TX: {len(np.unique(idx_tx))} vs merged {len(merged)}")
        
    merged = merged.sort_values('idx').reset_index(drop=True)
    return merged

def build_oof_dataframe_refit(
    bp_lgb, bp_cb, bp_tx, bp_moirai,
    Xt, yt, sp, device,
    lgb_predict_fn,
    cb_predict_fn,
    tx_predict_fn,
    moirai_predict_fn,
    seq_len_tx=96,
    pred_len_tx=1,
    model_size_mo='small',
    freq_mo='D'
):
    """
    Genera OOF con cobertura completa usando cross-boundary context
    para TX y MO. Ejecuta una CV completa con best_params ya conocidos.
    
    A diferencia de build_oof_dataframe, aquí:
    - No se usan los OOF de Optuna (descartados)
    - TX y MO reciben los últimos seq_len pasos del fold de train
      como contexto, cubriendo el 100% del fold de validación
    - El resultado es consistente con la inferencia en test
    """
    all_lgb     = []
    all_cb      = []
    all_tx      = []
    all_mo      = []
    all_idx     = []
    all_target  = []

    splits = list(sp.split(yt))  # materializar el generador de sktime
    
    for fold_i, (t_idx, v_idx) in enumerate(splits):
        print(f'  [Refit] Fold {fold_i+1}/{len(splits)} '
              f'| train={len(t_idx)} val={len(v_idx)}')

        # ── Datos del fold ──
        X_tr  = Xt.iloc[t_idx].reset_index(drop=True)
        y_tr  = yt.iloc[t_idx].reset_index(drop=True)
        X_val = Xt.iloc[v_idx].reset_index(drop=True)
        y_val = yt.iloc[v_idx].reset_index(drop=True)

        # ── LGB: sin restricción de contexto ──
        p_lgb, _ = lgb_predict_fn(X_tr, y_tr, X_val, bp_lgb)

        # ── CB: sin restricción de contexto ──
        p_cb, _ = cb_predict_fn(X_tr, y_tr, X_val, bp_cb)

        # ── TX: cross-boundary context ──
        # Replicar exactamente lo que se hace en test:
        # los últimos seq_len pasos de train se anteponen a val
        ctx_tx = seq_len_tx
        X_tail_tx = Xt.iloc[t_idx[-ctx_tx:]].reset_index(drop=True)
        y_tail_tx = yt.iloc[t_idx[-ctx_tx:]].reset_index(drop=True)

        X_val_ctx = pd.concat(
            [X_tail_tx, X_val], axis=0
        ).reset_index(drop=True)
        y_val_ctx = pd.concat(
            [y_tail_tx, y_val], axis=0
        ).reset_index(drop=True)

        p_tx_full, _, _ = tx_predict_fn(
            X_tr, y_tr,
            X_val_ctx, y_val_ctx,
            bp_tx, device,
            seq_len=seq_len_tx,
            pred_len=pred_len_tx,
            features='MS'
        )
        # Recortar: descartar los ctx_tx pasos de warm-up
        # solo conservar predicciones sobre val real
        p_tx = p_tx_full[-len(y_val):]

        # ── MO: cross-boundary context ──
        # Moirai solo usa la serie target como contexto
        # Anteponer los últimos ctx_mo pasos de y_tr a y_val
        ctx_mo = bp_moirai.get('context_length', 96)
        y_tail_mo = yt.iloc[t_idx[-ctx_mo:]].reset_index(drop=True)

        y_val_ctx_mo = pd.concat(
            [y_tail_mo, y_val], axis=0
        ).reset_index(drop=True)
        # y_train para Moirai: todo el fold de train
        # (internamente usa solo los últimos ctx_mo pasos)
        p_mo_full, _ = moirai_predict_fn(
            y_tr,
            y_val_ctx_mo,
            bp_moirai,
            model_size=model_size_mo,
            freq=freq_mo
        )
        # Recortar igual que TX
        p_mo = p_mo_full[-len(y_val):]

        # ── Verificación de cobertura por fold ──
        for name, arr in [('LGB', p_lgb), ('CB', p_cb),
                           ('TX', p_tx),  ('MO', p_mo)]:
            n_nan   = np.isnan(arr).sum()
            n_valid = len(arr) - n_nan
            if n_nan > 0:
                print(f'    [WARN] {name} fold {fold_i+1}: '
                      f'{n_valid}/{len(y_val)} válidas '
                      f'({n_nan} NaNs)')

        # ── Acumular usando índices ABSOLUTOS de v_idx ──
        all_lgb.extend(p_lgb.tolist())
        all_cb.extend(p_cb.tolist())
        all_tx.extend(p_tx.tolist())
        all_mo.extend(p_mo.tolist())
        all_idx.extend(v_idx.tolist())           # índices absolutos en Xt
        all_target.extend(y_val.tolist())

    # ── Construir DataFrame ──
    df_oof = pd.DataFrame({
        'idx':      all_idx,
        'lgb':      all_lgb,
        'catboost': all_cb,
        'timexer':  all_tx,
        'moirai':   all_mo,
        'target':   all_target,
    })

    nan_report = df_oof.isna().sum().to_dict()
    print(f'  [Refit OOF] NaN por columna: {nan_report}')
    print(f'  [Refit OOF] Shape antes de dropna: {df_oof.shape}')

    df_oof = df_oof.dropna().sort_values('idx').reset_index(drop=True)
    print(f'  [Refit OOF] Shape final: {df_oof.shape}')

    return df_oof