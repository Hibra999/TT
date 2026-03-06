import pandas as pd
import numpy as np
import os
import warnings
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

from features.macroeconomics import macroeconomicos
from model.bases_models.ligthGBM_model import objective_global
from model.bases_models.catboost_model import objective_catboost_global
from model.bases_models.timexer_model import objective_timexer_global
from model.bases_models.moraiMOE_model import objective_moirai_moe_global, preload_moirai_module
from model.meta_model.lstm_model import optimize_lstm_meta
from preprocessing.oof_generators import collect_oof_predictions, build_oof_dataframe
from preprocessing.walk_forward import wfrw
from features.tecnical_indicators import TA
from features.top_n import top_k

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def calcular_metricas(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def main():
    print("="*60)
    print("Iniciando Experimento de Sensibilidad Multi-Ventana CLI")
    print("="*60)
    
    # Configuracion Fija
    token = 'AAPL' # Por defecto, o modificar a conveniencia
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_BASE_DIR, "data", "tokens", f"{token}_2020-2025.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: Archivo de datos no encontrado: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    print(f"[1] Datos cargados para {token}: {len(df)} registros.")
    
    # Procesamiento (Similar al Tab 1, 2, 3 de main.py)
    log_close = np.log(df["Close"] / df["Close"].shift(-1)).dropna()
    
    df_ta = TA(df).reset_index(drop=True)
    df_ma = macroeconomicos(df["Date_final"]).reset_index(drop=True)
    
    df_final = pd.concat([df_ta, df_ma], axis=1).iloc[1:]
    log_close = log_close.reset_index(drop=True)
    
    min_len = min(len(df_final), len(log_close))
    df_final = df_final.iloc[:min_len].reset_index(drop=True)
    log_close = log_close.iloc[:min_len].reset_index(drop=True)
    
    # Limpiando columnas constantes
    cols_to_drop = [col for col in df_final.columns if df_final[col].max() - df_final[col].min() < 1e-8]
    df_final = df_final.drop(columns=cols_to_drop).replace([np.inf, -np.inf], 0.0)
    log_close = log_close.replace([np.inf, -np.inf], 0.0)
    
    if log_close.max() - log_close.min() < 1e-8:
        print("Error: log_close tiene valores constantes.")
        return
        
    n_total = len(df_final)
    train_size = int(n_total * 0.9)
    X_train_raw = df_final.iloc[:train_size].copy()
    X_test_raw = df_final.iloc[train_size:].copy()
    y_train_raw = log_close.iloc[:train_size].copy()
    y_test_raw = log_close.iloc[train_size:].copy()
    
    scaler_features = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler_features.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler_features.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
    
    scaler_target = MinMaxScaler()
    y_train = pd.Series(scaler_target.fit_transform(y_train_raw.values.reshape(-1,1)).flatten(), index=y_train_raw.index, name='log_close')
    y_test = pd.Series(scaler_target.transform(y_test_raw.values.reshape(-1,1)).flatten(), index=y_test_raw.index, name='log_close')
    
    # Seleccion top 15 features con MIC (solo train)
    print("[2] Calculando MIC Top-15 Features...")
    features, _ = top_k(X_train_scaled, y_train, 15)
    
    X_train = X_train_scaled[features].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    print(f"[3] Tamaño final Train: {len(X_train)} muestras con {len(features)} features.")
    
    # ---------------- EXPERIMENTACION MULTI-VENTANA ----------------
    k_val = 5
    fh_val = 30
    window_ratios_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    n_trials_fast = 2 # Usar un nro bajito de trials para experimentacion o el deseado
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[4] Dispositivo Torch: {device}")
    
    all_metrics = []
    
    for idx, temp_wr in enumerate(window_ratios_to_test):
        print(f"\n--- Probando Window Ratio: {temp_wr} ({idx+1}/{len(window_ratios_to_test)}) ---")
        
        oof_lgb, oof_cb, oof_tx, oof_moirai = {}, {}, {}, {}
        splitter_temp = wfrw(y_train, k=k_val, fh_val=fh_val, window_ratio=temp_wr)
        
        # 1. Optimizacion Base Models
        print("  Entrenando LightGBM...")
        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(lambda trial: objective_global(trial, X_train, y_train, splitter_temp, oof_storage=oof_lgb), n_trials=n_trials_fast, n_jobs=-1)
        
        print("  Entrenando CatBoost...")
        study_cb = optuna.create_study(direction="minimize")
        study_cb.optimize(lambda trial: objective_catboost_global(trial, X_train, y_train, splitter_temp, oof_storage=oof_cb), n_trials=n_trials_fast, n_jobs=-1)
        
        print("  Entrenando TimeXer...")
        study_tx = optuna.create_study(direction="minimize")
        study_tx.optimize(lambda trial: objective_timexer_global(trial, X_train, y_train, splitter_temp, device=device, seq_len=96, pred_len=fh_val, features='MS', oof_storage=oof_tx), n_trials=1, n_jobs=1)
        
        print("  Entrenando Moirai-MoE...")
        preload_moirai_module(model_size='small')
        study_moirai = optuna.create_study(direction="minimize")
        study_moirai.optimize(lambda trial: objective_moirai_moe_global(trial, X_train, y_train, splitter_temp, device=device, pred_len=fh_val, model_size='small', freq='D', use_full_train=True, oof_storage=oof_moirai), n_trials=1, n_jobs=1)
        
        # 2. Recoleccion OOF Dataframe
        preds_lgb, idx_lgb = collect_oof_predictions(oof_lgb)
        preds_cb, idx_cb = collect_oof_predictions(oof_cb)
        preds_tx, idx_tx = collect_oof_predictions(oof_tx)
        preds_moirai, idx_moirai, _ = collect_oof_predictions(oof_moirai)
        
        oof_df_temp = build_oof_dataframe(oof_lgb, oof_cb, oof_tx, oof_moirai, y_train)
        
        if len(oof_df_temp) < 20:
            print(f"  [Advertencia] Window Ratio {temp_wr} generó OOF insuficiente ({len(oof_df_temp)}). Saltando.")
            continue
            
        print("  Entrenando Meta-LSTM...")
        meta_model, mae_meta, results, _, _ = optimize_lstm_meta(oof_df_temp, device, n_trials=1)
        
        # 3. Escalado inverso y Metrics
        valid_indices = results['valid_indices']
        targets_scaled = np.array(results['targets']).reshape(-1, 1)
        targets_log = scaler_target.inverse_transform(targets_scaled).flatten()
        
        close_prices = df['Close'].values
        
        def _get_col(df_temp, keywords):
            for col in df_temp.columns:
                if any(kw in col.lower() for kw in keywords):
                    return col
            return None
            
        c_lgb = _get_col(oof_df_temp, ['lgb', 'light'])
        c_cb = _get_col(oof_df_temp, ['cb', 'catboost'])
        c_tx = _get_col(oof_df_temp, ['tx', 'timex'])
        c_moi = _get_col(oof_df_temp, ['moirai', 'moe'])
        
        precio_real, precio_lgb, precio_cb, precio_tx, precio_moi, precio_meta = [], [], [], [], [], []
        predictions_log = scaler_target.inverse_transform(np.array(results['predictions']).reshape(-1,1)).flatten()
        
        lgb_log = scaler_target.inverse_transform(oof_df_temp.loc[valid_indices, c_lgb].values.reshape(-1,1)).flatten() if c_lgb else None
        cb_log = scaler_target.inverse_transform(oof_df_temp.loc[valid_indices, c_cb].values.reshape(-1,1)).flatten() if c_cb else None
        tx_log = scaler_target.inverse_transform(oof_df_temp.loc[valid_indices, c_tx].values.reshape(-1,1)).flatten() if c_tx else None
        moi_log = scaler_target.inverse_transform(oof_df_temp.loc[valid_indices, c_moi].values.reshape(-1,1)).flatten() if c_moi else None
        
        for i, vidx in enumerate(valid_indices):
            if vidx > 0 and vidx < len(close_prices):
                precio_anterior = close_prices[vidx-1]
                precio_real.append(precio_anterior * np.exp(targets_log[i]))
                precio_meta.append(precio_anterior * np.exp(predictions_log[i]))
                if lgb_log is not None: precio_lgb.append(precio_anterior * np.exp(lgb_log[i]))
                if cb_log is not None: precio_cb.append(precio_anterior * np.exp(cb_log[i]))
                if tx_log is not None: precio_tx.append(precio_anterior * np.exp(tx_log[i]))
                if moi_log is not None: precio_moi.append(precio_anterior * np.exp(moi_log[i]))

        pr_arr = np.array(precio_real)
        
        if len(pr_arr) > 0:
            mse_m, rmse_m, mae_m, r2_m = calcular_metricas(pr_arr, np.array(precio_meta))
            all_metrics.append({'Window_Ratio': temp_wr, 'Modelo': 'Meta LSTM', 'MSE': mse_m, 'RMSE': rmse_m, 'MAE': mae_m, 'R2': r2_m})
            
            if precio_lgb:
                mse_l, rmse_l, mae_l, r2_l = calcular_metricas(pr_arr, np.array(precio_lgb))
                all_metrics.append({'Window_Ratio': temp_wr, 'Modelo': 'LightGBM', 'MSE': mse_l, 'RMSE': rmse_l, 'MAE': mae_l, 'R2': r2_l})
            if precio_cb:
                mse_c, rmse_c, mae_c, r2_c = calcular_metricas(pr_arr, np.array(precio_cb))
                all_metrics.append({'Window_Ratio': temp_wr, 'Modelo': 'CatBoost', 'MSE': mse_c, 'RMSE': rmse_c, 'MAE': mae_c, 'R2': r2_c})
            if precio_tx:
                mse_t, rmse_t, mae_t, r2_t = calcular_metricas(pr_arr, np.array(precio_tx))
                all_metrics.append({'Window_Ratio': temp_wr, 'Modelo': 'TimeXer', 'MSE': mse_t, 'RMSE': rmse_t, 'MAE': mae_t, 'R2': r2_t})
            if precio_moi:
                mse_mo, rmse_mo, mae_mo, r2_mo = calcular_metricas(pr_arr, np.array(precio_moi))
                all_metrics.append({'Window_Ratio': temp_wr, 'Modelo': 'Moirai-MoE', 'MSE': mse_mo, 'RMSE': rmse_mo, 'MAE': mae_mo, 'R2': r2_mo})

    # Guardando resultados
    print("\n" + "="*60)
    print("Recoleccion Finalizada.")
    print("="*60)
    
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        
        # Formatear la salida HTML
        out_html = os.path.join(_BASE_DIR, "metrics_compare.html")
        
        # Identificar mejores métricas globalmente
        min_mse = df_metrics['MSE'].min()
        min_rmse = df_metrics['RMSE'].min()
        min_mae = df_metrics['MAE'].min()
        max_r2 = df_metrics['R2'].max()
        
        def highlight_metrics(row):
            colors = [''] * len(row)
            if row['MSE'] == min_mse:
                # indice 2 es MSE asumiendo columnas [Window_Ratio, Modelo, MSE, RMSE, MAE, R2]
                colors[list(row.index).index('MSE')] = 'background-color: lightgreen'
            if row['RMSE'] == min_rmse:
                colors[list(row.index).index('RMSE')] = 'background-color: lightgreen'
            if row['MAE'] == min_mae:
                colors[list(row.index).index('MAE')] = 'background-color: lightgreen'
            if row['R2'] == max_r2:
                colors[list(row.index).index('R2')] = 'background-color: lightgreen'
            return colors
            
        styled_df = df_metrics.style.apply(highlight_metrics, axis=1) \
                              .format({"MSE": "{:.6f}", "RMSE": "{:.6f}", "MAE": "{:.6f}", "R2": "{:.6f}"}) \
                              .set_table_styles([
                                  {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-family', 'Arial')]},
                                  {'selector': 'td', 'props': [('font-family', 'Arial'), ('border', '1px solid #ddd'), ('padding', '8px')]},
                                  {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '80%'), ('margin', '20px auto')]}
                              ]).set_caption("Sensibilidad Multi-Ventana de Modelos Base y Meta-LSTM")
                              
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Reporte Experimental de Ventanas</title>
        </head>
        <body style="font-family: Arial, sans-serif; background-color: #f6f1e9; padding: 20px;">
            <h2 style="text-align: center; color: #2c3e50;">Resultados del Análisis Multi-Ventana</h2>
            {styled_df.to_html()}
        </body>
        </html>
        """
        
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nReporte HTML guardado en: {out_html}")
        print("\nVista previa de resultados en consola (ordenados por MAE):")
        print(df_metrics.sort_values(by="MAE", ascending=True).head(15).to_string(index=False))
        
        mejor_fila = df_metrics.loc[df_metrics['MAE'].idxmin()]
        print(f"\n[!] La MEJOR configuracion general es:")
        print(f"    Modelo: {mejor_fila['Modelo']}")
        print(f"    Window Ratio: {mejor_fila['Window_Ratio']}")
        print(f"    MAE: {mejor_fila['MAE']:.4f}")
    else:
        print("Aviso: No se generaron metricas.")

if __name__ == "__main__":
    main()
