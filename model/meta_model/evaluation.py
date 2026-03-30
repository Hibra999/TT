import numpy as np
import logging
from itertools import combinations
from model.meta_model.stats_utils import dm_test, check_dm_assumptions

def run_pairwise_dm_tests(preds_p, MDL, pr_r, target_metas):
    """
    Realiza pruebas de Diebold-Mariano entre todos los pares de meta-modelos.
    Retorna los resultados organizados en bloques para el reporte.
    """
    logging.info(f"[DM] Iniciando pruebas Diebold-Mariano sobre {len(target_metas)} meta modelos...")
    
    # Validar que MT existe en preds_p (modelo base de comparación)
    if 'MT' not in preds_p:
        logging.warning('[DM] Modelo "Meta LSTM (Ensamble Actual)" (key=MT) no encontrado en preds_p')
        # Continuar de todos modos si hay otros modelos
    
    # Generar todos los pares únicos
    dm_all_pairs = list(combinations(target_metas, 2))
    dm_results = []
    
    for (ki, kj) in dm_all_pairs:
        if ki in preds_p and kj in preds_p:
            pi, pj = preds_p[ki], preds_p[kj]
            # Alinear por índices no nulos comunes
            mask = (~np.isnan(pi)) & (~np.isnan(pj)) & (~np.isnan(pr_r))
            if mask.sum() > 30:  # Mínimo de muestras para validez estadística
                # Diferencial de pérdida: (e_i^2 - e_j^2)
                d = (pr_r[mask] - pi[mask])**2 - (pr_r[mask] - pj[mask])**2
                check_dm_assumptions(d, f"{ki} vs {kj}")
                stat, pval = dm_test(d)
                
                if pval < 0.05:
                    better = MDL[kj][1] if stat > 0 else MDL[ki][1]
                else:
                    better = '—'   # Sin diferencia significativa
                
                dm_results.append({
                    'key_a': ki, 'key_b': kj,
                    'model_a': MDL[ki][1], 'model_b': MDL[kj][1],
                    'stat': stat, 'p_value': pval, 'sig': pval < 0.05,
                    'better': better
                })
    
    # Organizar en bloques para el reporte
    actual_keys = {'MT', 'NC', 'SA', 'WA', 'RD', 'LS', 'EN', 
                   'LGB_META', 'RF_META', 'MLP_META', 'GRU_META', 'TRANS_META'}
    
    bloque1 = []  # Ensamble Actual vs Ensamble Actual
    bloque2 = []  # Ensamble Actual vs AB
    bloque3 = []  # Ensamble Actual vs SM
    bloque4 = []  # Ensamble Actual vs XGB_META_EXT (Parker)
    bloque5 = []  # Comparaciones cruzadas restantes
    
    for r in dm_results:
        ka, kb = r['key_a'], r['key_b']
        if ka in actual_keys and kb in actual_keys:
            bloque1.append(r)
        elif (ka in actual_keys and kb == 'AB') or (ka == 'AB' and kb in actual_keys):
            bloque2.append(r)
        elif (ka in actual_keys and kb == 'SM') or (ka == 'SM' and kb in actual_keys):
            bloque3.append(r)
        elif (ka in actual_keys and kb == 'XGB_META_EXT') or (ka == 'XGB_META_EXT' and kb in actual_keys):
            bloque4.append(r)
        else:
            bloque5.append(r)
            
    bloques = [
        ('Bloque 1: Ensamble Actual vs Ensamble Actual', bloque1),
        ('Bloque 2: Ensamble Actual vs Ours Sin TimeXer', bloque2),
        ('Bloque 3: Ensamble Actual vs Yu et al.', bloque3),
        ('Bloque 4: Ensamble Actual vs Parker et al.', bloque4),
        ('Bloque 5: Comparaciones cruzadas restantes', bloque5),
    ]
    
    return dm_results, bloques
