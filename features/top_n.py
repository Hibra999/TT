import numpy as np
import pandas as pd
from scipy.stats import rankdata
from numba import njit
from joblib import Parallel, delayed
import operator
@njit(fastmath=True, cache=True)
def _compute_mic_kernel(x, y, n, B):
    max_mic = 0.0
    log_n = np.log(n)
    inv_n = 1.0 / n
    limit_x = int(B) + 1
    for nx in range(2, limit_x):
        x_bins = np.empty(n, dtype=np.int32)
        for i in range(n):
            val = int(x[i] * nx)
            if val >= nx: val = nx - 1
            x_bins[i] = val
        limit_y = int(B / nx) + 1
        if limit_y < 2: limit_y = 2
        for ny in range(2, limit_y + 1):
            if nx * ny > B: 
                continue
            H = np.zeros((nx, ny), dtype=np.int32)
            rs = np.zeros(nx, dtype=np.int32)
            cs = np.zeros(ny, dtype=np.int32)
            for i in range(n):
                c = int(y[i] * ny)
                if c >= ny: c = ny - 1
                r = x_bins[i]
                H[r, c] += 1
                rs[r] += 1
                cs[c] += 1
            sum_nij_log = 0.0
            for r in range(nx):
                for c in range(ny):
                    val = H[r, c]
                    if val > 0:
                        sum_nij_log += val * np.log(float(val))
            sum_ni_log = 0.0
            for r in range(nx):
                val = rs[r]
                if val > 0:
                    sum_ni_log += val * np.log(float(val))
            sum_nj_log = 0.0
            for c in range(ny):
                val = cs[c]
                if val > 0:
                    sum_nj_log += val * np.log(float(val))
            mi = (sum_nij_log - sum_ni_log - sum_nj_log) * inv_n + log_n
            if mi < 0: mi = 0.0
            denom = np.log(min(nx, ny))
            if denom > 0:
                score = mi / denom
            else:
                score = 0.0
            if score > max_mic:
                max_mic = score
    return max_mic

def worker_mic_clean(col_name, x_raw, y_raw, min_samples=30):
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_clean = x_raw[mask]
    y_clean = y_raw[mask]
    n = len(x_clean)
    if n < min_samples:
        return col_name, 0.0
    B = n ** 0.6
    x_norm = (rankdata(x_clean, method='average') - 1) / (n - 1)
    y_norm = (rankdata(y_clean, method='average') - 1) / (n - 1)
    score = _compute_mic_kernel(np.ascontiguousarray(x_norm, dtype=np.float64), np.ascontiguousarray(y_norm, dtype=np.float64), n, B)
    return col_name, score

def top_k(X, y, k):
    common_index = X.index.intersection(y.index)
    X_aligned = X.loc[common_index].copy()
    y_aligned = y.loc[common_index]
    if isinstance(y, pd.Series) and y.name in X_aligned.columns:
        X_aligned = X_aligned.drop(columns=[y.name])
    X_cols = X_aligned.columns
    X_vals = X_aligned.values
    y_vals = y_aligned.values
    scores = Parallel(n_jobs=-1)(delayed(worker_mic_clean)(X_cols[i], X_vals[:, i], y_vals) for i in range(len(X_cols)))
    valid_scores = [s for s in scores if s[1] is not None]
    sorted_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)
    top = {}
    for i in range(min(k, len(sorted_scores))):
        feat_name, feat_score = sorted_scores[i]
        top[feat_name] = feat_score
    top = dict(sorted(top.items(), key=lambda item: item[1], reverse=True))
    top_features = [x[0] for x in sorted_scores[:k]]
    return top_features, top