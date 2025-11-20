import numpy as np
from scipy.stats import rankdata
from numba import njit
from joblib import Parallel, delayed

@njit(fastmath=True, cache=True)
def _mic_calc(x, y, B):
    n = len(x)
    limit = int(B) + 1
    max_mic = 0.0
    log_n = np.log(n)
    inv_n = 1.0 / n

    for nx in range(2, limit):
        x_bins = (x * nx).astype(np.int32)
        for i in range(n):
            if x_bins[i] >= nx: x_bins[i] = nx - 1

        for ny in range(2, limit):
            if nx * ny > B: continue

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

            sum_nij = 0.0
            for r in range(nx):
                for c in range(ny):
                    val = H[r, c]
                    if val > 0: sum_nij += val * np.log(val)
            
            sum_ni = 0.0
            for r in range(nx):
                if rs[r] > 0: sum_ni += rs[r] * np.log(rs[r])
            
            sum_nj = 0.0
            for c in range(ny):
                if cs[c] > 0: sum_nj += cs[c] * np.log(cs[c])

            mi = (sum_nij - sum_ni - sum_nj) * inv_n + log_n
            score = mi / np.log(min(nx, ny))
            if score > max_mic: max_mic = score

    return max_mic

def top_k(X, y, k):
    n = len(X)
    B = n ** 0.6
    y_norm = (rankdata(y, method='average') - 1) / (n - 1)
    def worker(col, data):
        x_norm = (rankdata(data, method='average') - 1) / (n - 1)
        return col, _mic_calc(x_norm, y_norm, B)
    scores = Parallel(n_jobs=-1)(
        delayed(worker)(c, X[c].values) for c in X.columns
    )
    return [x[0] for x in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]