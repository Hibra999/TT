import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import t as t_dist

def check_dm_assumptions(d: np.ndarray, name: str) -> None:
    """Verifica estacionariedad y autocorrelación del diferencial de pérdida."""
    # Guard: si d es constante, ADF y Ljung-Box no aplican
    if np.std(d) < 1e-15:
        logging.warning(f"[DM:{name}] d_t es constante (modelos idénticos). Se omiten tests de supuestos.")
        return
    # 1. ADF (Estacionariedad)
    try:
        adf_res = adfuller(d, autolag='AIC')
        if adf_res[1] > 0.05:
            logging.warning(f"[DM:{name}] d_t no estacionaria (ADF p={adf_res[1]:.3f})")
    except Exception as e:
        logging.warning(f"[DM:{name}] ADF failed: {e}")

    # 2. Ljung-Box (Autocorrelación)
    try:
        h = int(np.floor(len(d)**(1/3)))
        lb_res = acorr_ljungbox(d, lags=[h], return_df=True)
        if lb_res['lb_pvalue'].iloc[0] < 0.05:
            logging.info(f"[DM:{name}] Autocorrelación detectada (Ljung-Box p={lb_res['lb_pvalue'].iloc[0]:.3f}). Uso de HAC confirmado.")
    except Exception as e:
        logging.warning(f"[DM:{name}] Ljung-Box failed: {e}")

def dm_test(d: np.ndarray) -> tuple[float, float]:
    """Prueba DM con corrección HAC (Heteroskedasticity and Autocorrelation Consistent)."""
    # Guard: si d es constante, no hay diferencia → DM=0, p=1
    if np.std(d) < 1e-15:
        return 0.0, 1.0
    T = len(d)
    h = int(np.floor(T ** (1/3)))
    d_bar = d.mean()
    try:
        # OLS sobre constante para obtener varianza HAC
        model = OLS(d, np.ones(T)).fit()
        hac_var = cov_hac(model, nlags=h).item()
        if hac_var < 1e-15:
            return 0.0, 1.0
        dm_stat = d_bar / np.sqrt(hac_var / T)
        p_value = 2 * (1 - t_dist.cdf(abs(dm_stat), df=T-1))
        return dm_stat, p_value
    except Exception as e:
        logging.warning(f"[DM] Error en test: {e}")
        return 0.0, 1.0
