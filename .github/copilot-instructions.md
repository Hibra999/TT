# Copilot Instructions for TT Financial Forecasting Project

This file provides context for AI assistants working in this ML forecasting project.

## High-Level Architecture

The TT project is an **ensemble time-series forecasting system** that combines multiple deep learning and gradient boosting models for predicting financial asset price movements.

### Three-Layer Architecture

1. **Base Models** (`model/bases_models/`): Five independent forecasting models, each trained with Optuna hyperparameter optimization
   - LightGBM (`ligthGBM_model.py`) - Gradient boosting
   - CatBoost (`catboost_model.py`) - Categorical gradient boosting  
   - TimeXer (`timexer_model.py`) - Transformer-based time series
   - Moirai-MoE (`moraiMOE_model.py`) - Mixture of Experts deep learning
   - LSTM (`sota/base_lstm.py`) - Recurrent neural network baseline

2. **Feature Engineering** (`features/`): Technical and macroeconomic features
   - **TA-Lib indicators** (`tecnical_indicators.py`): SMA, EMA, RSI, BBANDS, MACD, etc.
   - **Macroeconomic features** (`macroeconomics.py`): Economic indicators from various sources
   - **Feature selection** (`top_n.py`): MIC (Maximal Information Coefficient) scoring using Numba JIT

3. **Meta-Learner** (`model/meta_model/lstm_model.py`): PyTorch LSTM that learns dynamic weights for base model predictions
   - Uses out-of-fold (OOF) predictions from base models
   - Applies minimum weight constraints to prevent model collapse
   - Temperature-based softmax for stable weight generation

### Data Flow and Transformations

```
Raw CSV Data (price + volume)
  ↓
TA-Lib Features + Macro Features (combined)
  ↓
MinMaxScaler (normalize to [0,1])
  ↓
MIC Feature Selection (top 15 features)
  ↓
Walk-Forward Split (5 folds, 30-step horizon)
  ↓
[For each fold]:
  → LightGBM fit → OOF predictions
  → CatBoost fit → OOF predictions
  → TimeXer fit → OOF predictions
  → Moirai-MoE fit → OOF predictions
  ↓
Collect all OOF predictions → Stack as meta-features
  ↓
LSTM Meta-Learner fit (learns feature importance weights)
  ↓
Test predictions with adaptive weighting
  ↓
Inverse transform (MinMaxScaler → log returns → prices)
```

## Key Conventions and Patterns

### Out-of-Fold (OOF) Storage Pattern

All base models use a **dictionary-based OOF storage** pattern:

```python
oof_storage = {
    'preds': [fold1_preds, fold2_preds, ...],    # List of arrays per fold
    'indices': [fold1_indices, fold2_indices, ...], # Row indices for alignment
    'best_score': 0.123,                           # Validation metric
    'params': {...}                                # Best Optuna hyperparameters
}
```

This enables:
- Proper data alignment across folds (critical for time series where fold order matters)
- Meta-learner training with OOF predictions only (prevents data leakage)
- Retraining the final model with best parameters discovered

**Usage in preprocessing**: `preprocessing/oof_generators.py` provides `collect_oof_predictions()` and `build_oof_dataframe()` to align OOF from all base models.

### Scale Transformations

Predictions flow through **three scales**:
1. **Log-return scale**: Raw target is `log(Close_t / Close_{t-1})`
2. **Normalized scale** (MinMaxScaler): Applied for model training (improves convergence)
3. **Price scale**: Reconstructed via `_recon()` for final metrics and reporting

The reconstruction formula: `Price_t = Close_{t-1} * exp(log_return_t)`

### Model Configuration Sections

Each main script (`main2.py`, `main_compare.py`) contains a clearly marked config section:

```python
# ===== CONFIG =====
TOKEN = 'ETH/USDT'                      # Asset to forecast (symbol or pair)
N_LGB, N_CB, N_TX, N_MO, N_MT = 3, 3, 3, 3, 3  # Optuna trials per model
START, END = '2020-01-01', '2025-12-31'  # Date range
# ==================
```

Modify this section to change which asset/period is analyzed.

### Numba JIT Optimization

Performance-critical functions use **Numba JIT compilation**:

```python
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _recon(yl, cp, n):  # Parallel reconstruction
        o = np.empty(n, dtype=np.float64)
        for i in prange(n):
            o[i] = cp[i] * np.exp(yl[i])
        return o
else:
    def _recon(yl, cp, n):  # Fallback for Numba unavailable
        return cp * np.exp(yl)
```

Always provide both compiled and fallback implementations.

### Model Training Functions Pattern

Each base model file exports two functions:

1. **`objective_global(trial, X, y, splitter, oof_storage=None)`**
   - Called by Optuna during hyperparameter search
   - Returns validation score (lower is better)
   - Stores best OOF predictions in `oof_storage` dict

2. **`train_final_and_predict_test(X_train, y_train, X_test, best_params)`**
   - Trains final model with best parameters on full training set
   - Returns `(test_predictions, trained_model)`

This pattern ensures clean separation between optimization and inference phases.

## Build and Run Commands

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# PyTorch with CUDA 12.8 (GPU acceleration, optional)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128

# TA-Lib requires system-level installation (varies by OS)
# macOS: brew install ta-lib
# Ubuntu: apt-get install libta-lib0 libta-lib0-dev
```

### Running Pipelines

```bash
# Main ensemble pipeline (LGB + CatBoost + TimeXer + Moirai + Meta-LSTM)
python main2.py

# Comparative analysis (Actual vs Ablation vs SOTA models)
python main_compare.py
```

**Output files**:
- `report.html` - Interactive Plotly report with predictions, metrics, and comparison plots
- `report_compare.html` - Ablation study results (generated by `main_compare.py`)

### Debugging and Development

```bash
# Test data alignment utilities (used when troubleshooting OOF issues)
python test_align.py

# Debug TimeXer model-specific issues
python debug_timexer.py

# Debug multi-model prediction alignment
python debug_preds.py
```

These scripts help identify shape mismatches or index alignment problems during development.

## Python Code Style Notes

- **Compact multi-line imports**: Multiple imports on a single line where logically related
  - Example: `from features.top_n import top_k; from features.tecnical_indicators import TA`
- **Type hints**: Used in PyTorch module definitions; less common in utility functions
- **Minimal comments**: Code is self-documenting; comments only for non-obvious logic
- **Graceful fallbacks**: Optional dependencies (Numba, GPU) should have CPU equivalents
- **Warnings suppressed**: `warnings.filterwarnings("ignore")` used at pipeline entry points (common for ML libraries)

## Data and Features

### Feature Engineering Workflow

1. **TA-Lib Technical Indicators** (from OHLCV data):
   - Trend: SMA, EMA, DEMA, TEMA, TRIMA, KAMA
   - Momentum: RSI, STOCH, MACD, ROC
   - Volatility: BBANDS, ATR, NATR, TRANGE
   - Volume: OBV, AD, ADOSC
   - Pattern: CDLMORNING, CDLEVENING, etc.

2. **Macroeconomic Features** (fetched from external APIs):
   - Interest rates, inflation, unemployment, VIX volatility
   - Stock indices (S&P 500), commodities (gold), crypto benchmarks

3. **MIC-Based Selection**:
   - Numba JIT computes Information Coefficient for each feature
   - Top 15 features selected based on mutual information with target
   - Reduces dimensionality and noise

### Target and Scaling

- **Target**: Log returns = `log(Close_t / Close_{t-1})`
- **Normalization**: MinMaxScaler applied to both features and target
- **Validation**: Walk-forward with 5 folds and 30-step forecast horizon

## Testing and Validation Strategy

### Walk-Forward Cross-Validation

The project uses **walk-forward validation** (`preprocessing/walk_forward.py`) instead of standard K-fold to respect time-series causality:

- 5 folds with sequential, non-overlapping training/validation windows
- 30-step forecast horizon (test set size)
- Each fold trains on all prior data, validates on next 30 steps
- Prevents data leakage by respecting temporal order

### Metrics Computed

All reported metrics are calculated on **both scales**:

1. **Price scale**: MSE, RMSE, MAE, R²
2. **Log-return scale**: MSE, RMSE, MAE, R²

Allows evaluation of both absolute price accuracy and relative return prediction.

## Common Workflows

### Adding a New Base Model

1. Create `model/bases_models/new_model.py` with:
   ```python
   def objective_global(trial, X, y, splitter, oof_storage=None):
       # Optuna objective - return validation score
       pass

   def train_final_and_predict_test(X_train, y_train, X_test, best_params):
       # Return (predictions, trained_model)
       pass
   ```

2. Import and call in `main2.py`:
   ```python
   from model.bases_models.new_model import objective_new, train_final_and_predict_test as new_predict_test
   study_new = optuna.create_study(direction='minimize')
   study_new.optimize(lambda trial: objective_new(trial, X, y, splitter, oof_new), n_trials=N_NEW)
   ```

3. Collect OOF and integrate into meta-learner stack

### Debugging Alignment Issues

If OOF predictions don't align:

1. Check fold indices in `oof_storage['indices']` - must be consistent across models
2. Verify feature dimensions: `X.shape` before/after scaling
3. Use `preprocessing/oof_generators.py:build_oof_dataframe()` - it performs `inner` joins to find common indices
4. Run `debug_preds.py` to visualize alignment

### Hyperparameter Tuning

- Edit the **config section** in `main2.py`: change `N_LGB`, `N_CB`, etc. to run more Optuna trials
- Trials can be expensive (hours for large datasets); test with smaller `N_*` first
- Best hyperparameters are auto-saved in `oof_storage['params']` after optimization

## Reporting

The project generates **interactive HTML reports** using Plotly:

- `report_html.py` contains the report generation logic
- Reports show: actual vs predicted prices/returns, metrics tables, fold-by-fold performance
- Reports are created automatically at the end of `main2.py` and `main_compare.py`

## Key Files Reference

| File | Purpose |
|------|---------|
| `main2.py` | Main pipeline entry point |
| `main_compare.py` | Comparative analysis (ablation studies) |
| `model/bases_models/*.py` | Individual base model implementations |
| `model/meta_model/lstm_model.py` | Meta-learner (adaptive weighting) |
| `preprocessing/oof_generators.py` | OOF alignment and stacking utilities |
| `preprocessing/walk_forward.py` | Walk-forward split generator |
| `features/top_n.py` | MIC-based feature selection |
| `features/tecnical_indicators.py` | TA-Lib indicator computation |
| `features/macroeconomics.py` | External economic indicator fetchers |
| `data/yfinance_data.py` | Yahoo Finance data downloader |
| `data/ccxt_data.py` | Cryptocurrency exchange data (CCXT) |
| `report_html.py` | HTML report generation |
