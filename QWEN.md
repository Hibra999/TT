# TT - Financial Time Series Ensemble Forecasting

## Project Overview

This is a **machine learning project for financial time series forecasting** using ensemble methods. The project implements multiple base models (LightGBM, CatBoost, TimeXer, Moirai-MoE, XGBoost, LSTM) combined through meta-learning approaches (LSTM-based stacking) to predict asset price movements.

### Key Features

- **Multi-model ensemble**: Combines gradient boosting (LightGBM, CatBoost, XGBoost) with deep learning models (TimeXer, Moirai-MoE, LSTM)
- **Meta-learning stacking**: LSTM-based meta-learner that dynamically weights base model predictions
- **Feature engineering**: Technical indicators (TA-Lib) + macroeconomic features
- **Feature selection**: MIC (Maximal Information Coefficient) based feature selection
- **Walk-forward validation**: Time-series cross-validation with sliding window
- **Hyperparameter optimization**: Optuna-based Bayesian optimization for all models
- **Comparative analysis**: Ablation studies comparing different ensemble configurations

### Architecture

```
TT/
├── data/                    # Data downloaders and storage
│   ├── yfinance_data.py     # Stock/crypto data from Yahoo Finance
│   ├── ccxt_data.py         # Crypto data from CCXT exchange API
│   └── tokens/              # Downloaded CSV data files
├── features/                # Feature engineering
│   ├── tecnical_indicators.py  # TA-Lib technical indicators (SMA, EMA, RSI, BBANDS, etc.)
│   ├── macroeconomics.py    # Macroeconomic features
│   └── top_n.py             # MIC-based feature selection
├── model/                   # Model implementations
│   ├── bases_models/        # Base forecasting models
│   │   ├── ligthGBM_model.py
│   │   ├── catboost_model.py
│   │   ├── timexer_model.py
│   │   ├── moraiMOE_model.py
│   │   └── (sota/)          # Additional SOTA models
│   └── meta_model/          # Meta-learners
│       └── lstm_model.py    # LSTM stacking meta-learner
├── preprocessing/           # Data preprocessing
│   ├── walk_forward.py      # Walk-forward cross-validation
│   └── oof_generators.py    # Out-of-fold prediction collection
├── main2.py                 # Main pipeline (single ensemble)
├── main_compare.py          # Comparative analysis pipeline
├── report_html.py           # HTML report generation
└── requirements.txt         # Python dependencies
```

## Building and Running

### Prerequisites

- **Python 3.11+**
- **CUDA 12.8** (optional, for GPU acceleration)
- **TA-Lib** (technical analysis library - requires system installation)

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# PyTorch with CUDA 12.8 (if GPU available)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

### Running the Pipeline

**Main forecasting pipeline** (single ensemble with LightGBM, CatBoost, TimeXer, Moirai-MoE + Meta LSTM):

```bash
python main2.py
```

**Comparative analysis** (compares multiple ensemble configurations):

```bash
python main_compare.py
```

### Configuration

Edit the config section in `main2.py` or `main_compare.py`:

```python
TOKEN = 'ETH/USDT'           # Asset to forecast
N_LGB, N_CB, N_TX, N_MO = 3, 3, 3, 3  # Optuna trials per model
START, END = '2020-01-01', '2025-12-31'  # Date range
```

### Output

- `report.html` - Interactive HTML report with predictions and metrics
- `report_compare.html` - Comparative analysis between ensemble configurations
- Model predictions in USD price scale and log-return scale

## Development Conventions

### Code Style

- **Compact imports**: Multiple imports on single line where logical
- **Type hints**: Used in model definitions (PyTorch modules)
- **Numba JIT**: Performance-critical functions decorated with `@njit`
- **Error handling**: Graceful fallbacks when optional dependencies unavailable

### Testing Practices

- **Walk-forward validation**: 5-fold sliding window with 30-step forecast horizon
- **Out-of-fold predictions**: Proper separation between training and validation
- **Metrics**: MSE, RMSE, MAE, R² computed on both log-return and price scales

### Key Design Patterns

1. **OOF Storage Pattern**: Base models store out-of-fold predictions during Optuna optimization for meta-learner training
2. **Scale Transformation**: Predictions flow through multiple scales:
   - Log returns → MinMax normalized → Model predictions → Inverse transform → Price scale
3. **Dynamic Weighting**: LSTM meta-learner produces adaptive weights for base models with minimum weight constraints

### Available Scripts

| Script | Purpose |
|--------|---------|
| `main2.py` | Main ensemble pipeline |
| `main_compare.py` | Comparative analysis (Actual vs Ablation vs SOTA) |
| `debug_preds.py` | Debug prediction alignment issues |
| `debug_timexer.py` | Debug TimeXer model specifics |
| `test_align.py` | Test data alignment utilities |
| `report_html.py` | HTML report generation module |

## Dependencies

**Core ML**: scikit-learn, lightgbm, catboost, xgboost, torch, keras  
**Time Series**: sktime, skforecast, statsmodels, hmmlearn  
**Deep Learning**: transformers (TimeXer, Moirai)  
**Optimization**: optuna  
**Technical Analysis**: TA-Lib  
**Data**: yfinance, ccxt, pandas_datareader  
**Visualization**: plotly, matplotlib, seaborn  

## Data Sources

- **Yahoo Finance**: Stocks (^GSPC, AAPL, NVDA, etc.), commodities (GC=F gold), volatility (CBOE)
- **CCXT**: Cryptocurrencies (BTC/USDT, ETH/USDT)
- **Macroeconomic indicators**: Via `features/macroeconomics.py`
