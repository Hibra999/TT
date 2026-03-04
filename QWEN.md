# TT - Financial Time Series Forecasting Ensemble

## Project Overview

**TT** is a Python-based financial time series forecasting system that uses an ensemble of machine learning models with a meta-learning approach. The project predicts asset prices (stocks and cryptocurrencies) using technical indicators, macroeconomic data, and advanced deep learning models.

### Key Features

- **Multi-Asset Support**: Stocks (AAPL, NVDA, JNJ, KO, ^GSPC) and cryptocurrencies (BTC/USDT, ETH/USDT)
- **Ensemble Architecture**: Combines 4 base models with an LSTM meta-learner
- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning
- **Walk-Forward Validation**: Time-series cross-validation to prevent data leakage
- **Interactive Reports**: Streamlit dashboard and static HTML reports

### Base Models

| Model | Description |
|-------|-------------|
| **LightGBM** | Gradient boosting with GPU support |
| **CatBoost** | Categorical boosting algorithm |
| **TimeXer** | Time-series transformer model |
| **Moirai-MoE** | Mixture of Experts foundation model |

### Meta-Model

- **LSTM Meta-Learner**: Dynamically weights base model predictions using an LSTM network that learns temporal patterns in model performance

## Project Structure

```
TT/
├── main.py              # Streamlit interactive dashboard
├── main2.py             # Batch training script (generates HTML report)
├── report_html.py       # HTML report generator
├── requirements.txt     # Python dependencies
├── data/
│   ├── yfinance_data.py    # Stock data downloader (yfinance)
│   └── ccxt_data.py        # Crypto data downloader (ccxt)
├── features/
│   ├── tecnical_indicators.py  # TA-Lib technical indicators (SMA, EMA, RSI, etc.)
│   ├── macroeconomics.py       # Macroeconomic data features
│   └── top_n.py                # MIC-based feature selection
├── model/
│   ├── bases_models/
│   │   ├── ligthGBM_model.py   # LightGBM objective & training
│   │   ├── catboost_model.py   # CatBoost objective & training
│   │   ├── timexer_model.py    # TimeXer objective & training
│   │   └── moraiMOE_model.py   # Moirai-MoE objective & training
│   └── meta_model/
│       └── lstm_model.py       # LSTM meta-learner architecture
└── preprocessing/
    ├── walk_forward.py     # Walk-forward cross-validation splitter
    └── oof_generators.py   # Out-of-fold prediction collection
```

## Building and Running

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for deep learning models)
- TA-Lib installed on system (for `talib` Python package)

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# For PyTorch with CUDA 12.8 (if using GPU)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

### Running the Application

**Interactive Dashboard (Streamlit):**
```bash
streamlit run main.py
```

**Batch Training with Report Generation:**
```bash
python main2.py
```

### Configuration

Edit `main2.py` to customize:

```python
TOKEN = 'ETH/USDT'           # Asset to analyze
N_LGB, N_CB, N_TX, N_MO, N_MT = 3, 3, 3, 3, 3  # Trials per model
START, END = '2020-01-01', '2025-12-31'  # Date range
```

## Development Conventions

### Code Style

- **Imports**: Grouped by category (standard library, third-party, local)
- **Naming**: Spanish variable/function names in features/preprocessing, English in models
- **Type Hints**: Used in model definitions (PyTorch modules)

### Testing Practices

- **Walk-Forward Validation**: 5-fold sliding window with 30-period forecast horizon
- **Train/Test Split**: 90%/10% temporal split
- **Feature Selection**: MIC (Maximal Information Coefficient) selects top 15 features

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 5 | Number of CV folds |
| `fh_val` | 30 | Forecast horizon for validation |
| `seq_len` | 96 | Sequence length for TimeXer |
| `pred_len` | 30 | Prediction length |
| `window_size` | 5-30 | LSTM meta-learner lookback window |

## Data Flow

1. **Download**: Fetch historical data via yfinance (stocks) and ccxt (crypto)
2. **Feature Engineering**: 
   - Technical indicators (30-365 day lags for SMA, EMA, RSI, etc.)
   - Macroeconomic features
3. **Feature Selection**: MIC-based selection of top 15 features
4. **Scaling**: MinMaxScaler for features and target
5. **Base Model Training**: Optuna optimization with OOF predictions
6. **Meta-Model Training**: LSTM learns to weight base model predictions
7. **Prediction**: Ensemble combines all models with learned weights
8. **Reporting**: Generate interactive HTML report with metrics

## Output

- **report.html**: Interactive report with:
  - Zoom chart showing predictions vs actual prices
  - Metrics table (MSE, RMSE, MAE, R²)
  - Bar charts comparing model performance

## Dependencies Highlights

- **ML Libraries**: scikit-learn, catboost, lightgbm, xgboost
- **Deep Learning**: PyTorch 2.9.0, Keras 3.12.0
- **Time Series**: sktime, skforecast, statsmodels
- **Data**: yfinance, ccxt, pandas_datareader
- **Visualization**: plotly, seaborn, matplotlib, streamlit
- **Optimization**: optuna

## Notes

- The project uses Spanish comments and variable names in some modules (features, preprocessing)
- GPU is recommended for TimeXer and Moirai-MoE models
- TA-Lib must be installed at the system level before installing the Python package
