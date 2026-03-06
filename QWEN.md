# TT - Time Series Ensemble Forecasting Project

## Project Overview

This is a **financial time series forecasting project** that implements an ensemble learning approach combining multiple machine learning and deep learning models. The project focuses on predicting asset price movements using technical indicators, macroeconomic data, and advanced ensemble techniques.

### Main Technologies

- **Python** with extensive ML/DL stack
- **Base Models**: LightGBM, CatBoost, TimeXer, Moirai-MoE
- **Meta-Learner**: Custom LSTM-based ensemble model
- **Data Sources**: yfinance (stock data), ccxt (crypto data), FRED (macroeconomic data)
- **Key Libraries**: PyTorch, scikit-learn, Optuna (hyperparameter optimization), TA-Lib (technical analysis)

### Architecture

```
TT/
├── data/               # Data ingestion modules
│   ├── yfinance_data.py    # Stock/ETF data downloader
│   ├── ccxt_data.py        # Cryptocurrency data downloader
│   └── tokens/             # Stored CSV data files
├── features/           # Feature engineering
│   ├── tecnical_indicators.py  # TA-Lib based technical analysis
│   ├── macroeconomics.py       # FRED macroeconomic data
│   └── top_n.py                # MIC-based feature selection
├── model/              # Model definitions
│   ├── bases_models/       # Base learners (LGB, CatBoost, TimeXer, Moirai)
│   └── meta_model/         # LSTM meta-learner for ensemble
├── preprocessing/      # Data preprocessing
│   ├── walk_forward.py     # Walk-forward cross-validation
│   └── oof_generators.py   # Out-of-fold prediction collection
├── main2.py            # Main pipeline (full training + report generation)
├── main_compare.py     # Multi-window sensitivity analysis
└── report_html.py      # Interactive HTML report generator
```

## Building and Running

### Installation

```bash
pip install -r requirements.txt
```

**Note**: PyTorch installation may require specific CUDA version. Check the commented line in `requirements.txt` for CUDA 12.8 specific packages.

### Running the Pipeline

**Full Pipeline (Download + Train + Report):**
```bash
python main2.py
```

This script:
1. Downloads data from yfinance and ccxt
2. Computes technical indicators and macroeconomic features
3. Performs MIC-based feature selection (top 15)
4. Runs walk-forward cross-validation
5. Trains base models with Optuna hyperparameter optimization
6. Trains LSTM meta-learner
7. Generates `report.html` with interactive visualizations

**Multi-Window Sensitivity Analysis:**
```bash
python main_compare.py
```

Tests different window ratios (0.3, 0.4, 0.5, 0.6, 0.7) for the walk-forward validation and generates `metrics_compare.html`.

### Configuration

Edit `main2.py` constants to customize:
```python
TOKEN = 'ETH/USDT'           # Asset to analyze
N_LGB, N_CB, N_TX, N_MO = 3, 3, 3, 3  # Optuna trials per model
START, END = '2020-01-01', '2025-12-31'  # Date range
```

## Development Conventions

### Code Style
- **Compact code**: `main2.py` uses semicolon-separated statements for brevity
- **Numba JIT**: Performance-critical functions use `@njit` decorators
- **Type hints**: Used in newer modules (meta_model, preprocessing)

### Testing Practices
- Walk-forward cross-validation with configurable k-folds
- Out-of-fold (OOF) predictions for unbiased meta-learner training
- Multiple metrics: MSE, RMSE, MAE, R²

### Key Implementation Details

1. **Feature Engineering**:
   - Technical indicators with multiple lag periods (30-365 days)
   - Macroeconomic variables from FRED (GDP, CPI, FEDFUNDS, DGS10, SOFR, UNRATE)
   - MIC (Maximal Information Coefficient) for feature selection

2. **Ensemble Strategy**:
   - Base models produce OOF predictions
   - LSTM meta-learner learns optimal dynamic weights
   - Softmax-weighted combination with temperature scaling

3. **Data Preprocessing**:
   - MinMaxScaler for features and target
   - Log returns for target variable
   - Constant/near-constant column removal

### Output Files

| File | Description |
|------|-------------|
| `report.html` | Interactive report with zoom chart and metrics |
| `metrics_compare.html` | Multi-window analysis comparison table |
| `data/tokens/*.csv` | Downloaded OHLCV data |

## Project Structure Summary

| Directory | Purpose |
|-----------|---------|
| `data/` | Data ingestion from external APIs |
| `features/` | Feature engineering (technical + macro) |
| `model/` | ML/DL model definitions and training |
| `preprocessing/` | Cross-validation and OOF generation |
