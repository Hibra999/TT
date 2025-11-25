# model/bases_models/moraiMOE_model.py

import torch
import numpy as np
import pandas as pd
import gc
import warnings
from typing import Optional, Tuple, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
warnings.filterwarnings('ignore')

MODEL_SIZES = {
    'small': 'Salesforce/moirai-moe-1.0-R-small',
    'base': 'Salesforce/moirai-moe-1.0-R-base',
    'large': 'Salesforce/moirai-moe-1.0-R-large',
}

_CACHED_MODULE = None
_CACHED_MODEL_SIZE = None

def prepare_simple_dataset(
    y: Union[pd.Series, np.ndarray],
    freq: str = 'D',
    start_date: str = '2020-01-01'
) -> PandasDataset:
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    y = y.reset_index(drop=True)
    date_index = pd.date_range(start=start_date, periods=len(y), freq=freq)
    df = pd.DataFrame({'target': y.values}, index=date_index)
    df.index.name = None
    ds = PandasDataset(dict(df))
    return ds

def get_cached_module(model_size: str = 'large'):
    global _CACHED_MODULE, _CACHED_MODEL_SIZE
    if _CACHED_MODULE is None or _CACHED_MODEL_SIZE != model_size:
        if _CACHED_MODULE is not None:
            del _CACHED_MODULE
            torch.cuda.empty_cache()
            gc.collect()
        model_path = MODEL_SIZES.get(model_size, model_size)
        print(f"Cargando modulo Moirai-MoE: {model_path}")
        _CACHED_MODULE = MoiraiMoEModule.from_pretrained(model_path)
        _CACHED_MODEL_SIZE = model_size
        print(f"Modulo cargado exitosamente: {model_size}")
    return _CACHED_MODULE

def clear_module_cache():
    global _CACHED_MODULE, _CACHED_MODEL_SIZE
    if _CACHED_MODULE is not None:
        del _CACHED_MODULE
        _CACHED_MODULE = None
        _CACHED_MODEL_SIZE = None
        torch.cuda.empty_cache()
        gc.collect()

class MoiraiMoEWrapper:
    def __init__(
        self,
        model_size: str = 'large',
        prediction_length: int = 30,
        context_length: int = 240,
        patch_size: int = 16,
        num_samples: int = 100,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        use_cache: bool = True
    ):
        self.model_size = model_size
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cache = use_cache
        self.model = None
        self.predictor = None
        self._module = None
        self._load_model()

    def _load_model(self):
        try:
            if self.use_cache:
                self._module = get_cached_module(self.model_size)
            else:
                model_path = MODEL_SIZES.get(self.model_size, self.model_size)
                self._module = MoiraiMoEModule.from_pretrained(model_path)
            self.model = MoiraiMoEForecast(
                module=self._module,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            self.predictor = self.model.create_predictor(batch_size=self.batch_size)
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo Moirai-MoE ({self.model_size}): {e}")

    def predict(self, ds: PandasDataset, return_samples: bool = False) -> np.ndarray:
        forecasts = list(self.predictor.predict(ds))
        if return_samples:
            return np.array([f.samples for f in forecasts])
        else:
            return np.array([np.median(f.samples, axis=0) for f in forecasts])

def objective_moirai_moe_global(
    trial,
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
    device: Optional[torch.device] = None,
    pred_len: int = 30,
    model_size: str = 'large',
    freq: str = 'D',
    use_full_train: bool = True
) -> float:
    context_length = trial.suggest_int('context_length', 64, 512, step=32)
    patch_size = trial.suggest_categorical('patch_size', [16, 32])
    num_samples = trial.suggest_int('num_samples', 50, 200, step=25)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    try:
        if use_full_train:
            mae = _evaluate_full_train(
                y_array=y_array,
                context_length=context_length,
                patch_size=patch_size,
                num_samples=num_samples,
                batch_size=batch_size,
                pred_len=pred_len,
                model_size=model_size,
                freq=freq,
                val_size=pred_len * 2
            )
        else:
            mae = _evaluate_with_folds(
                y_array=y_array,
                splitter=splitter,
                context_length=context_length,
                patch_size=patch_size,
                num_samples=num_samples,
                batch_size=batch_size,
                pred_len=pred_len,
                model_size=model_size,
                freq=freq
            )
        return mae
    except Exception as e:
        print(f"Error en trial: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return float('inf')

def _evaluate_full_train(
    y_array: np.ndarray,
    context_length: int,
    patch_size: int,
    num_samples: int,
    batch_size: int,
    pred_len: int,
    model_size: str,
    freq: str,
    val_size: int
) -> float:
    train_end = len(y_array) - val_size
    if train_end < context_length:
        return float('inf')
    wrapper = None
    try:
        wrapper = MoiraiMoEWrapper(
            model_size=model_size,
            prediction_length=pred_len,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            batch_size=batch_size,
            use_cache=True
        )
        ds = prepare_simple_dataset(y_array, freq=freq)
        train_ds, test_template = split(ds, offset=-val_size)
        n_windows = max(1, val_size // pred_len)
        test_data = test_template.generate_instances(
            prediction_length=pred_len,
            windows=n_windows,
            distance=pred_len,
        )
        forecasts = list(wrapper.predictor.predict(test_data.input))
        all_preds = []
        all_labels = []
        for forecast, label in zip(forecasts, test_data.label):
            pred = np.median(forecast.samples, axis=0)
            actual = label['target']
            min_len = min(len(pred), len(actual))
            all_preds.extend(pred[:min_len])
            all_labels.extend(actual[:min_len])
        if len(all_preds) == 0 or len(all_labels) == 0:
            return float('inf')
        mae = mean_absolute_error(all_labels, all_preds)
        return mae
    except Exception as e:
        print(f"Error en _evaluate_full_train: {e}")
        return float('inf')
    finally:
        if wrapper is not None:
            del wrapper.model
            del wrapper.predictor
            del wrapper
        torch.cuda.empty_cache()
        gc.collect()

def _evaluate_with_folds(
    y_array: np.ndarray,
    splitter,
    context_length: int,
    patch_size: int,
    num_samples: int,
    batch_size: int,
    pred_len: int,
    model_size: str,
    freq: str
) -> float:
    mae_scores = []
    for train_idx, val_idx in splitter.split(y_array):
        y_train = y_array[train_idx]
        y_val = y_array[val_idx]
        if len(y_train) < context_length:
            continue
        wrapper = None
        try:
            wrapper = MoiraiMoEWrapper(
                model_size=model_size,
                prediction_length=min(pred_len, len(y_val)),
                context_length=min(context_length, len(y_train)),
                patch_size=patch_size,
                num_samples=num_samples,
                batch_size=batch_size,
                use_cache=True
            )
            full_series = np.concatenate([y_train, y_val])
            ds = prepare_simple_dataset(full_series, freq=freq)
            train_ds, test_template = split(ds, offset=-len(y_val))
            test_data = test_template.generate_instances(
                prediction_length=min(pred_len, len(y_val)),
                windows=1,
                distance=pred_len,
            )
            forecasts = list(wrapper.predictor.predict(test_data.input))
            for forecast, label in zip(forecasts, test_data.label):
                pred = np.median(forecast.samples, axis=0)
                actual = label['target']
                min_len = min(len(pred), len(actual))
                mae = mean_absolute_error(actual[:min_len], pred[:min_len])
                mae_scores.append(mae)
        except Exception as e:
            print(f"Error en fold: {e}")
            continue
        finally:
            if wrapper is not None:
                del wrapper.model
                del wrapper.predictor
                del wrapper
            torch.cuda.empty_cache()
            gc.collect()
    if not mae_scores:
        return float('inf')
    return np.mean(mae_scores)

def predict_with_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    pred_len: int = 30,
    model_size: str = 'large',
    freq: str = 'D'
) -> Tuple[np.ndarray, MoiraiMoEWrapper]:
    wrapper = MoiraiMoEWrapper(
        model_size=model_size,
        prediction_length=pred_len,
        context_length=best_params.get('context_length', 240),
        patch_size=best_params.get('patch_size', 16),
        num_samples=best_params.get('num_samples', 100),
        batch_size=best_params.get('batch_size', 32),
        use_cache=True
    )
    y_array = y.values if isinstance(y, pd.Series) else y
    ds = prepare_simple_dataset(y_array, freq=freq)
    forecasts = list(wrapper.predictor.predict(ds))
    if forecasts:
        predictions = np.median(forecasts[0].samples, axis=0)
        return predictions, wrapper
    return np.array([]), wrapper

def preload_moirai_module(model_size: str = 'large'):
    print(f"Precargando modulo Moirai-MoE ({model_size})...")
    module = get_cached_module(model_size)
    print("Modulo precargado exitosamente!")
    return module