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


def objective_moirai_moe_global(trial, X, y, splitter, device=None, pred_len=30, model_size='small', freq='D', use_full_train=True, oof_storage=None):
    
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    
    fold_sizes = []
    for train_idx, val_idx in splitter.split(y_array):
        fold_sizes.append((len(train_idx), len(val_idx)))
    
    min_train_size = min(fs[0] for fs in fold_sizes)
    
    max_context = min(512, min_train_size - 10)
    if max_context < 64:
        return float('inf')
    
    context_length = trial.suggest_int('context_length', 64, max_context, step=32)
    patch_size = trial.suggest_categorical('patch_size', [16, 32])
    num_samples = trial.suggest_int('num_samples', 50, 200, step=25)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    try:
        fold_scores = []
        fold_preds = []
        fold_indices = []
        
        for fold_num, (train_idx, val_idx) in enumerate(splitter.split(y_array)):
            y_train = y_array[train_idx]
            y_val = y_array[val_idx]
            
            effective_context = min(context_length, len(y_train) - 1)
            if effective_context < 32:
                continue
            
            wrapper = None
            try:
                wrapper = MoiraiMoEWrapper(
                    model_size=model_size,
                    prediction_length=1,
                    context_length=effective_context,
                    patch_size=patch_size,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    use_cache=True
                )
                
                full_series = np.concatenate([y_train, y_val])
                
                val_preds = []
                val_indices_list = []
                
                for i, target_idx in enumerate(val_idx):
                    local_target_idx = len(y_train) + i
                    
                    context_start = max(0, local_target_idx - effective_context)
                    context_end = local_target_idx
                    
                    if context_end <= context_start:
                        continue
                    
                    context_data = full_series[context_start:context_end]
                    
                    if len(context_data) < 10:
                        continue
                    
                    try:
                        ds = prepare_simple_dataset(context_data, freq=freq)
                        forecasts = list(wrapper.predictor.predict(ds))
                        
                        if forecasts:
                            pred_value = np.median(forecasts[0].samples)
                            val_preds.append(float(pred_value))
                            val_indices_list.append(int(target_idx))
                    except Exception as e:
                        continue
                
                if len(val_preds) == 0:
                    continue
                
                val_preds = np.array(val_preds)
                val_indices_arr = np.array(val_indices_list)
                val_targets = y_array[val_indices_arr]
                
                mae = mean_absolute_error(val_targets, val_preds)
                fold_scores.append(mae)
                fold_preds.append(val_preds)
                fold_indices.append(val_indices_arr)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
            finally:
                if wrapper is not None:
                    del wrapper.model, wrapper.predictor, wrapper
                torch.cuda.empty_cache()
                gc.collect()
        
        if not fold_scores:
            return float('inf')
        
        mean_score = np.mean(fold_scores)
        
        if oof_storage is not None:
            if 'best_score' not in oof_storage or mean_score < oof_storage['best_score']:
                oof_storage['best_score'] = mean_score
                oof_storage['preds'] = fold_preds
                oof_storage['indices'] = fold_indices
        
        return mean_score
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return float('inf')


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