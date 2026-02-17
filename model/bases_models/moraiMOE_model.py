import torch
import numpy as np
import pandas as pd
import gc
import warnings
from typing import Optional,Tuple,Union
from sklearn.metrics import mean_absolute_error
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai_moe import MoiraiMoEForecast,MoiraiMoEModule
warnings.filterwarnings('ignore')

MODEL_SIZES={'small':'Salesforce/moirai-moe-1.0-R-small','base':'Salesforce/moirai-moe-1.0-R-base','large':'Salesforce/moirai-moe-1.0-R-large'}
_CACHED_MODULE=None
_CACHED_MODEL_SIZE=None

def prepare_simple_dataset(y:Union[pd.Series,np.ndarray],freq:str='D',start_date:str='2020-01-01')->PandasDataset:
    if isinstance(y,np.ndarray):y=pd.Series(y)
    y=y.reset_index(drop=True)
    df=pd.DataFrame({'target':y.values},index=pd.date_range(start=start_date,periods=len(y),freq=freq))
    df.index.name=None
    return PandasDataset(dict(df))

def get_cached_module(model_size:str='large'):
    global _CACHED_MODULE,_CACHED_MODEL_SIZE
    if _CACHED_MODULE is None or _CACHED_MODEL_SIZE!=model_size:
        if _CACHED_MODULE is not None:del _CACHED_MODULE;torch.cuda.empty_cache();gc.collect()
        mp=MODEL_SIZES.get(model_size,model_size);print(f"Cargando modulo Moirai-MoE: {mp}")
        _CACHED_MODULE=MoiraiMoEModule.from_pretrained(mp);_CACHED_MODEL_SIZE=model_size
        print(f"Modulo cargado exitosamente: {model_size}")
    return _CACHED_MODULE

def clear_module_cache():
    global _CACHED_MODULE,_CACHED_MODEL_SIZE
    if _CACHED_MODULE is not None:
        del _CACHED_MODULE;_CACHED_MODULE=None;_CACHED_MODEL_SIZE=None
        torch.cuda.empty_cache();gc.collect()

class MoiraiMoEWrapper:
    def __init__(self,model_size:str='large',prediction_length:int=30,context_length:int=240,patch_size:int=16,num_samples:int=100,batch_size:int=32,device:Optional[torch.device]=None,use_cache:bool=True):
        self.model_size,self.prediction_length,self.context_length=model_size,prediction_length,context_length
        self.patch_size,self.num_samples,self.batch_size=patch_size,num_samples,batch_size
        self.device=device or torch.device('cuda'if torch.cuda.is_available()else'cpu')
        self.use_cache,self.model,self.predictor,self._module=use_cache,None,None,None
        self._load_model()
    def _load_model(self):
        try:
            self._module=get_cached_module(self.model_size)if self.use_cache else MoiraiMoEModule.from_pretrained(MODEL_SIZES.get(self.model_size,self.model_size))
            self.model=MoiraiMoEForecast(module=self._module,prediction_length=self.prediction_length,context_length=self.context_length,patch_size=self.patch_size,num_samples=self.num_samples,target_dim=1,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
            self.predictor=self.model.create_predictor(batch_size=self.batch_size)
        except Exception as e:raise RuntimeError(f"Error cargando modelo Moirai-MoE ({self.model_size}): {e}")
    def predict(self,ds:PandasDataset,return_samples:bool=False)->np.ndarray:
        forecasts=list(self.predictor.predict(ds))
        return np.array([f.samples for f in forecasts])if return_samples else np.array([np.median(f.samples,axis=0)for f in forecasts])

def objective_moirai_moe_global(trial,X,y,splitter,device=None,pred_len=30,model_size='small',freq='D',use_full_train=True,oof_storage=None):
    y_arr=y.values if isinstance(y,pd.Series)else np.array(y)
    fold_sizes=[(len(ti),len(vi))for ti,vi in splitter.split(y_arr)]
    max_ctx=min(512,min(fs[0]for fs in fold_sizes)-10)
    if max_ctx<64:return float('inf')
    ctx_len=trial.suggest_int('context_length',64,max_ctx,step=32)
    patch_sz=trial.suggest_categorical('patch_size',[16,32])
    n_samples=trial.suggest_int('num_samples',50,200,step=25)
    bs=trial.suggest_categorical('batch_size',[16,32,64])
    try:
        fold_scores,fold_preds,fold_indices=[],[],[]
        for fold_num,(t_idx,v_idx)in enumerate(splitter.split(y_arr)):
            y_tr,y_vl=y_arr[t_idx],y_arr[v_idx]
            eff_ctx=min(ctx_len,len(y_tr)-1)
            if eff_ctx<32:continue
            wrapper=None
            try:
                wrapper=MoiraiMoEWrapper(model_size=model_size,prediction_length=1,context_length=eff_ctx,patch_size=patch_sz,num_samples=n_samples,batch_size=bs,use_cache=True)
                full_s=np.concatenate([y_tr,y_vl]);vp,vi_list=[],[]
                for i,ti in enumerate(v_idx):
                    loc_ti=len(y_tr)+i;cs,ce=max(0,loc_ti-eff_ctx),loc_ti
                    if ce<=cs:continue
                    ctx_data=full_s[cs:ce]
                    if len(ctx_data)<10:continue
                    try:
                        ds=prepare_simple_dataset(ctx_data,freq=freq)
                        fc=list(wrapper.predictor.predict(ds))
                        if fc:vp.append(float(np.median(fc[0].samples)));vi_list.append(int(ti))
                    except:continue
                if not vp:continue
                vp,vi_arr=np.array(vp),np.array(vi_list)
                mae=mean_absolute_error(y_arr[vi_arr],vp)
                fold_scores.append(mae);fold_preds.append(vp);fold_indices.append(vi_arr)
            except:import traceback;traceback.print_exc();continue
            finally:
                if wrapper:del wrapper.model,wrapper.predictor,wrapper
                torch.cuda.empty_cache();gc.collect()
        if not fold_scores:return float('inf')
        ms=np.mean(fold_scores)
        if oof_storage is not None and('best_score'not in oof_storage or ms<oof_storage['best_score']):
            oof_storage['best_score'],oof_storage['preds'],oof_storage['indices']=ms,fold_preds,fold_indices
        return ms
    except:import traceback;traceback.print_exc();torch.cuda.empty_cache();gc.collect();return float('inf')

def predict_with_best_params(X:pd.DataFrame,y:pd.Series,best_params:dict,pred_len:int=30,model_size:str='large',freq:str='D')->Tuple[np.ndarray,MoiraiMoEWrapper]:
    wrapper=MoiraiMoEWrapper(model_size=model_size,prediction_length=pred_len,context_length=best_params.get('context_length',240),patch_size=best_params.get('patch_size',16),num_samples=best_params.get('num_samples',100),batch_size=best_params.get('batch_size',32),use_cache=True)
    y_arr=y.values if isinstance(y,pd.Series)else y
    fc=list(wrapper.predictor.predict(prepare_simple_dataset(y_arr,freq=freq)))
    return(np.median(fc[0].samples,axis=0),wrapper)if fc else(np.array([]),wrapper)

def preload_moirai_module(model_size:str='large'):
    print(f"Precargando modulo Moirai-MoE ({model_size})...");m=get_cached_module(model_size);print("Modulo precargado exitosamente!");return m

def train_final_and_predict_test(y_train, y_test, best_params, model_size='small', freq='D'):
    """
    Entrena Moirai con todo el train y predice en test punto por punto.
    Predicción recursiva: usa predicciones propias (no y_test real) en el contexto.
    """
    y_train_arr = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)
    
    # Inicializar test con ceros para evitar data leakage
    full_series = np.concatenate([y_train_arr, np.zeros(len(y_test_arr))])
    train_len = len(y_train_arr)
    
    ctx_len = best_params.get('context_length', 128)
    patch_sz = best_params.get('patch_size', 16)
    n_samples = best_params.get('num_samples', 100)
    bs = best_params.get('batch_size', 32)
    
    wrapper = MoiraiMoEWrapper(
        model_size=model_size,
        prediction_length=1,
        context_length=ctx_len,
        patch_size=patch_sz,
        num_samples=n_samples,
        batch_size=bs,
        use_cache=True
    )
    
    predictions = []
    test_indices = []
    
    for i in range(len(y_test_arr)):
        global_idx = train_len + i
        ctx_start = max(0, global_idx - ctx_len)
        ctx_end = global_idx
        
        if ctx_end <= ctx_start:
            continue
        
        ctx_data = full_series[ctx_start:ctx_end]
        
        if len(ctx_data) < 10:
            continue
        
        try:
            ds = prepare_simple_dataset(ctx_data, freq=freq)
            fc = list(wrapper.predictor.predict(ds))
            if fc:
                pred = float(np.median(fc[0].samples))
                predictions.append(pred)
                test_indices.append(i)
                
                # Escribir predicción en full_series para que contextos futuros la usen
                full_series[global_idx] = pred
        except:
            continue
    
    # Limpiar
    del wrapper.model, wrapper.predictor, wrapper
    torch.cuda.empty_cache()
    gc.collect()
    
    return np.array(predictions), np.array(test_indices)