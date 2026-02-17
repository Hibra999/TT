import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import mean_absolute_error

class LSTMMetaLearner(nn.Module):
    def __init__(self,num_models,hidden_size=64,num_layers=2,dropout=0.1,min_weight=0.05,temperature=1.0):
        super().__init__()
        self.num_models,self.hidden_size,self.temperature=num_models,hidden_size,max(0.1,temperature)
        self.min_weight=max(0.01,min(min_weight,(1.0/num_models)-0.01))
        self.lstm=nn.LSTM(input_size=num_models,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout if num_layers>1 else 0.0)
        self.fc=nn.Linear(hidden_size,num_models);self._init_weights()
    def _init_weights(self):
        for n,p in self.named_parameters():
            if'weight_ih'in n:nn.init.xavier_uniform_(p)
            elif'weight_hh'in n:nn.init.orthogonal_(p)
            elif'bias'in n:nn.init.zeros_(p)
            elif'fc.weight'in n:nn.init.xavier_uniform_(p)
    def _compute_weights(self,z_t):
        sf=1.0-self.num_models*self.min_weight
        return self.min_weight+sf*F.softmax(z_t/self.temperature,dim=-1)
    def forward(self,x,return_weights=False):
        p_t=x[:,-1,:];lstm_out,_=self.lstm(x);h_t=lstm_out[:,-1,:]
        alpha_t=self._compute_weights(self.fc(h_t));y_hat=torch.sum(alpha_t*p_t,dim=-1)
        return(y_hat,alpha_t)if return_weights else y_hat

class MetaDataset(Dataset):
    def __init__(self,oof_matrix,y_true,window_size,noise_std=0.0,training=True):
        self.oof_matrix,self.y_true=oof_matrix.astype(np.float32),y_true.astype(np.float32)
        self.window_size,self.noise_std,self.training=window_size,noise_std,training
        self.valid_indices=self._get_valid_indices()
    def _get_valid_indices(self):
        valid=[]
        for t in range(self.window_size-1,len(self.y_true)):
            w=self.oof_matrix[t-self.window_size+1:t+1]
            if not np.isnan(w).any()and not np.isnan(self.y_true[t]):valid.append(t)
        return valid
    def __len__(self):return len(self.valid_indices)
    def __getitem__(self,idx):
        t=self.valid_indices[idx];X_t=self.oof_matrix[t-self.window_size+1:t+1].copy()
        if self.training and self.noise_std>0:X_t+=np.random.normal(0,self.noise_std,X_t.shape).astype(np.float32)
        return torch.from_numpy(X_t),torch.tensor(self.y_true[t])

def train_lstm_meta_model(oof_df,window_size=10,hidden_size=64,num_layers=2,dropout=0.1,lr=1e-3,weight_decay=1e-5,epochs=100,batch_size=32,patience=15,device=None,noise_std=0.0,min_weight=0.05,temperature=2.0):
    device=device or torch.device('cuda'if torch.cuda.is_available()else'cpu')
    model_cols=[c for c in oof_df.columns if c not in['idx','target']]
    oof_matrix,y_array,num_models=oof_df[model_cols].values,oof_df['target'].values,len(model_cols)
    full_ds=MetaDataset(oof_matrix,y_array,window_size,noise_std=0.0,training=False)
    if len(full_ds)<20:print(f"Dataset muy pequeno: {len(full_ds)} muestras");return None,None,None,None
    tr_sz=int(len(full_ds)*0.8);vl_sz=len(full_ds)-tr_sz
    if vl_sz<5:print(f"Validation set muy pequeno: {vl_sz}");return None,None,None,None
    tr_idx,vl_idx=list(range(tr_sz)),list(range(tr_sz,len(full_ds)))
    tr_ds,vl_ds=MetaDataset(oof_matrix,y_array,window_size,noise_std=noise_std,training=True),MetaDataset(oof_matrix,y_array,window_size,noise_std=0.0,training=False)
    tr_loader=DataLoader(torch.utils.data.Subset(tr_ds,tr_idx),batch_size=batch_size,shuffle=True)
    vl_loader=DataLoader(torch.utils.data.Subset(vl_ds,vl_idx),batch_size=batch_size,shuffle=False)
    model=LSTMMetaLearner(num_models=num_models,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,min_weight=min_weight,temperature=temperature).to(device)
    criterion,optimizer=nn.MSELoss(),torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=patience//2,min_lr=1e-6)
    best_vl,pat_cnt,best_st,tr_losses,vl_losses,best_ep=float('inf'),0,None,[],[],1
    for ep in range(epochs):
        model.train();ep_tr=0.0
        for xb,yb in tr_loader:
            xb,yb=xb.to(device),yb.to(device);optimizer.zero_grad()
            loss=criterion(model(xb),yb);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            ep_tr+=loss.item()*xb.size(0)
        ep_tr/=tr_sz;tr_losses.append(ep_tr)
        model.eval();ep_vl=0.0
        with torch.no_grad():
            for xb,yb in vl_loader:
                xb,yb=xb.to(device),yb.to(device)
                ep_vl+=criterion(model(xb),yb).item()*xb.size(0)
        ep_vl/=vl_sz;vl_losses.append(ep_vl);scheduler.step(ep_vl)
        if ep_vl<best_vl:best_vl,best_st,pat_cnt,best_ep=ep_vl,{k:v.cpu().clone()for k,v in model.state_dict().items()},0,ep+1
        else:
            pat_cnt+=1
            if pat_cnt>=patience:break
    if best_st:model.load_state_dict({k:v.to(device)for k,v in best_st.items()})
    model.eval();all_p,all_t,all_w=[],[],[]
    with torch.no_grad():
        for xb,yb in DataLoader(full_ds,batch_size=batch_size,shuffle=False):
            yp,wt=model(xb.to(device),return_weights=True)
            all_p.extend(yp.cpu().numpy());all_t.extend(yb.numpy());all_w.extend(wt.cpu().numpy())
    all_p,all_t,all_w=np.array(all_p),np.array(all_t),np.array(all_w)
    mae_m=mean_absolute_error(all_t,all_p);mse_m=np.mean((all_t-all_p)**2)
    results={'train_losses':tr_losses,'val_losses':vl_losses,'best_epoch':best_ep,'mae':mae_m,'mse':mse_m,'rmse':np.sqrt(mse_m),'predictions':all_p,'targets':all_t,'weights':all_w,'valid_indices':full_ds.valid_indices,'window_size':window_size,'model_names':model_cols,'weights_min':all_w.min(),'weights_max':all_w.max(),'weights_mean_per_model':all_w.mean(axis=0)}
    print(f"Pesos promedio por modelo: {dict(zip(model_cols,all_w.mean(axis=0).round(4)))}")
    print(f"Min peso: {all_w.min():.4f}, Max peso: {all_w.max():.4f}")
    return model,mae_m,results,device

def objective_lstm_meta(trial,oof_df,device):
    ws=trial.suggest_int('window_size',5,30);hs=trial.suggest_categorical('hidden_size',[32,64,128,256])
    nl=trial.suggest_int('num_layers',1,4);do=trial.suggest_float('dropout',0.1,0.5)
    lr=trial.suggest_float('lr',1e-5,1e-2,log=True);wd=trial.suggest_float('weight_decay',1e-6,1e-3,log=True)
    bs=trial.suggest_categorical('batch_size',[16,32,64,128]);ns=trial.suggest_float('noise_std',0.0,0.05)
    mw=trial.suggest_float('min_weight',0.02,0.20);tp=trial.suggest_float('temperature',1.0,5.0)
    model,mae,res,_=train_lstm_meta_model(oof_df=oof_df,window_size=ws,hidden_size=hs,num_layers=nl,dropout=do,lr=lr,weight_decay=wd,epochs=100,batch_size=bs,patience=15,device=device,noise_std=ns,min_weight=mw,temperature=tp)
    if model is None or mae is None:return float('inf')
    wts=res.get('weights',np.array([]))
    if len(wts)>0 and np.std(wts.mean(axis=0))<0.01:return float('inf')
    return mae

def optimize_lstm_meta(oof_df,device,n_trials=50):
    sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=min(10,n_trials//3))
    study=optuna.create_study(direction='minimize',sampler=sampler)
    study.optimize(lambda t:objective_lstm_meta(t,oof_df,device),n_trials=n_trials,n_jobs=1,show_progress_bar=True)
    bp=study.best_params
    print("\n=== Mejores Hiperparametros ===")
    for k,v in bp.items():print(f"  {k}: {v}")
    model,mae,res,device=train_lstm_meta_model(oof_df=oof_df,window_size=bp.get('window_size',10),hidden_size=bp.get('hidden_size',64),num_layers=bp.get('num_layers',2),dropout=bp.get('dropout',0.2),lr=bp.get('lr',1e-3),weight_decay=bp.get('weight_decay',1e-4),epochs=200,batch_size=bp.get('batch_size',32),patience=25,device=device,noise_std=bp.get('noise_std',0.01),min_weight=bp.get('min_weight',0.05),temperature=bp.get('temperature',2.0))
    return model,mae,res,bp,study

def get_average_weights(weights_history,model_names):
    if len(weights_history)==0:return pd.DataFrame({'Modelo':model_names,'Peso_Promedio':[np.nan]*len(model_names)})
    vw=weights_history[~np.isnan(weights_history).any(axis=1)]
    if len(vw)==0:return pd.DataFrame({'Modelo':model_names,'Peso_Promedio':[np.nan]*len(model_names)})
    return pd.DataFrame({'Modelo':model_names,'Peso_Promedio':np.mean(vw,axis=0),'Peso_Std':np.std(vw,axis=0),'Peso_Min':np.min(vw,axis=0),'Peso_Max':np.max(vw,axis=0)})

def collect_oof_predictions(oof_storage):
    if not oof_storage or'preds'not in oof_storage or'indices'not in oof_storage:return np.array([]),np.array([]),0.0
    ap,ai=[],[]
    for p,i in zip(oof_storage['preds'],oof_storage['indices']):
        pf,if_=np.array(p).flatten(),np.array(i).flatten();ml=min(len(pf),len(if_))
        ap.extend(pf[:ml]);ai.extend(if_[:ml])
    return np.array(ap),np.array(ai),oof_storage.get('best_score',0.0)

def build_oof_dataframe(oof_lgb,oof_cb,oof_tx,oof_moirai,y_train):
    ya=y_train.values if hasattr(y_train,'values')else np.array(y_train)
    p_lgb,i_lgb,_=collect_oof_predictions(oof_lgb);p_cb,i_cb,_=collect_oof_predictions(oof_cb)
    p_tx,i_tx,_=collect_oof_predictions(oof_tx);p_mo,i_mo,_=collect_oof_predictions(oof_moirai)
    dfs=[]
    if len(p_lgb)>0:dfs.append(pd.DataFrame({'idx':i_lgb.astype(int),'pred_lgb':p_lgb}))
    if len(p_cb)>0:dfs.append(pd.DataFrame({'idx':i_cb.astype(int),'pred_catboost':p_cb}))
    if len(p_tx)>0:dfs.append(pd.DataFrame({'idx':i_tx.astype(int),'pred_timexer':p_tx}))
    if len(p_mo)>0:dfs.append(pd.DataFrame({'idx':i_mo.astype(int),'pred_moirai':p_mo}))
    
    if not dfs:return pd.DataFrame()
    
    print(f"DEBUG: Found {len(dfs)} model OOF dataframes")
    for i, df in enumerate(dfs):
        print(f"DEBUG: DF {i} shape: {df.shape}, head indices: {df['idx'].head().tolist()}")
        
    res=dfs[0]
    for i, df in enumerate(dfs[1:]):
        res=pd.merge(res,df,on='idx',how='inner')
        print(f"DEBUG: After merging DF {i+1}, shape: {res.shape}")
        
    res['target']=res['idx'].apply(lambda i:ya[i]if i<len(ya)else np.nan)
    return res.dropna().reset_index(drop=True)