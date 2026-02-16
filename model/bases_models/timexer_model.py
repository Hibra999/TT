import math
from math import sqrt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from types import SimpleNamespace

class FlattenHead(nn.Module):
    def __init__(self,n_vars,nf,target_window,head_dropout=0):
        super().__init__()
        self.n_vars,self.flatten,self.linear,self.dropout=n_vars,nn.Flatten(start_dim=-2),nn.Linear(nf,target_window),nn.Dropout(head_dropout)
    def forward(self,x):return self.dropout(self.linear(self.flatten(x)))

class EnEmbedding(nn.Module):
    def __init__(self,n_vars,d_model,patch_len,dropout):
        super().__init__()
        self.patch_len,self.value_embedding,self.glb_token=patch_len,nn.Linear(patch_len,d_model,bias=False),nn.Parameter(torch.randn(1,n_vars,1,d_model))
        self.position_embedding,self.dropout=PositionalEmbedding(d_model),nn.Dropout(dropout)
    def forward(self,x):
        n_vars=x.shape[1];glb=self.glb_token.repeat((x.shape[0],1,1,1))
        x=x.unfold(dimension=-1,size=self.patch_len,step=self.patch_len)
        x=torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        x=self.value_embedding(x)+self.position_embedding(x)
        x=torch.reshape(x,(-1,n_vars,x.shape[-2],x.shape[-1]))
        x=torch.cat([x,glb],dim=2)
        return self.dropout(torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))),n_vars

class Encoder(nn.Module):
    def __init__(self,layers,norm_layer=None,projection=None):
        super().__init__()
        self.layers,self.norm,self.projection=nn.ModuleList(layers),norm_layer,projection
    def forward(self,x,cross,x_mask=None,cross_mask=None,tau=None,delta=None):
        for l in self.layers:x=l(x,cross,x_mask=x_mask,cross_mask=cross_mask,tau=tau,delta=delta)
        if self.norm:x=self.norm(x)
        if self.projection:x=self.projection(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,self_attention,cross_attention,d_model,d_ff=None,dropout=0.1,activation="relu"):
        super().__init__()
        d_ff=d_ff or 4*d_model
        self.self_attention,self.cross_attention=self_attention,cross_attention
        self.conv1,self.conv2=nn.Conv1d(d_model,d_ff,1),nn.Conv1d(d_ff,d_model,1)
        self.norm1,self.norm2,self.norm3=nn.LayerNorm(d_model),nn.LayerNorm(d_model),nn.LayerNorm(d_model)
        self.dropout,self.activation=nn.Dropout(dropout),F.relu if activation=="relu"else F.gelu
    def forward(self,x,cross,x_mask=None,cross_mask=None,tau=None,delta=None):
        B,L,D=cross.shape
        x=x+self.dropout(self.self_attention(x,x,x,attn_mask=x_mask,tau=tau,delta=None)[0])
        x=self.norm1(x);x_glb_ori=x[:,-1,:].unsqueeze(1);x_glb=torch.reshape(x_glb_ori,(B,-1,D))
        x_glb_attn=self.dropout(self.cross_attention(x_glb,cross,cross,attn_mask=cross_mask,tau=tau,delta=delta)[0])
        x_glb_attn=torch.reshape(x_glb_attn,(x_glb_attn.shape[0]*x_glb_attn.shape[1],x_glb_attn.shape[2])).unsqueeze(1)
        x_glb=self.norm2(x_glb_ori+x_glb_attn);y=x=torch.cat([x[:,:-1,:],x_glb],dim=1)
        y=self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        return self.norm3(x+self.dropout(self.conv2(y).transpose(-1,1)))

class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.task_name,self.features,self.seq_len,self.pred_len,self.use_norm=configs.task_name,configs.features,configs.seq_len,configs.pred_len,configs.use_norm
        self.patch_len,self.patch_num=configs.patch_len,int(configs.seq_len//configs.patch_len)
        self.n_vars=1 if configs.features=='MS'else configs.enc_in
        self.en_embedding=EnEmbedding(self.n_vars,configs.d_model,self.patch_len,configs.dropout)
        self.ex_embedding=DataEmbedding_inverted(configs.seq_len,configs.d_model,configs.embed,configs.freq,configs.dropout)
        self.encoder=Encoder([EncoderLayer(AttentionLayer(FullAttention(False,configs.factor,attention_dropout=configs.dropout,output_attention=False),configs.d_model,configs.n_heads),AttentionLayer(FullAttention(False,configs.factor,attention_dropout=configs.dropout,output_attention=False),configs.d_model,configs.n_heads),configs.d_model,configs.d_ff,dropout=configs.dropout,activation=configs.activation)for _ in range(configs.e_layers)],norm_layer=nn.LayerNorm(configs.d_model))
        self.head_nf=configs.d_model*(self.patch_num+1)
        self.head=FlattenHead(configs.enc_in,self.head_nf,configs.pred_len,head_dropout=configs.dropout)
    def forecast(self,x_enc,x_mark_enc,x_dec,x_mark_dec):
        if self.use_norm:
            means=x_enc.mean(1,keepdim=True).detach();x_enc=x_enc-means
            stdev=torch.sqrt(torch.var(x_enc,dim=1,keepdim=True,unbiased=False)+1e-5);x_enc/=stdev
        en_embed,n_vars=self.en_embedding(x_enc[:,:,-1].unsqueeze(-1).permute(0,2,1))
        ex_embed=self.ex_embedding(x_enc[:,:,:-1],x_mark_enc)
        enc_out=self.encoder(en_embed,ex_embed)
        enc_out=torch.reshape(enc_out,(-1,n_vars,enc_out.shape[-2],enc_out.shape[-1])).permute(0,1,3,2)
        dec_out=self.head(enc_out).permute(0,2,1)
        if self.use_norm:
            dec_out=dec_out*(stdev[:,0,-1:].unsqueeze(1).repeat(1,self.pred_len,1))
            dec_out=dec_out+(means[:,0,-1:].unsqueeze(1).repeat(1,self.pred_len,1))
        return dec_out
    def forecast_multi(self,x_enc,x_mark_enc,x_dec,x_mark_dec):
        if self.use_norm:
            means=x_enc.mean(1,keepdim=True).detach();x_enc=x_enc-means
            stdev=torch.sqrt(torch.var(x_enc,dim=1,keepdim=True,unbiased=False)+1e-5);x_enc/=stdev
        en_embed,n_vars=self.en_embedding(x_enc.permute(0,2,1))
        ex_embed=self.ex_embedding(x_enc,x_mark_enc)
        enc_out=self.encoder(en_embed,ex_embed)
        enc_out=torch.reshape(enc_out,(-1,n_vars,enc_out.shape[-2],enc_out.shape[-1])).permute(0,1,3,2)
        dec_out=self.head(enc_out).permute(0,2,1)
        if self.use_norm:
            dec_out=dec_out*(stdev[:,0,:].unsqueeze(1).repeat(1,self.pred_len,1))
            dec_out=dec_out+(means[:,0,:].unsqueeze(1).repeat(1,self.pred_len,1))
        return dec_out
    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec,mask=None):
        if self.task_name in('long_term_forecast','short_term_forecast'):
            dec_out=self.forecast_multi(x_enc,x_mark_enc,x_dec,x_mark_dec)if self.features=='M'else self.forecast(x_enc,x_mark_enc,x_dec,x_mark_dec)
            return dec_out[:,-self.pred_len:,:]
        return None

class TriangularCausalMask:
    def __init__(self,B,L,device="cpu"):
        self._mask=torch.triu(torch.ones((B,1,L,L),dtype=torch.bool,device=device),diagonal=1)
    @property
    def mask(self):return self._mask

class FullAttention(nn.Module):
    def __init__(self,mask_flag=True,factor=5,scale=None,attention_dropout=0.1,output_attention=False):
        super().__init__()
        self.scale,self.mask_flag,self.output_attention,self.dropout=scale,mask_flag,output_attention,nn.Dropout(attention_dropout)
    def forward(self,queries,keys,values,attn_mask,tau=None,delta=None):
        B,L,H,E=queries.shape;_,S,_,D=values.shape
        scale=self.scale or 1./sqrt(E);scores=torch.einsum("blhe,bshe->bhls",queries,keys)
        if self.mask_flag:
            if attn_mask is None:attn_mask=TriangularCausalMask(B,L,device=queries.device)
            scores.masked_fill_(attn_mask.mask,-np.inf)
        A=self.dropout(torch.softmax(scale*scores,dim=-1));V=torch.einsum("bhls,bshd->blhd",A,values)
        return(V.contiguous(),A)if self.output_attention else(V.contiguous(),None)

class AttentionLayer(nn.Module):
    def __init__(self,attention,d_model,n_heads,d_keys=None,d_values=None):
        super().__init__()
        d_keys=d_keys or(d_model//n_heads);d_values=d_values or(d_model//n_heads)
        self.inner_attention,self.n_heads=attention,n_heads
        self.query_projection,self.key_projection=nn.Linear(d_model,d_keys*n_heads),nn.Linear(d_model,d_keys*n_heads)
        self.value_projection,self.out_projection=nn.Linear(d_model,d_values*n_heads),nn.Linear(d_values*n_heads,d_model)
    def forward(self,queries,keys,values,attn_mask,tau=None,delta=None):
        B,L,_=queries.shape;_,S,_=keys.shape;H=self.n_heads
        queries,keys=self.query_projection(queries).view(B,L,H,-1),self.key_projection(keys).view(B,S,H,-1)
        values=self.value_projection(values).view(B,S,H,-1)
        out,attn=self.inner_attention(queries,keys,values,attn_mask,tau=tau,delta=delta)
        return self.out_projection(out.view(B,L,-1)),attn

class DataEmbedding_inverted(nn.Module):
    def __init__(self,c_in,d_model,embed_type='fixed',freq='h',dropout=0.1):
        super().__init__()
        self.value_embedding,self.dropout=nn.Linear(c_in,d_model),nn.Dropout(p=dropout)
    def forward(self,x,x_mark):
        x=x.permute(0,2,1)
        return self.dropout(self.value_embedding(x)if x_mark is None else self.value_embedding(torch.cat([x,x_mark.permute(0,2,1)],1)))

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe=torch.zeros(max_len,d_model).float();pe.require_grad=False
        pos=torch.arange(0,max_len).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*-(math.log(10000.0)/d_model))
        pe[:,0::2],pe[:,1::2]=torch.sin(pos*div),torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):return self.pe[:,:x.size(1)]

class SeqDataset(Dataset):
    def __init__(self,X,y,seq_len,pred_len):
        assert len(X)==len(y)
        self.seq_len,self.pred_len=seq_len,pred_len
        X=np.nan_to_num(X.astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
        y=np.nan_to_num(y.astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0).reshape(-1,1)
        self.data=np.concatenate([X,y],axis=1)
        self.n_samples=len(self.data)-self.seq_len-self.pred_len+1
        if self.n_samples<=0:raise ValueError(f"Datos insuficientes: {len(self.data)} para seq_len={seq_len}, pred_len={pred_len}")
    def __len__(self):return self.n_samples
    def __getitem__(self,idx):
        return torch.from_numpy(self.data[idx:idx+self.seq_len,:]),torch.from_numpy(self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len,-1])

def build_timexer_config(trial,enc_in,seq_len,pred_len,features='MS'):
    return SimpleNamespace(task_name='long_term_forecast',features=features,seq_len=seq_len,pred_len=pred_len,use_norm=False,
        patch_len=trial.suggest_categorical("patch_len",[4,8,12,16,24]),d_model=trial.suggest_categorical("d_model",[64,128,256]),
        dropout=trial.suggest_float("dropout",0.0,0.3),embed='fixed',freq='h',factor=trial.suggest_int("factor",1,5),
        n_heads=trial.suggest_categorical("n_heads",[4,8]),e_layers=trial.suggest_int("e_layers",1,4),
        d_ff=trial.suggest_categorical("d_ff",[256,512,1024]),activation=trial.suggest_categorical("activation",["relu","gelu"]),enc_in=enc_in)

def build_model_from_trial(trial,enc_in,seq_len,pred_len,device,features='MS',pretrained_path=None,freeze_backbone=False):
    model=Model(build_timexer_config(trial,enc_in,seq_len,pred_len,features)).to(device)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path,map_location=device),strict=False)
        if freeze_backbone:
            for n,p in model.named_parameters():
                if"head"not in n:p.requires_grad=False
    return model

def objective_timexer_global(trial,X,y,splitter,device=None,seq_len=96,pred_len=30,features='MS',pretrained_path=None,freeze_backbone=False,oof_storage=None):
    device=device or torch.device('cuda'if torch.cuda.is_available()else'cpu')
    enc_in=X.shape[1]+1
    batch_size,lr=trial.suggest_categorical("batch_size",[16,32,64]),trial.suggest_float("lr",1e-5,1e-3,log=True)
    weight_decay,max_epochs,patience_val=trial.suggest_float("weight_decay",1e-6,1e-3,log=True),trial.suggest_int("max_epochs",10,50),trial.suggest_int("patience",5,15)
    X_v=np.nan_to_num((X.values if hasattr(X,'values')else np.array(X)).astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    y_v=np.nan_to_num((y.values if hasattr(y,'values')else np.array(y)).astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0).reshape(-1,1)
    full_data=np.concatenate([X_v,y_v],axis=1)
    fold_scores,fold_preds,fold_indices=[],[],[]
    for fold_num,(t_idx,v_idx)in enumerate(splitter.split(y)):
        ts,te=int(t_idx[0]),int(t_idx[-1])+1
        if te-ts<seq_len+pred_len+10:continue
        train_data=full_data[ts:te]
        class TrainDataset(Dataset):
            def __init__(s,data,sl,pl):s.data,s.seq_len,s.pred_len,s.n_samples=data,sl,pl,len(data)-sl-pl+1
            def __len__(s):return max(0,s.n_samples)
            def __getitem__(s,i):return torch.from_numpy(s.data[i:i+s.seq_len].astype(np.float32)),torch.from_numpy(s.data[i+s.seq_len:i+s.seq_len+s.pred_len,-1].astype(np.float32))
        train_ds=TrainDataset(train_data,seq_len,pred_len)
        if len(train_ds)<=0:continue
        train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,drop_last=True)
        if len(train_loader)==0:continue
        model=build_model_from_trial(trial,enc_in=enc_in,seq_len=seq_len,pred_len=pred_len,device=device,features=features,pretrained_path=pretrained_path,freeze_backbone=freeze_backbone)
        criterion,optimizer=nn.L1Loss(),torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=weight_decay)
        best_loss,pat_cnt=float("inf"),0
        for _ in range(max_epochs):
            model.train();losses=[]
            for xb,yb in train_loader:
                xb,yb=xb.to(device),yb.to(device);optimizer.zero_grad()
                loss=criterion(model(xb,None,None,None).squeeze(-1),yb)
                if torch.isnan(loss):return float("inf")
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(loss.item())
            ml=np.mean(losses)
            if ml<best_loss:best_loss,pat_cnt=ml,0
            else:
                pat_cnt+=1
                if pat_cnt>=patience_val:break
        model.eval();vp,vi=[],[]
        with torch.no_grad():
            for ti in v_idx:
                ti=int(ti);ws,we=ti-seq_len,ti
                if ws<0 or we-ws!=seq_len:continue
                xt=torch.from_numpy(full_data[ws:we].astype(np.float32)).unsqueeze(0).to(device)
                vp.append(float(model(xt,None,None,None).squeeze(-1)[0,0].cpu().numpy()));vi.append(ti)
        if not vp:continue
        vp,vi=np.array(vp),np.array(vi);vt=y_v[vi].flatten()
        fm=np.mean(np.abs(vp-vt))
        if np.isnan(fm):continue
        fold_scores.append(fm);fold_preds.append(vp);fold_indices.append(vi)
        trial.report(fm,fold_num)
        if trial.should_prune():raise optuna.TrialPruned()
    if not fold_scores:return float("inf")
    ms=float(np.mean(fold_scores))
    if oof_storage and('best_score'not in oof_storage or ms<oof_storage['best_score']):
        oof_storage['best_score'],oof_storage['preds'],oof_storage['indices']=ms,fold_preds,fold_indices
    return ms

def train_final_and_predict_test(X_train, y_train, X_test, y_test, best_params, device, seq_len=96, pred_len=1, features='MS'):
    """
    Entrena TimeXer con todo el train y predice en test punto por punto.
    """
    enc_in = X_train.shape[1] + 1
    
    # Preparar datos de entrenamiento
    X_train_v = np.nan_to_num(X_train.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_train_v = np.nan_to_num(y_train.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1)
    train_data = np.concatenate([X_train_v, y_train_v], axis=1)
    
    # Preparar datos de test (features solamente, y=0 para evitar leakage)
    X_test_v = np.nan_to_num(X_test.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    test_data_noy = np.concatenate([X_test_v, np.zeros((len(X_test_v), 1), dtype=np.float32)], axis=1)
    
    # Datos completos para predicción rolling (test y inicializado a 0)
    full_data = np.concatenate([train_data, test_data_noy], axis=0)
    train_len = len(train_data)
    
    # Crear config con mejores params
    config = SimpleNamespace(
        task_name='long_term_forecast',
        features=features,
        seq_len=seq_len,
        pred_len=pred_len,
        use_norm=False,
        patch_len=best_params.get('patch_len', 16),
        d_model=best_params.get('d_model', 128),
        dropout=best_params.get('dropout', 0.1),
        embed='fixed',
        freq='h',
        factor=best_params.get('factor', 3),
        n_heads=best_params.get('n_heads', 8),
        e_layers=best_params.get('e_layers', 2),
        d_ff=best_params.get('d_ff', 512),
        activation=best_params.get('activation', 'gelu'),
        enc_in=enc_in
    )
    
    # Construir modelo
    model = Model(config).to(device)
    
    # Dataset de entrenamiento
    class TrainDataset(Dataset):
        def __init__(self, data, sl, pl):
            self.data = data
            self.seq_len = sl
            self.pred_len = pl
            self.n_samples = len(data) - sl - pl + 1
        
        def __len__(self):
            return max(0, self.n_samples)
        
        def __getitem__(self, i):
            return (
                torch.from_numpy(self.data[i:i+self.seq_len].astype(np.float32)),
                torch.from_numpy(self.data[i+self.seq_len:i+self.seq_len+self.pred_len, -1].astype(np.float32))
            )
    
    train_ds = TrainDataset(train_data, seq_len, pred_len)
    batch_size = best_params.get('batch_size', 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Entrenar
    criterion = nn.L1Loss()
    lr = best_params.get('lr', 1e-4)
    weight_decay = best_params.get('weight_decay', 1e-5)
    max_epochs = best_params.get('max_epochs', 30)
    patience_val = best_params.get('patience', 10)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_loss, pat_cnt = float("inf"), 0
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, None, None, None).squeeze(-1)
            loss = criterion(out, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        if losses:
            ml = np.mean(losses)
            if ml < best_loss:
                best_loss, pat_cnt = ml, 0
            else:
                pat_cnt += 1
                if pat_cnt >= patience_val:
                    break
    
    # Predecir en test (recursivo: usar predicciones propias, no y_test real)
    model.eval()
    predictions = []
    test_indices = []
    
    with torch.no_grad():
        for i in range(len(X_test_v)):
            global_idx = train_len + i
            window_start = global_idx - seq_len
            window_end = global_idx
            
            if window_start < 0:
                continue
            
            x_window = full_data[window_start:window_end]
            xt = torch.from_numpy(x_window.astype(np.float32)).unsqueeze(0).to(device)
            out = model(xt, None, None, None).squeeze(-1)
            pred = float(out[0, 0].cpu().numpy())
            predictions.append(pred)
            test_indices.append(i)
            
            # Escribir predicción en full_data para que ventanas futuras la usen
            full_data[global_idx, -1] = pred
    
    return np.array(predictions), np.array(test_indices), model