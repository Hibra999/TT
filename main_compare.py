import os,warnings,torch;import pandas as pd;import numpy as np;import optuna,plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from numba import njit

from features.macroeconomics import macroeconomicos
from features.tecnical_indicators import TA
from features.top_n import top_k
from model.bases_models.ligthGBM_model import objective_global
from model.bases_models.catboost_model import objective_catboost_global
from model.bases_models.timexer_model import objective_timexer_global
from model.bases_models.moraiMOE_model import objective_moirai_moe_global,preload_moirai_module
from model.meta_model.lstm_model import optimize_lstm_meta
from preprocessing.oof_generators import collect_oof_predictions,build_oof_dataframe
from preprocessing.walk_forward import wfrw

warnings.filterwarnings("ignore");optuna.logging.set_verbosity(optuna.logging.ERROR)

@njit(fastmath=True)
def fast_metrics(yt,yp):
    err=yt-yp;mse=np.mean(err**2);mae=np.mean(np.abs(err))
    return mse,np.sqrt(mse),mae

def main():
    bd=os.path.dirname(os.path.abspath(__file__));f=os.path.join(bd,"data","tokens","AAPL2020-2025.csv")
    if not os.path.exists(f):return print(f"File missing: {f}")
    df=pd.read_csv(f); lc=np.log(df["Close"]/df["Close"].shift(-1)).dropna().reset_index(drop=True)
    dff=pd.concat([TA(df).reset_index(drop=True),macroeconomicos(df["Date_final"]).reset_index(drop=True)],axis=1).iloc[1:]
    ml=min(len(dff),len(lc));dff=dff.iloc[:ml].reset_index(drop=True);lc=lc.iloc[:ml]
    cd=[c for c in dff.columns if dff[c].max()-dff[c].min()<1e-8];dff=dff.drop(columns=cd).replace([np.inf,-np.inf],0.0)
    if lc.max()-lc.min()<1e-8:return
    ts=int(len(dff)*0.9);Xr,yr=dff.iloc[:ts].copy(),lc.iloc[:ts].copy()
    sf,st=MinMaxScaler(),MinMaxScaler();Xs=pd.DataFrame(sf.fit_transform(Xr),columns=Xr.columns)
    ys=pd.Series(st.fit_transform(yr.values.reshape(-1,1)).flatten(),name='lc')
    ft,_=top_k(Xs,ys,15);X=Xs[ft].reset_index(drop=True)
    
    dv=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=[]
    v_rt=[0.3,0.4,0.5,0.6,0.7];cp=df['Close'].values;sc=st.inverse_transform
    preload_moirai_module(model_size='small')
    
    for wr in v_rt:
        ol,oc,ot,om={},{},{},{};sp=wfrw(ys,k=5,fh_val=30,window_ratio=wr)
        optuna.create_study().optimize(lambda t:objective_global(t,X,ys,sp,oof_storage=ol),n_trials=2,n_jobs=-1)
        optuna.create_study().optimize(lambda t:objective_catboost_global(t,X,ys,sp,oof_storage=oc),n_trials=2,n_jobs=-1)
        optuna.create_study().optimize(lambda t:objective_timexer_global(t,X,ys,sp,device=dv,seq_len=96,pred_len=30,features='MS',oof_storage=ot),n_trials=1)
        optuna.create_study().optimize(lambda t:objective_moirai_moe_global(t,X,ys,sp,device=dv,pred_len=30,model_size='small',freq='D',use_full_train=True,oof_storage=om),n_trials=1)
        
        odf=build_oof_dataframe(ol,oc,ot,om,ys)
        if len(odf)<20:continue
        _,_,rs,_,_=optimize_lstm_meta(odf,dv,n_trials=1)
        
        vi=np.array(rs['valid_indices']);vi=vi[(vi>0)&(vi<len(cp))] # bounds
        if not len(vi):continue
        pa=cp[vi-1] # Vectorized previous prices
        pr=pa*np.exp(sc(np.array(rs['targets']).reshape(-1,1)[:len(vi)]).flatten())
        pm=pa*np.exp(sc(np.array(rs['predictions']).reshape(-1,1)[:len(vi)]).flatten())
        ms,rm,ma=fast_metrics(pr,pm);m.append({'WR':wr,'Mod':'Meta LSTM','MSE':ms,'RMSE':rm,'MAE':ma,'R2':r2_score(pr,pm)})
        
        def _get(df_c,kws):
            return next((c for c in df_c.columns if any(k in c.lower() for k in kws)),None)
        
        d={'LightGBM':_get(odf,['lgb','light']),'CatBoost':_get(odf,['cb','catboost']),'TimeXer':_get(odf,['tx','timex']),'Moirai-MoE':_get(odf,['moirai','moe'])}
        for n,c in d.items():
            if c:
                pxn=pa*np.exp(sc(odf.loc[vi,c].values.reshape(-1,1)).flatten())
                ms,rm,ma=fast_metrics(pr,pxn);m.append({'WR':wr,'Mod':n,'MSE':ms,'RMSE':rm,'MAE':ma,'R2':r2_score(pr,pxn)})
                
    if m:
        dm=pd.DataFrame(m);out=os.path.join(bd,"metrics_compare.html")
        fig=px.bar(dm,x='WR',y='MAE',color='Mod',barmode='group',text_auto='.2f',title="Sensibilidad Multi-Ventana: MAE por Modelo en Escala Real (Menor es Mejor)",labels={'WR':'Window Ratio (Proporción Ventana)'})
        fig.update_layout(template='plotly_dark',title_x=0.5,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5));fig.write_html(out)
        print(f"Grafico Barras HTML generado en: {out}")
        b=dm.loc[dm['MAE'].idxmin()]
        print(f"MEJOR Mod: {b['Mod']} | WR: {b['WR']} | MAE: {b['MAE']}")

if __name__=="__main__":main()
