import os,warnings,torch;import pandas as pd;import numpy as np;import optuna,plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from numba import njit
from plotly.subplots import make_subplots
try:
    from scipy.stats import friedmanchisquare
except ImportError:
    friedmanchisquare = None

from features.macroeconomics import macroeconomicos
from features.tecnical_indicators import TA
from features.top_n import top_k
from model.bases_models.ligthGBM_model import objective_global,train_final_and_predict_test as t_lgb
from model.bases_models.catboost_model import objective_catboost_global,train_final_and_predict_test as t_cb
from model.bases_models.timexer_model import objective_timexer_global,train_final_and_predict_test as t_tx
from model.bases_models.moraiMOE_model import objective_moirai_moe_global,preload_moirai_module,train_final_and_predict_test as t_mo
from model.meta_model.lstm_model import optimize_lstm_meta
from preprocessing.oof_generators import collect_oof_predictions,build_oof_dataframe
from preprocessing.walk_forward import wfrw

warnings.filterwarnings("ignore");optuna.logging.set_verbosity(optuna.logging.ERROR)

@njit(fastmath=True)
def fast_metrics(yt,yp):
    err=yt-yp;mse=np.mean(err**2);mae=np.mean(np.abs(err))
    return mse,np.sqrt(mse),mae

def main():
    bd=os.path.dirname(os.path.abspath(__file__))
    token="AAPL"
    f=os.path.join(bd,"data","tokens",f"{token}_2020-2025.csv")
    if not os.path.exists(f):return print(f"File missing: {f}")
    df=pd.read_csv(f); lc=np.log(df["Close"]/df["Close"].shift(-1)).dropna().reset_index(drop=True)
    dff=pd.concat([TA(df).reset_index(drop=True),macroeconomicos(df["Date_final"]).reset_index(drop=True)],axis=1).iloc[1:]
    ml=min(len(dff),len(lc));dff=dff.iloc[:ml].reset_index(drop=True);lc=lc.iloc[:ml]
    cd=[c for c in dff.columns if dff[c].max()-dff[c].min()<1e-8];dff=dff.drop(columns=cd).replace([np.inf,-np.inf],0.0)
    if lc.max()-lc.min()<1e-8:return
    ts=int(len(dff)*0.9);Xr,yr=dff.iloc[:ts].copy(),lc.iloc[:ts].copy()
    Xte_r,yte_r=dff.iloc[ts:].copy(),lc.iloc[ts:].copy()
    
    sf,st=MinMaxScaler(),MinMaxScaler();Xs=pd.DataFrame(sf.fit_transform(Xr),columns=Xr.columns)
    ys=pd.Series(st.fit_transform(yr.values.reshape(-1,1)).flatten(),name='lc')
    ft,_=top_k(Xs,ys,15);X=Xs[ft].reset_index(drop=True)
    
    Xt=pd.DataFrame(sf.transform(Xte_r),columns=Xte_r.columns)[ft].reset_index(drop=True)
    yt=pd.Series(st.transform(yte_r.values.reshape(-1,1)).flatten(),name='lc')
    
    dv=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=[]
    v_rt=[0.3, 0.4, 0.42, 0.45, 0.48, 0.5, 0.6, 0.7];cp=df['Close'].values;sc=st.inverse_transform
    preload_moirai_module(model_size='small')
    
    for wr in v_rt:
        print(f"Probando Ventana: {wr}")
        ol,oc,ot,om={},{},{},{};sp=wfrw(ys,k=5,fh_val=30,window_ratio=wr)
        optuna.create_study().optimize(lambda t:objective_global(t,X,ys,sp,oof_storage=ol),n_trials=2,n_jobs=-1)
        optuna.create_study().optimize(lambda t:objective_catboost_global(t,X,ys,sp,oof_storage=oc),n_trials=2,n_jobs=-1)
        optuna.create_study().optimize(lambda t:objective_timexer_global(t,X,ys,sp,device=dv,seq_len=96,pred_len=30,features='MS',oof_storage=ot),n_trials=1)
        optuna.create_study().optimize(lambda t:objective_moirai_moe_global(t,X,ys,sp,device=dv,pred_len=30,model_size='small',freq='D',use_full_train=True,oof_storage=om),n_trials=1)
        
        odf=build_oof_dataframe(ol,oc,ot,om,ys)
        if len(odf)<20:continue
        mdl,_,rs,bp_meta,_=optimize_lstm_meta(odf,dv,n_trials=1)
        ws=bp_meta.get('window_size',10)
        
        # VALIDATION SCORING (OOF)
        vi=np.array(rs['valid_indices']);vi=vi[(vi>0)&(vi<len(cp))]
        if not len(vi):continue
        pa=cp[vi-1]
        pr=pa*np.exp(sc(np.array(rs['targets']).reshape(-1,1)[:len(vi)]).flatten())
        pm=pa*np.exp(sc(np.array(rs['predictions']).reshape(-1,1)[:len(vi)]).flatten())
        ms,rm,ma=fast_metrics(pr,pm);m.append({'Fase':'Train (OOF)','WR':wr,'Mod':'Meta LSTM','MSE':ms,'RMSE':rm,'MAE':ma,'R2':r2_score(pr,pm)})
        
        def _get(df_c,kws):return next((c for c in df_c.columns if any(k in c.lower() for k in kws)),None)
        
        d={'LightGBM':_get(odf,['lgb','light']),'CatBoost':_get(odf,['cb','catboost']),'TimeXer':_get(odf,['tx','timex']),'Moirai-MoE':_get(odf,['moirai','moe'])}
        for n,c in d.items():
            if c:
                pxn=pa*np.exp(sc(odf.loc[vi,c].values.reshape(-1,1)).flatten())
                ms,rm,ma=fast_metrics(pr,pxn);m.append({'Fase':'Train (OOF)','WR':wr,'Mod':n,'MSE':ms,'RMSE':rm,'MAE':ma,'R2':r2_score(pr,pxn)})
                
        # TEST SET PROJECTIONS
        pl,pc,ptx,pmo=None,None,None,None
        if 'params' in ol: pl,_=t_lgb(X,ys,Xt,ol['params'])
        if 'params' in oc: pc,_=t_cb(X,ys,Xt,oc['params'])
        if 'params' in ot: ptx,tx_idx,_=t_tx(X,ys,Xt,yt,ot['params'],dv,seq_len=96,pred_len=1)
        if 'params' in om: pmo,mo_idx=t_mo(ys,yt,om['params'],model_size='small',freq='D')
        
        # Align Test sets - use OOF column names so Meta-LSTM gets same order
        tlen=len(yt); oof_m_cols=[c for c in odf.columns if c not in ['idx','target']]
        # Map: display name -> oof col name
        col_map={'LightGBM':_get(odf,['lgb','light']),'CatBoost':_get(odf,['cb','catboost']),'TimeXer':_get(odf,['tx','timex']),'Moirai-MoE':_get(odf,['moirai','moe'])}
        test_map={}
        if pl is not None and col_map.get('LightGBM'): test_map[col_map['LightGBM']]=pl.flatten()
        if pc is not None and col_map.get('CatBoost'): test_map[col_map['CatBoost']]=pc.flatten()
        if ptx is not None and col_map.get('TimeXer'):
            tx_arr=np.full(tlen,np.nan);tx_arr[tx_idx]=ptx;test_map[col_map['TimeXer']]=tx_arr
        if pmo is not None and col_map.get('Moirai-MoE'):
            mo_arr=np.full(tlen,np.nan);mo_arr[mo_idx]=pmo;test_map[col_map['Moirai-MoE']]=mo_arr
            
        tdf=pd.DataFrame(test_map); tdf['idx']=np.arange(tlen); tdf['target']=yt.values
        tdf=tdf.dropna().reset_index(drop=True)
        if len(tdf)<5: continue
        
        # Predict Meta-LSTM in Test (only if all base model columns present)
        m_cols=[c for c in tdf.columns if c not in ['idx','target']]
        if mdl and len(tdf)>0 and set(oof_m_cols)==set(m_cols):
            oof_mat=odf[oof_m_cols].values
            tt_mat=tdf[oof_m_cols].values
            comp_mat=np.vstack([oof_mat[-(ws-1):],tt_mat]) if ws>1 and len(oof_mat)>=(ws-1) else tt_mat
            
            p_meta=[]
            with torch.no_grad():
                for t in range(ws-1,len(comp_mat)):
                    w_t=comp_mat[t-ws+1:t+1]
                    t_in=torch.from_numpy(w_t.astype(np.float32)).unsqueeze(0).to(dv)
                    p_meta.append(float(mdl(t_in)[0].cpu().numpy()))
            
            if len(p_meta)==len(tdf):
                tdf['Meta LSTM']=p_meta
            
        # SCORE TEST - map oof col names back to display names
        rev_map={v:k for k,v in col_map.items() if v}
        t_vi=tdf['idx'].values+ts
        t_vi=t_vi[(t_vi>0)&(t_vi<len(cp))]
        if not len(t_vi):continue
        t_pa=cp[t_vi-1]
        t_pr=t_pa*np.exp(sc(tdf['target'].values[:len(t_vi)].reshape(-1,1)).flatten())
        
        for tgt in m_cols+(['Meta LSTM'] if 'Meta LSTM' in tdf.columns else []):
            t_pxn=t_pa*np.exp(sc(tdf[tgt].values[:len(t_vi)].reshape(-1,1)).flatten())
            ms,rm,ma=fast_metrics(t_pr,t_pxn)
            display_name=rev_map.get(tgt,tgt)
            m.append({'Fase':'Test','WR':wr,'Mod':display_name,'MSE':ms,'RMSE':rm,'MAE':ma,'R2':r2_score(t_pr,t_pxn)})

    if m:
        dm=pd.DataFrame(m);out=os.path.join(bd,"metrics_compare.html")
        # Multi-facet Plotly 
        mm=dm.melt(id_vars=['WR','Mod','Fase'],value_vars=['MSE','RMSE','MAE','R2'],var_name='Met',value_name='Val')
        mm['Subplot']=mm['Fase']+" | "+mm['Met']
        
        # Friedman Statistical Test per Metric
        p_vals = {}
        if friedmanchisquare is not None:
            for met in ['MSE','RMSE','MAE','R2']:
                df_m = dm.copy()
                df_m['Block'] = df_m['Mod'] + " | " + df_m['Fase']
                piv = df_m.pivot(index='Block', columns='WR', values=met).dropna()
                if len(piv) >= 2 and len(piv.columns) >= 2:
                    try:
                        _, p = friedmanchisquare(*[piv[c] for c in piv.columns])
                        p_vals[met] = p
                    except Exception:
                        pass

        # Build grouped bar chart using ploty subplots manually to allow 2x4 grid
        fig=px.bar(mm,x='WR',y='Val',color='Mod',barmode='group',facet_col='Met',facet_row='Fase',text_auto='.3s',title=f"Sensibilidad Multi-Ventana de Modelos e Inferencia Estadística ({token})",height=800)
        
        fig.update_layout(template='plotly_white',title_x=0.5,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
        fig.update_yaxes(matches=None)
        
        # Inject p-values into annotations
        for a in fig.layout.annotations:
            if 'Met=' in a.text:
                met_name = a.text.split('=')[1]
                p = p_vals.get(met_name)
                if p is not None:
                    sig = "⭐" if p < 0.05 else "❌"
                    a.text = f"{met_name} (Friedman p={p:.3f} {sig})"
                else:
                    a.text = met_name
            elif 'Fase=' in a.text:
                a.text = f"<b>{a.text.split('=')[1]}</b>"
        
        for r,fase in enumerate(['Test','Train (OOF)']):
            for c,met in enumerate(['MSE','RMSE','MAE','R2']):
                sub=mm[(mm['Fase']==fase)&(mm['Met']==met)]
                if len(sub)==0: continue
                bv=sub['Val'].max() if met=='R2' else sub['Val'].min()
                if pd.isna(bv): continue
                br=sub[sub['Val']==bv].iloc[0]
                
                # Plotly assigns axes from bottom-left
                row_idx=2 if r==0 else 1 # facet_row='Fase' orders bottom to top: Train first? Plotly reverses alphabetical if string? Train(OOF), Test. Let's trace it.
                # Find matching subplot data directly from x/y/mod combination
                for d in fig.data:
                    if d.name==br['Mod']:
                        if hasattr(d,'axis') or True:
                            c_mark=list(d.marker.color) if isinstance(d.marker.color,(list,tuple,np.ndarray)) else [d.marker.color]*len(d.x)
                            c_mark_mod=False
                            for i,xx in enumerate(d.x):
                                # Check if this trace point belongs to this Subplot by matching the exact value
                                if xx==br['WR'] and abs(d.y[i]-bv)<1e-9:
                                    # Since plotly might reuse traces across facets with yaxis2, yaxis3 etc
                                    # We can color it by trusting the unique matching (WR, Val, Mod) 
                                    c_mark[i]='#2ecc71'
                                    c_mark_mod=True
                            if c_mark_mod:
                                d.marker.color=c_mark

        fig.write_html(out)
        print(f"Grafico HTML 2x4 generado: {out}")

if __name__=="__main__":main()
