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
    v_rt=[0.3, 0.4, 0.45, 0.5, 0.6, 0.7];cp=df['Close'].values;sc=st.inverse_transform
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
        
        # Friedman Statistical Test per Metric and Phase
        p_vals = {}
        if friedmanchisquare is not None:
            for fase in ['Train (OOF)', 'Test']:
                p_vals[fase] = {}
                for met in ['MSE','RMSE','MAE','R2']:
                    df_m = dm[dm['Fase']==fase].copy()
                    piv = df_m.pivot(index='Mod', columns='WR', values=met).dropna()
                    if len(piv) >= 2 and len(piv.columns) >= 2:
                        try:
                            _, p = friedmanchisquare(*[piv[c] for c in piv.columns])
                            p_vals[fase][met] = p
                        except Exception: pass

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Distinctive color map for models
        color_map = {
            'LightGBM': '#1f77b4',
            'CatBoost': '#ff7f0e',
            'TimeXer': '#2ca02c',
            'Moirai-MoE': '#d62728',
            'Meta LSTM': '#9467bd'
        }
        
        # We will create two sets of traces, one for Train and one for Test
        fig = make_subplots(rows=2, cols=2, subplot_titles=['MSE','RMSE','MAE','R2'], horizontal_spacing=0.12, vertical_spacing=0.18)
        
        phases = ['Train (OOF)', 'Test']
        metrics = ['MSE','RMSE','MAE','R2']
        
        for f_idx, fase in enumerate(phases):
            f_dm = dm[dm['Fase']==fase]
            
            for m_idx, met in enumerate(metrics, 1):
                mods_in_phase = f_dm['Mod'].unique()
                for mod in mods_in_phase:
                    mod_data = f_dm[f_dm['Mod'] == mod].sort_values('WR')
                    trace = go.Bar(
                        x=mod_data['WR'], y=mod_data[met], name=mod,
                        text=mod_data[met], texttemplate='%{y:.4f}', textposition='outside',
                        legendgroup=mod, showlegend=(m_idx==1), # Show legend for both sets (visible logic handles visibility)
                        visible=(fase=='Train (OOF)'),
                        marker=dict(color=color_map.get(mod, '#7f7f7f')),
                        cliponaxis=False
                    )
                    r, c = ((m_idx-1)//2)+1, ((m_idx-1)%2)+1
                    fig.add_trace(trace, row=r, col=c)

        # Update layout with interactive buttons
        n_train = len(dm[dm['Fase']=='Train (OOF)']['Mod'].unique()) * 4
        n_test = len(dm[dm['Fase']=='Test']['Mod'].unique()) * 4
        
        # Function to generate button args
        def get_args(show_train):
            vis = [show_train]*n_train + [(not show_train)]*n_test
            title = f"<b>{'Validación (OOF)' if show_train else 'Desempeño Real (Test)'} - {token}</b>"
            # Update annotations (titles) to include Friedman p-values for the active phase
            new_annotations = []
            current_fase = 'Train (OOF)' if show_train else 'Test'
            for i, met in enumerate(metrics):
                p = p_vals.get(current_fase, {}).get(met)
                sig = f" (p={p:.4f} *)" if p is not None and p < 0.05 else (f" (p={p:.4f})" if p is not None else "")
                r, c = (i//2), (i%2)
                new_annotations.append(dict(text=f"<b>{met}</b>{sig}", xref="paper", yref="paper", x=0.25+(c*0.5), y=1.0 - (r*0.55) + 0.06, showarrow=False, font=dict(size=13), xanchor='center'))
            return [{"visible": vis}, {"annotations": new_annotations, "title": title}]

        fig.update_layout(
            title=f"<b>Análisis de Sensibilidad de Ventanas ({token})</b>",
            template='plotly_white',
            height=900,
            width=1200,
            margin=dict(t=160, b=100, l=50, r=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            updatemenus=[dict(
                type="buttons", direction="right", active=0, x=0.5, y=1.18, xanchor="center",
                buttons=[
                    dict(label="Fase de Entrenamiento (OOF)", method="update", args=get_args(True)),
                    dict(label="Fase de Evaluación (Test)", method="update", args=get_args(False))
                ]
            )]
        )

        # Initialize with Train annotations
        initial_args = get_args(True)
        fig.update_layout(annotations=initial_args[1]['annotations'])
        fig.update_yaxes(matches=None, showgrid=True, gridcolor='lightgrey', title_standoff=10)
        fig.update_xaxes(title_text="Window Ratio", tickmode='linear', dtick=0.1, tickangle=45)

        # Generate Premium HTML Report
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        premium_html = f"""<!DOCTYPE html>
<html lang="es" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Métricas - {token}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: rgba(255, 255, 255, 0.8);
            --text-main: #1e293b;
            --text-muted: #64748b;
            --accent: #3b82f6;
            --border: #e2e8f0;
            --glass-border: rgba(255, 255, 255, 0.3);
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }}

        [data-theme="dark"] {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #60a5fa;
            --border: #334155;
            --glass-border: rgba(255, 255, 255, 0.1);
            --shadow: 0 10px 15px -3px rgb(0 0 0 / 0.3);
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            transition: background-color 0.3s ease, color 0.3s ease;
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}

        .brand h1 {{ font-size: 1.75rem; font-weight: 700; letter-spacing: -0.025em; }}
        .brand p {{ color: var(--text-muted); font-size: 0.875rem; margin-top: 0.25rem; }}

        .theme-toggle {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            cursor: pointer;
            color: var(--text-main);
            font-weight: 600;
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
            box-shadow: var(--shadow);
            backdrop-filter: blur(8px);
        }}
        .theme-toggle:hover {{ transform: translateY(-1px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }}

        .dashboard-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 1.5rem;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
            transition: transform 0.2s ease;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            padding: 1.25rem;
            background: rgba(255,255,255,0.03);
            border-radius: 1rem;
            border: 1px solid var(--border);
        }}
        .stat-card span {{ font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; }}
        .stat-card h3 {{ font-size: 1.25rem; margin-top: 0.5rem; }}

        footer {{
            margin-top: 4rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        /* Plotly Customization Overrides */
        .js-plotly-plot {{ background: transparent !important; }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .container {{ animation: fadeIn 0.6s ease-out; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <h1>{token} Dashboard</h1>
                <p>Análisis de Sensibilidad de Ventanas y Comparativa de Modelos</p>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()">
                <span id="theme-icon">🌙</span> Dark Mode
            </button>
        </header>

        <div class="dashboard-card">
            {html_content}
        </div>

        <footer>
            Generado por Antigravity AI &bull; {token} Market Analysis &bull; 2024
        </footer>
    </div>

    <script>
        function toggleTheme() {{
            const html = document.documentElement;
            const icon = document.getElementById('theme-icon');
            const btnText = document.querySelector('.theme-toggle');
            
            if (html.getAttribute('data-theme') === 'light') {{
                html.setAttribute('data-theme', 'dark');
                icon.innerText = '☀️';
                btnText.innerHTML = '<span>☀️</span> Light Mode';
                updatePlotlyTheme(true);
            }} else {{
                html.setAttribute('data-theme', 'light');
                icon.innerText = '🌙';
                btnText.innerHTML = '<span>🌙</span> Dark Mode';
                updatePlotlyTheme(false);
            }}
        }}

        function updatePlotlyTheme(isDark) {{
            const plots = document.querySelectorAll('.js-plotly-plot');
            const layout = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: isDark ? '#f1f5f9' : '#1e293b' }}
            }};
            
            plots.forEach(plot => {{
                Plotly.relayout(plot, layout);
            }});
        }}

        // Initial check for system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
            // toggleTheme(); // Uncomment to auto-switch to dark
        }}
    </script>
</body>
</html>"""
        
        with open(out, 'w', encoding='utf-8') as f:
            f.write(premium_html)
        print(f"Reporte Premium generado correctamente: {out}")

if __name__=="__main__":main()
