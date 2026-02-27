import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import os
import warnings
import streamlit as st
import matplotlib.pyplot as plt
import optuna
from data.yfinance_data import download_yf
from data.ccxt_data import download_cx
from features.macroeconomics import macroeconomicos
from model.bases_models.ligthGBM_model import objective_global
from model.bases_models.catboost_model import objective_catboost_global
from model.bases_models.timexer_model import objective_timexer_global
from model.bases_models.moraiMOE_model import objective_moirai_moe_global,preload_moirai_module
from model.meta_model.lstm_model import optimize_lstm_meta,get_average_weights
from preprocessing.oof_generators import collect_oof_predictions,build_oof_dataframe
from preprocessing.walk_forward import wfrw
from features.tecnical_indicators import TA
from features.top_n import top_k
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
warnings.filterwarnings("ignore")

PROFESSIONAL_STYLE = {
    'layout': {
        'font': dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
        'title_font': dict(family='Arial, sans-serif', size=16, color='#2c3e50', weight='bold'),
        'legend_font': dict(size=12, family='Arial'),
        'paper_bgcolor': '#f6f1e9',
        'plot_bgcolor': '#f6f1e9',
        'hoverlabel': dict(font_size=12, font_family="Arial"),
        'margin': dict(l=60, r=40, t=80, b=60),
        'hovermode': 'x unified'
    },
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'tertiary': '#2ca02c',
        'quaternary': '#d62728',
        'quinary': '#9467bd',
        'senary': '#8c564b',
        'grid': 'rgba(128, 128, 128, 0.1)',
        'grid_dark': 'rgba(128, 128, 128, 0.2)'
    }
}

COLORS={
    'primary':'#1f77b4',
    'secondary':'#ff7f0e',
    'success':'#2ca02c',
    'danger':'#d62728',
    'purple':'#9467bd',
    'brown':'#8c564b',
    'pink':'#e377c2',
    'gray':'#7f7f7f',
    'olive':'#bcbd22',
    'cyan':'#17becf',
    'black':'#2c3e50',
    'background':'#f6f1e9'
}

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stApp {background-color: #f6f1e9;}
    h1, h2, h3 {color: #2c3e50;font-family: 'Arial', sans-serif;}
    [data-testid="stMetricValue"] {font-size: 24px !important;font-weight: bold !important;}
    [data-testid="stMetricLabel"] {font-size: 14px !important;font-weight: normal !important;}
    .dataframe {font-size: 14px !important;}
    .stButton button {font-weight: bold !important;}
    .stSelectbox label {font-size: 16px !important;font-weight: bold !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {height: 50px;font-weight: bold;font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    start="2020-01-01"
    end="2025-10-31"
    @st.cache_data
    def load_data():
        tokens=['KO','AAPL','NVDA','JNJ','^GSPC',"GC=F","CBOE"]
        dy=download_yf(tokens,start,end)
        cryptos=["BTC/USDT","ETH/USDT"]
        dc=download_cx(cryptos,start,end)
        return dy,dc
    load_data()
    token=st.selectbox(label="ACTIVO FINANCIERO: ",options=['KO','AAPL','NVDA','JNJ','^GSPC',"BTC-USDT","ETH-USDT"])
    st.divider()
    st.subheader("Configuracion de Trials")
    n_trials_lgb=st.number_input("Trials LightGBM",min_value=1,max_value=1000,value=5,step=1)
    n_trials_cb=st.number_input("Trials CatBoost",min_value=1,max_value=1000,value=5,step=1)
    n_trials_tx=st.number_input("Trials TimeXer",min_value=1,max_value=1000,value=5,step=1)
    n_trials_moirai=st.number_input("Trials Moirai",min_value=1,max_value=1000,value=5,step=1)
    n_trials_lstm=st.number_input("Trials Meta LSTM",min_value=1,max_value=1000,value=3,step=1)

st.title('TT')
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df=pd.read_csv(os.path.join(_BASE_DIR, "data", "tokens", f"{token}2020-2025.csv"))

tab1,tab2,tab3,tab4,tab5=st.tabs(["Datos & Retornos","Caracteristicas (TA/Macro)","MICFS","Walk Folward","BaseModelsTrain"])

with tab1:
    col1,col2=st.columns(2)
    with col1:
        st.subheader("CLOSE")
        fig_close=go.Figure()
        fig_close.add_trace(go.Scatter(x=list(range(len(df))),y=df["Close"],mode='lines',name='Close',line=dict(color=PROFESSIONAL_STYLE['colors']['primary'], width=2),fill='tozeroy',fillcolor='rgba(31,119,180,0.15)'))
        fig_close.update_layout(template='plotly_white',title=dict(text=f'Precio de Cierre - {token}',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='Periodo', font=dict(size=14)),yaxis_title=dict(text='Precio (USD)', font=dict(size=14)),height=420,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=80, b=70),showlegend=True,legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12)))
        fig_close.update_xaxes(showgrid=True,gridwidth=1,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
        fig_close.update_yaxes(showgrid=True,gridwidth=1,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
        st.plotly_chart(fig_close,use_container_width=True)
        st.dataframe(df.tail(),use_container_width=True)
    with col2:
        st.subheader("LOG RETURN")
        log_close=np.log(df["Close"]/df["Close"].shift(-1)).dropna()
        colors_log=[COLORS['success'] if x>=0 else COLORS['danger'] for x in log_close]
        fig_log=go.Figure()
        fig_log.add_trace(go.Bar(x=list(range(len(log_close))),y=log_close,marker_color=colors_log,name='Log Return',opacity=0.8,marker_line_width=0))
        fig_log.add_hline(y=0,line_dash="solid",line_color=COLORS['black'],line_width=1.5,opacity=0.7)
        fig_log.add_hline(y=log_close.mean(),line_dash="dash",line_color=COLORS['secondary'],line_width=2,annotation_text=f"Media: {log_close.mean():.4f}",annotation_position="right",annotation_font=dict(size=11, color=COLORS['secondary']),annotation_bgcolor='rgba(255, 255, 255, 0.8)')
        fig_log.update_layout(template='plotly_white',title=dict(text='Retornos Logarítmicos',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='Periodo', font=dict(size=14)),yaxis_title=dict(text='Log Return', font=dict(size=14)),height=400,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=80, b=70),showlegend=True,legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12)),bargap=0.05)
        fig_log.update_xaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
        fig_log.update_yaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
        st.plotly_chart(fig_log,use_container_width=True)
        st.subheader("Visualizacion normalizada (solo para grafico)")
        log_close_viz=(log_close-log_close.min())/(log_close.max()-log_close.min())
        fig_norm=go.Figure()
        fig_norm.add_trace(go.Scatter(x=list(range(len(log_close_viz))),y=log_close_viz,mode='lines',name='Normalizado',line=dict(color=PROFESSIONAL_STYLE['colors']['quinary'], width=2)))
        fig_norm.update_layout(template='plotly_white',title=dict(text='Retornos Normalizados [0, 1]',x=0.5,xanchor='center',font=dict(size=16, family='Arial', weight='bold')),xaxis_title=dict(text='Periodo', font=dict(size=13)),yaxis_title=dict(text='Valor Normalizado', font=dict(size=13)),height=350,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=70, b=70),showlegend=True,legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12)))
        fig_norm.update_xaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=12))
        fig_norm.update_yaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=12))
        st.plotly_chart(fig_norm,use_container_width=True)

with tab2:
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Indicadores Tecnicos")
        df_ta=TA(df)
        st.dataframe(df_ta.tail(),use_container_width=True)
    with col2:
        st.subheader("Datos Macroeconomicos")
        df_ma=macroeconomicos(df["Date_final"])
        st.dataframe(df_ma.tail(),use_container_width=True)

with tab3:
    df_ta=df_ta.reset_index(drop=True)
    df_ma=df_ma.reset_index(drop=True)
    st.subheader("DF_final")
    df_final=pd.concat([df_ta,df_ma],axis=1)
    df_final=df_final.iloc[1:]
    log_close=log_close.reset_index(drop=True)
    min_len=min(len(df_final),len(log_close))
    df_final=df_final.iloc[:min_len].reset_index(drop=True)
    log_close=log_close.iloc[:min_len].reset_index(drop=True)
    cols_to_drop=[]
    for col in df_final.columns:
        if df_final[col].max()-df_final[col].min()<1e-8:
            cols_to_drop.append(col)
    st.warning(f"Columna {col} eliminada (valores constantes)")
    df_final=df_final.drop(columns=cols_to_drop)
    df_final=df_final.replace([np.inf,-np.inf],0.0)
    log_close=log_close.replace([np.inf,-np.inf],0.0)
    log_range=log_close.max()-log_close.min()
    if log_range<1e-8:
        st.error("log_close tiene valores constantes")
        st.stop()
    n_total=len(df_final)
    train_size=int(n_total*0.9)
    X_train_raw=df_final.iloc[:train_size].copy()
    X_test_raw=df_final.iloc[train_size:].copy()
    y_train_raw=log_close.iloc[:train_size].copy()
    y_test_raw=log_close.iloc[train_size:].copy()
    scaler_features=MinMaxScaler()
    X_train_scaled=pd.DataFrame(scaler_features.fit_transform(X_train_raw),columns=X_train_raw.columns,index=X_train_raw.index)
    X_test_scaled=pd.DataFrame(scaler_features.transform(X_test_raw),columns=X_test_raw.columns,index=X_test_raw.index)
    scaler_target=MinMaxScaler()
    y_train=pd.Series(scaler_target.fit_transform(y_train_raw.values.reshape(-1,1)).flatten(),index=y_train_raw.index,name='log_close')
    y_test=pd.Series(scaler_target.transform(y_test_raw.values.reshape(-1,1)).flatten(),index=y_test_raw.index,name='log_close')
    st.write(f"Train size: {len(X_train_scaled)}, Test size: {len(X_test_scaled)}")
    st.dataframe(X_train_scaled.tail(),use_container_width=True)
    st.subheader("MIC: top n caracteristicas (solo con train)")
    features,valores_mic=top_k(X_train_scaled,y_train,15)
    df_importance=pd.DataFrame(list(valores_mic.items()),columns=['Feature','Score'])
    df_importance=df_importance.sort_values('Score',ascending=True)
    fig_mic=go.Figure()
    fig_mic.add_trace(go.Bar(y=df_importance['Feature'],x=df_importance['Score'],orientation='h',marker=dict(color=df_importance['Score'],colorscale='Viridis',showscale=True,colorbar=dict(title=dict(text='MIC Score',side='right', font=dict(size=12)),tickfont=dict(size=11))),text=df_importance['Score'].round(3),textposition='outside',textfont=dict(size=11, color=COLORS['black'])))
    fig_mic.update_layout(template='plotly_white',title=dict(text='Maximal Information Coefficient (MIC) - Feature Importance',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='MIC Score', font=dict(size=14)),yaxis_title='',height=550,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=180, r=100, t=90, b=60),xaxis=dict(range=[0, df_importance['Score'].max() * 1.15]),yaxis=dict(tickfont=dict(size=11)))
    fig_mic.update_xaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
    st.plotly_chart(fig_mic,use_container_width=True)
    X_train=X_train_scaled[features].reset_index(drop=True)
    X_test=X_test_scaled[features].reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    st.dataframe(X_train,use_container_width=True)

with tab4:
    k=5
    splitter=wfrw(y_train,k=k,fh_val=30)
    fig_tt=go.Figure()
    fig_tt.add_trace(go.Scatter(x=y_train.index.tolist(),y=y_train.tolist(),name='Train',mode='lines',line=dict(color=PROFESSIONAL_STYLE['colors']['primary'], width=2.5),fill='tozeroy',fillcolor='rgba(31, 119, 180, 0.15)'))
    test_x=[i+len(y_train) for i in y_test.index.tolist()]
    fig_tt.add_trace(go.Scatter(x=test_x,y=y_test.tolist(),name='Test',mode='lines',line=dict(color=PROFESSIONAL_STYLE['colors']['secondary'], width=2.5),fill='tozeroy',fillcolor='rgba(255, 127, 14, 0.15)'))
    fig_tt.add_vline(x=len(y_train),line_dash="dash",line_color=PROFESSIONAL_STYLE['colors']['quaternary'],line_width=2.5,annotation_text="Train/Test Split",annotation_position="top",annotation_font=dict(size=12, color=PROFESSIONAL_STYLE['colors']['quaternary']),annotation_bgcolor='rgba(255, 255, 255, 0.8)')
    fig_tt.update_layout(template='plotly_white',title=dict(text='División Temporal: Train vs Test',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='Índice Temporal', font=dict(size=14)),yaxis_title=dict(text='Valor Normalizado', font=dict(size=14)),height=450,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=90, b=70),legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12),bordercolor='lightgray',borderwidth=1),hovermode='x unified')
    fig_tt.update_xaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
    fig_tt.update_yaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
    st.plotly_chart(fig_tt,use_container_width=True)
    fold_colors_train=['#1f77b4','#2ca02c','#9467bd','#8c564b','#e377c2']
    fold_colors_val=['#ff7f0e','#d62728','#bcbd22','#17becf','#7f7f7f']
    fig_folds=make_subplots(rows=k,cols=1,shared_xaxes=True,vertical_spacing=0.05,subplot_titles=[f'Fold {i+1}' for i in range(k)])
    for i in range(k):
        fig_folds.layout.annotations[i].update(font=dict(size=14, weight='bold'))
    for i,(t_idx,v_idx) in enumerate(wfrw(y_train,k=k,fh_val=30).split(y_train)):
        fig_folds.add_trace(go.Scatter(x=y_train.index[t_idx].tolist(),y=y_train.iloc[t_idx].tolist(),mode='lines',name=f'Train Fold {i+1}',line=dict(color=fold_colors_train[i], width=2),showlegend=(i==0),legendgroup='train'),row=i+1,col=1)
        fig_folds.add_trace(go.Scatter(x=y_train.index[v_idx].tolist(),y=y_train.iloc[v_idx].tolist(),mode='lines',name=f'Val Fold {i+1}',line=dict(color=fold_colors_val[i], width=3),showlegend=(i==0),legendgroup='val'),row=i+1,col=1)
    fig_folds.update_layout(height=900,template='plotly_white',title=dict(text='Walk-Forward Cross-Validation Folds',x=0.5,xanchor='center',font=dict(size=20, family='Arial', weight='bold')),paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=120, b=70),showlegend=True,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='center',x=0.5,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12),bordercolor='lightgray',borderwidth=1))
    for i in range(k):
        fig_folds.update_yaxes(title_text='Valor',row=i+1,col=1,title_font=dict(size=12),gridcolor=PROFESSIONAL_STYLE['colors']['grid'])
    fig_folds.update_xaxes(title_text='Índice Temporal',row=k,col=1,title_font=dict(size=13),gridcolor=PROFESSIONAL_STYLE['colors']['grid'])
    st.plotly_chart(fig_folds,use_container_width=True)

with tab5:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Device: {device}")
    st.subheader("Configuracion de Entrenamiento")
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.metric("LightGBM Trials",n_trials_lgb)
    with col2:
        st.metric("CatBoost Trials",n_trials_cb)
    with col3:
        st.metric("TimeXer Trials",n_trials_tx)
    with col4:
        st.metric("Moirai Trials",n_trials_moirai)
    with col5:
        st.metric("Meta LSTM Trials",n_trials_lstm)
    if st.button("Iniciar Entrenamiento de Modelos",type="primary",use_container_width=True):
        oof_lgb,oof_cb,oof_tx,oof_moirai={},{},{},{}
        st.subheader("Fase 1: Optimizacion de Hiperparametros (solo con train)")
        st.write("Optimizando LightGBM...")
        with st.spinner('Optimizando LGB'):
            study_lgb=optuna.create_study(direction="minimize")
            study_lgb.optimize(lambda trial:objective_global(trial,X_train,y_train,splitter,oof_storage=oof_lgb),n_trials=n_trials_lgb,n_jobs=-1)
            best_params_lgb=study_lgb.best_params
            st.json(best_params_lgb)
            st.write(f"Mejor MAE LGB: {study_lgb.best_value:.4f}")
        st.write("Optimizando CatBoost...")
        with st.spinner('Optimizando CatBoost'):
            study_cb=optuna.create_study(direction="minimize")
            study_cb.optimize(lambda trial:objective_catboost_global(trial,X_train,y_train,splitter,oof_storage=oof_cb),n_trials=n_trials_cb,n_jobs=-1)
            best_params_cb=study_cb.best_params
            st.json(best_params_cb)
            st.write(f"Mejor MAE CatBoost: {study_cb.best_value:.4f}")
        st.write("Optimizando TimeXer...")
        with st.spinner('Optimizando TimeXer'):
            study_tx=optuna.create_study(direction="minimize")
            study_tx.optimize(lambda trial:objective_timexer_global(trial,X_train,y_train,splitter,device=device,seq_len=96,pred_len=30,features='MS',oof_storage=oof_tx),n_trials=n_trials_tx,n_jobs=1)
            best_params_tx=study_tx.best_params
            st.json(best_params_tx)
            st.write(f"Mejor MAE TimeXer: {study_tx.best_value:.4f}")
        st.write("Optimizando Moirai-MoE...")
        with st.spinner('Optimizando Moirai-MoE'):
            preload_moirai_module(model_size='small')
            study_moirai=optuna.create_study(direction="minimize")
            study_moirai.optimize(lambda trial:objective_moirai_moe_global(trial,X_train,y_train,splitter,device=device,pred_len=30,model_size='small',freq='D',use_full_train=True,oof_storage=oof_moirai),n_trials=n_trials_moirai,n_jobs=1)
            best_params_moirai=study_moirai.best_params
            st.json(best_params_moirai)
            st.write(f"Mejor MAE Moirai: {study_moirai.best_value:.4f}")
        st.subheader("Fase 2: Matriz OOF")
        preds_lgb,idx_lgb,=collect_oof_predictions(oof_lgb)
        preds_cb,idx_cb,=collect_oof_predictions(oof_cb)
        preds_tx,idx_tx,=collect_oof_predictions(oof_tx)
        preds_moirai,idx_moirai,_=collect_oof_predictions(oof_moirai)
        st.write(f"Predicciones recolectadas:")
        st.write(f" - LGB: {len(preds_lgb)}")
        st.write(f" - CatBoost: {len(preds_cb)}")
        st.write(f" - TimeXer: {len(preds_tx)}")
        st.write(f" - Moirai: {len(preds_moirai)}")
        oof_df=build_oof_dataframe(oof_lgb,oof_cb,oof_tx,oof_moirai,y_train)
        st.write(f"Matriz OOF (inner join) shape: {oof_df.shape}")
        st.write(f"Columnas: {list(oof_df.columns)}")
        st.dataframe(oof_df.head(30),use_container_width=True)
        def get_column_name(df,keywords):
            for col in df.columns:
                col_lower=col.lower()
                for kw in keywords:
                    if kw in col_lower:
                        return col
            return None
        col_lgb=get_column_name(oof_df,['lgb','lightgbm','light'])
        col_cb=get_column_name(oof_df,['cb','cat','catboost'])
        col_tx=get_column_name(oof_df,['tx','timex','timexer'])
        col_moirai=get_column_name(oof_df,['moirai','moe'])
        col_target=get_column_name(oof_df,['target','y','real'])
        st.write(f"Columnas detectadas: LGB={col_lgb}, CB={col_cb}, TX={col_tx}, Moirai={col_moirai}, Target={col_target}")
        st.subheader("Fase 3: Meta-Modelo LSTM")
        if len(oof_df)<50:
            st.warning(f"Solo hay {len(oof_df)} filas en la matriz OOF. Se recomienda al menos 50.")
        with st.spinner('Optimizando Meta-Modelo LSTM con Optuna...'):
            meta_model,mae_meta,results,best_params_lstm,study_lstm=optimize_lstm_meta(oof_df,device,n_trials=n_trials_lstm)
        if meta_model is not None:
            st.success("Meta-Modelo LSTM entrenado exitosamente")
            st.write("Mejores hiperparametros LSTM:")
            st.json(best_params_lstm)
            col1,col2,col3=st.columns(3)
            with col1:
                st.metric("MAE Meta-Modelo",f"{results['mae']:.4f}")
            with col2:
                st.metric("RMSE Meta-Modelo",f"{results['rmse']:.4f}")
            with col3:
                st.metric("Mejor Epoch",results['best_epoch'])
            st.subheader("Curvas de Entrenamiento Completas")
            fig_loss=go.Figure()
            epochs=list(range(1,len(results['train_losses'])+1))
            fig_loss.add_trace(go.Scatter(x=epochs,y=results['train_losses'],mode='lines',name='Train Loss',line=dict(color=PROFESSIONAL_STYLE['colors']['primary'], width=3)))
            fig_loss.add_trace(go.Scatter(x=epochs,y=results['val_losses'],mode='lines',name='Validation Loss',line=dict(color=PROFESSIONAL_STYLE['colors']['secondary'], width=3)))
            best_epoch=results['best_epoch']
            best_val_loss=results['val_losses'][best_epoch-1] if best_epoch<=len(results['val_losses']) else min(results['val_losses'])
            fig_loss.add_trace(go.Scatter(x=[best_epoch],y=[best_val_loss],mode='markers+text',name=f'Mejor Epoch ({best_epoch})',marker=dict(color=PROFESSIONAL_STYLE['colors']['quaternary'],size=15,symbol='star',line=dict(color='white', width=2)),text=[f'Epoch {best_epoch}'],textposition="top right",textfont=dict(size=12, color=PROFESSIONAL_STYLE['colors']['quaternary'])))
            fig_loss.update_layout(template='plotly_white',title=dict(text='Curvas de Entrenamiento - MSE Loss',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='Epoch', font=dict(size=14)),yaxis_title=dict(text='MSE Loss', font=dict(size=14)),height=500,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=90, b=70),legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12),bordercolor='lightgray',borderwidth=1),hovermode='x unified')
            fig_loss.update_xaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
            fig_loss.update_yaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
            st.plotly_chart(fig_loss,use_container_width=True)
            st.subheader("Pesos Promedio por Modelo Base")
            weights_df=get_average_weights(results['weights'],results['model_names'])
            fig_weights=go.Figure()
            colors_weights=[PROFESSIONAL_STYLE['colors']['primary'],PROFESSIONAL_STYLE['colors']['tertiary'],PROFESSIONAL_STYLE['colors']['quinary'],PROFESSIONAL_STYLE['colors']['secondary']][:len(weights_df)]
            fig_weights.add_trace(go.Bar(x=weights_df['Modelo'],y=weights_df['Peso_Promedio'],marker=dict(color=colors_weights,line=dict(color='rgba(0,0,0,0.3)', width=1.5)),text=weights_df['Peso_Promedio'].round(3),textposition='outside',textfont=dict(size=12, color=COLORS['black'], weight='bold')))
            fig_weights.update_layout(template='plotly_white',title=dict(text='Pesos Promedio Asignados por LSTM a Cada Modelo Base',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='Modelo', font=dict(size=14)),yaxis_title=dict(text='Peso Promedio (Softmax)', font=dict(size=14)),height=450,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=90, b=70),yaxis=dict(range=[0, weights_df['Peso_Promedio'].max() * 1.3]),xaxis=dict(tickfont=dict(size=12)))
            fig_weights.update_xaxes(showgrid=False,title_font=dict(size=13))
            fig_weights.update_yaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
            st.plotly_chart(fig_weights,use_container_width=True)
            st.subheader("Evolucion de Pesos en el Tiempo")
            weights_evolution=pd.DataFrame(results['weights'],columns=results['model_names'])
            weights_evolution['index']=results['valid_indices']
            fig_evo=go.Figure()
            colors_evo=[PROFESSIONAL_STYLE['colors']['primary'],PROFESSIONAL_STYLE['colors']['tertiary'],PROFESSIONAL_STYLE['colors']['quinary'],PROFESSIONAL_STYLE['colors']['secondary']]
            for i,col in enumerate(results['model_names']):
                fig_evo.add_trace(go.Scatter(x=weights_evolution['index'],y=weights_evolution[col],mode='lines',name=col,line=dict(color=colors_evo[i%len(colors_evo)], width=2.5),opacity=0.9))
            fig_evo.update_layout(template='plotly_white',title=dict(text='Pesos Dinámicos (α_t) por Modelo a lo Largo del Tiempo',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='Índice Temporal', font=dict(size=14)),yaxis_title=dict(text='Peso (Softmax)', font=dict(size=14)),height=500,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=90, b=70),legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='center',x=0.5,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12),bordercolor='lightgray',borderwidth=1),hovermode='x unified')
            fig_evo.update_xaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
            fig_evo.update_yaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
            st.plotly_chart(fig_evo,use_container_width=True)
            st.subheader("Predicciones vs Real (Normalizado) - Todos los Modelos")
            valid_indices=results['valid_indices']
            oof_subset=oof_df.loc[oof_df.index.isin(valid_indices)].copy()
            oof_subset=oof_subset.loc[valid_indices]
            pred_df=pd.DataFrame({'Indice':valid_indices,'Real':results['targets'],'Meta_LSTM':results['predictions']})
            if col_lgb:
                pred_df['LGB']=oof_subset[col_lgb].values
            if col_cb:
                pred_df['CatBoost']=oof_subset[col_cb].values
            if col_tx:
                pred_df['TimeXer']=oof_subset[col_tx].values
            if col_moirai:
                pred_df['Moirai']=oof_subset[col_moirai].values
            fig_pred=go.Figure()
            if 'LGB' in pred_df.columns:
                fig_pred.add_trace(go.Scatter(x=pred_df['Indice'],y=pred_df['LGB'],mode='lines',name='LightGBM',line=dict(color=PROFESSIONAL_STYLE['colors']['primary'], width=2, dash='dot'),opacity=0.8))
            if 'CatBoost' in pred_df.columns:
                fig_pred.add_trace(go.Scatter(x=pred_df['Indice'],y=pred_df['CatBoost'],mode='lines',name='CatBoost',line=dict(color=PROFESSIONAL_STYLE['colors']['tertiary'], width=2, dash='dot'),opacity=0.8))
            if 'TimeXer' in pred_df.columns:
                fig_pred.add_trace(go.Scatter(x=pred_df['Indice'],y=pred_df['TimeXer'],mode='lines',name='TimeXer',line=dict(color=PROFESSIONAL_STYLE['colors']['quinary'], width=2, dash='dot'),opacity=0.8))
            if 'Moirai' in pred_df.columns:
                fig_pred.add_trace(go.Scatter(x=pred_df['Indice'],y=pred_df['Moirai'],mode='lines',name='Moirai-MoE',line=dict(color=PROFESSIONAL_STYLE['colors']['senary'], width=2, dash='dot'),opacity=0.8))
            fig_pred.add_trace(go.Scatter(x=pred_df['Indice'],y=pred_df['Meta_LSTM'],mode='lines',name='Meta LSTM',line=dict(color=PROFESSIONAL_STYLE['colors']['quaternary'], width=3.5)))
            fig_pred.add_trace(go.Scatter(x=pred_df['Indice'],y=pred_df['Real'],mode='lines',name='Real',line=dict(color=COLORS['black'], width=4)))
            fig_pred.update_layout(template='plotly_white',title=dict(text='Predicciones de Todos los Modelos vs Valor Real (Normalizado)',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='Índice Temporal', font=dict(size=14)),yaxis_title=dict(text='Valor Normalizado', font=dict(size=14)),height=550,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=90, b=70),legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='center',x=0.5,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12),bordercolor='lightgray',borderwidth=1,itemsizing='constant'),hovermode='x unified')
            fig_pred.update_xaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
            fig_pred.update_yaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],title_font=dict(size=13))
            st.plotly_chart(fig_pred,use_container_width=True)
            st.subheader("Predicciones vs Precio Real (Escala Original) - Todos los Modelos")
            predictions_scaled=np.array(results['predictions']).reshape(-1,1)
            targets_scaled=np.array(results['targets']).reshape(-1,1)
            predictions_log=scaler_target.inverse_transform(predictions_scaled).flatten()
            targets_log=scaler_target.inverse_transform(targets_scaled).flatten()
            lgb_log,cb_log,tx_log,moirai_log=None,None,None,None
            if col_lgb:
                lgb_scaled=oof_subset[col_lgb].values.reshape(-1,1)
                lgb_log=scaler_target.inverse_transform(lgb_scaled).flatten()
            if col_cb:
                cb_scaled=oof_subset[col_cb].values.reshape(-1,1)
                cb_log=scaler_target.inverse_transform(cb_scaled).flatten()
            if col_tx:
                tx_scaled=oof_subset[col_tx].values.reshape(-1,1)
                tx_log=scaler_target.inverse_transform(tx_scaled).flatten()
            if col_moirai:
                moirai_scaled=oof_subset[col_moirai].values.reshape(-1,1)
                moirai_log=scaler_target.inverse_transform(moirai_scaled).flatten()
            close_prices=df['Close'].values
            precio_real,precio_meta=[],[]
            precio_lgb,precio_cb,precio_tx,precio_moirai=[],[],[],[]
            indices_validos=[]
            for i,idx in enumerate(valid_indices):
                if idx>0 and idx<len(close_prices):
                    precio_anterior=close_prices[idx-1]
                    precio_real.append(precio_anterior*np.exp(targets_log[i]))
                    precio_meta.append(precio_anterior*np.exp(predictions_log[i]))
                    if lgb_log is not None:
                        precio_lgb.append(precio_anterior*np.exp(lgb_log[i]))
                    if cb_log is not None:
                        precio_cb.append(precio_anterior*np.exp(cb_log[i]))
                    if tx_log is not None:
                        precio_tx.append(precio_anterior*np.exp(tx_log[i]))
                    if moirai_log is not None:
                        precio_moirai.append(precio_anterior*np.exp(moirai_log[i]))
                    indices_validos.append(idx)
            precio_df=pd.DataFrame({'Indice':indices_validos,'Precio_Real':precio_real,'Meta_LSTM':precio_meta})
            if precio_lgb:
                precio_df['LGB']=precio_lgb
            if precio_cb:
                precio_df['CatBoost']=precio_cb
            if precio_tx:
                precio_df['TimeXer']=precio_tx
            if precio_moirai:
                precio_df['Moirai']=precio_moirai
            fig_precio=go.Figure()
            fig_precio.add_trace(go.Scatter(x=precio_df['Indice'],y=precio_df['Precio_Real'],mode='lines',name='Precio Real',line=dict(color=COLORS['black'],width=3)))
            fig_precio.add_trace(go.Scatter(x=precio_df['Indice'],y=precio_df['Meta_LSTM'],mode='lines',name='Meta LSTM',line=dict(color=COLORS['danger'],width=2.5,dash='dash')))
            if 'LGB' in precio_df.columns:
                fig_precio.add_trace(go.Scatter(x=precio_df['Indice'],y=precio_df['LGB'],mode='lines',name='LightGBM',line=dict(color=COLORS['primary'],width=1.5),opacity=0.7))
            if 'CatBoost' in precio_df.columns:
                fig_precio.add_trace(go.Scatter(x=precio_df['Indice'],y=precio_df['CatBoost'],mode='lines',name='CatBoost',line=dict(color=COLORS['success'],width=1.5),opacity=0.7))
            if 'TimeXer' in precio_df.columns:
                fig_precio.add_trace(go.Scatter(x=precio_df['Indice'],y=precio_df['TimeXer'],mode='lines',name='TimeXer',line=dict(color=COLORS['purple'],width=1.5),opacity=0.7))
            if 'Moirai' in precio_df.columns:
                fig_precio.add_trace(go.Scatter(x=precio_df['Indice'],y=precio_df['Moirai'],mode='lines',name='Moirai-MoE',line=dict(color=COLORS['secondary'],width=1.5),opacity=0.7))
            fig_precio.update_layout(template='plotly_white',title=dict(text='Predicciones en Escala de Precio Original (USD)',x=0.5,xanchor='center',font=dict(size=18, family='Arial', weight='bold')),xaxis_title=dict(text='Índice Temporal', font=dict(size=14)),yaxis_title=dict(text='Precio (USD)', font=dict(size=14)),height=500,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=70, r=40, t=80, b=50),legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='center',x=0.5,bgcolor='rgba(255, 255, 255, 0.8)',font=dict(size=12)),hovermode='x unified')
            st.plotly_chart(fig_precio,use_container_width=True)
            st.subheader("Metricas sobre Precio Real (Escala Original)")
            precio_real_arr=np.array(precio_real)
            precio_meta_arr=np.array(precio_meta)
            def calcular_metricas(y_true,y_pred):
                mse=mean_squared_error(y_true,y_pred)
                rmse=np.sqrt(mse)
                mae=mean_absolute_error(y_true,y_pred)
                r2=r2_score(y_true,y_pred)
                return mse,rmse,mae,r2
            metricas_lista=[]
            mse_meta,rmse_meta,mae_meta,r2_meta=calcular_metricas(precio_real_arr,precio_meta_arr)
            metricas_lista.append({'Modelo':'Meta LSTM','MSE':mse_meta,'RMSE':rmse_meta,'MAE':mae_meta,'R2':r2_meta})
            if precio_lgb:
                mse_lgb,rmse_lgb,mae_lgb,r2_lgb=calcular_metricas(precio_real_arr,np.array(precio_lgb))
                metricas_lista.append({'Modelo':'LightGBM','MSE':mse_lgb,'RMSE':rmse_lgb,'MAE':mae_lgb,'R2':r2_lgb})
            if precio_cb:
                mse_cb,rmse_cb,mae_cb,r2_cb=calcular_metricas(precio_real_arr,np.array(precio_cb))
                metricas_lista.append({'Modelo':'CatBoost','MSE':mse_cb,'RMSE':rmse_cb,'MAE':mae_cb,'R2':r2_cb})
            if precio_tx:
                mse_tx,rmse_tx,mae_tx,r2_tx=calcular_metricas(precio_real_arr,np.array(precio_tx))
                metricas_lista.append({'Modelo':'TimeXer','MSE':mse_tx,'RMSE':rmse_tx,'MAE':mae_tx,'R2':r2_tx})
            if precio_moirai:
                mse_moirai,rmse_moirai,mae_moirai,r2_moirai=calcular_metricas(precio_real_arr,np.array(precio_moirai))
                metricas_lista.append({'Modelo':'Moirai-MoE','MSE':mse_moirai,'RMSE':rmse_moirai,'MAE':mae_moirai,'R2':r2_moirai})
            metricas_df=pd.DataFrame(metricas_lista)
            metricas_df=metricas_df.round(4)
            metricas_df=metricas_df.sort_values(by='MAE',ascending=True).reset_index(drop=True)
            st.dataframe(metricas_df,use_container_width=True)
            mejor_modelo=metricas_df.iloc[0]['Modelo']
            mejor_mae=metricas_df.iloc[0]['MAE']
            mejor_r2=metricas_df.iloc[0]['R2']
            st.write(f"Mejor modelo por MAE: {mejor_modelo} (MAE={mejor_mae}, R2={mejor_r2})")
            st.subheader("Comparacion Visual de Metricas (Precio Real)")
            fig_metricas=make_subplots(rows=2,cols=2,subplot_titles=['MSE por Modelo','RMSE por Modelo','MAE por Modelo','R² por Modelo'],vertical_spacing=0.15,horizontal_spacing=0.12)
            for i in range(4):
                fig_metricas.layout.annotations[i].update(font=dict(size=14, weight='bold'))
            color_mse=px.colors.sequential.Blues[3:3+len(metricas_df)]
            color_rmse=px.colors.sequential.Greens[3:3+len(metricas_df)]
            color_mae=px.colors.sequential.Oranges[3:3+len(metricas_df)]
            color_r2=px.colors.sequential.Purples[3:3+len(metricas_df)]
            fig_metricas.add_trace(go.Bar(x=metricas_df['Modelo'],y=metricas_df['MSE'],marker=dict(color=color_mse,line=dict(color='rgba(0,0,0,0.3)',width=1)),text=metricas_df['MSE'].round(2),textposition='outside',textfont=dict(size=10),showlegend=False),row=1,col=1)
            fig_metricas.add_trace(go.Bar(x=metricas_df['Modelo'],y=metricas_df['RMSE'],marker=dict(color=color_rmse,line=dict(color='rgba(0,0,0,0.3)',width=1)),text=metricas_df['RMSE'].round(2),textposition='outside',textfont=dict(size=10),showlegend=False),row=1,col=2)
            fig_metricas.add_trace(go.Bar(x=metricas_df['Modelo'],y=metricas_df['MAE'],marker=dict(color=color_mae,line=dict(color='rgba(0,0,0,0.3)',width=1)),text=metricas_df['MAE'].round(2),textposition='outside',textfont=dict(size=10),showlegend=False),row=2,col=1)
            fig_metricas.add_trace(go.Bar(x=metricas_df['Modelo'],y=metricas_df['R2'],marker=dict(color=color_r2,line=dict(color='rgba(0,0,0,0.3)',width=1)),text=metricas_df['R2'].round(2),textposition='outside',textfont=dict(size=10),showlegend=False),row=2,col=2)
            fig_metricas.update_layout(template='plotly_white',title=dict(text='Comparación de Métricas entre Modelos',x=0.5,xanchor='center',font=dict(size=20, family='Arial', weight='bold')),height=750,paper_bgcolor='#f6f1e9',plot_bgcolor='#f6f1e9',margin=dict(l=60, r=60, t=120, b=60),showlegend=False)
            for i in range(1,3):
                for j in range(1,3):
                    fig_metricas.update_xaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],row=i,col=j)
                    fig_metricas.update_yaxes(showgrid=True,gridcolor=PROFESSIONAL_STYLE['colors']['grid'],row=i,col=j)
            st.plotly_chart(fig_metricas,use_container_width=True)
            if metricas_df.iloc[0]['Modelo']=='Meta LSTM':
                st.success("El Meta-Modelo LSTM es el mejor modelo en escala de precio real")
            else:
                st.info(f"El mejor modelo en escala de precio real es: {mejor_modelo}")
        else:
            st.error("No hay suficientes datos para entrenar el meta-modelo. Se requieren al menos 20 muestras validas.")
            st.write("Intenta con mas trials o ajusta los parametros de los modelos base.")