import json,os,numpy as np

def generate_html_report(token,cp,gi_v,pr_r,preds_p,mp,MDL,zs,ze,out_dir):
    """Genera report.html con Zoom Test interactivo y métricas."""
    zoom_x=list(range(zs,ze));zoom_close=[float(v) for v in cp[zs:ze]]
    zoom_models={}
    for km,(cl,nm) in MDL.items():
        v=preds_p[km];m=~np.isnan(v)
        if m.any():zoom_models[nm]={'x':[int(x) for x in gi_v[m]],'y':[float(y) for y in v[m]],'color':cl}
    # Add color to metrics for the bar charts
    mp_c=[]
    for row in mp:
        r=dict(row)
        for km,(cl,nm) in MDL.items():
            if nm==r['Modelo']:r['Color']=cl;break
        mp_c.append(r)
    html=f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{token} - Reporte Interactivo</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#e0e0e0;min-height:100vh;padding:24px}}
  .container{{max-width:1400px;margin:0 auto}}
  h1{{text-align:center;font-size:2.2rem;font-weight:700;background:linear-gradient(90deg,#00d2ff,#7b2ff7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}}
  .subtitle{{text-align:center;color:#aaa;font-size:.95rem;margin-bottom:32px}}
  .card{{background:rgba(255,255,255,.06);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:24px;margin-bottom:28px;box-shadow:0 8px 32px rgba(0,0,0,.3)}}
  .card h2{{font-size:1.3rem;font-weight:600;margin-bottom:16px;color:#00d2ff;letter-spacing:.5px}}
  .metrics-table{{width:100%;border-collapse:separate;border-spacing:0;border-radius:12px;overflow:hidden}}
  .metrics-table thead th{{background:rgba(0,210,255,.15);color:#00d2ff;padding:14px 18px;font-weight:600;text-align:left;font-size:.9rem;text-transform:uppercase;letter-spacing:1px}}
  .metrics-table tbody td{{padding:12px 18px;border-bottom:1px solid rgba(255,255,255,.06);font-size:.95rem;font-variant-numeric:tabular-nums}}
  .metrics-table tbody tr:hover{{background:rgba(255,255,255,.04)}}
  .metrics-table tbody tr:last-child td{{border-bottom:none}}
  .model-badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:600;font-size:.85rem;color:#fff}}
  .best-badge{{display:inline-block;margin-left:8px;padding:2px 8px;border-radius:10px;background:rgba(0,210,255,.2);color:#00d2ff;font-size:.7rem;font-weight:600}}
  .metrics-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  @media(max-width:768px){{.metrics-grid{{grid-template-columns:1fr}}}}
  footer{{text-align:center;color:#555;font-size:.8rem;margin-top:40px;padding:20px}}
</style>
</head>
<body>
<div class="container">
  <h1>\U0001f4ca {token} \u2014 Reporte Ensemble</h1>
  <p class="subtitle">Zoom sobre zona de test \u00b7 M\u00e9tricas de predicci\u00f3n en escala de precio (USD)</p>
  <div class="card"><h2>\U0001f50d Zoom \u2014 Zona Test</h2><div id="zoom-chart"></div></div>
  <div class="card"><h2>\U0001f4cb M\u00e9tricas por Modelo</h2>
    <table class="metrics-table"><thead><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R\u00b2</th></tr></thead><tbody>
"""
    for i,m_ in enumerate(mp_c):
        best='<span class="best-badge">BEST</span>' if i==0 else ''
        html+=f'<tr><td><span class="model-badge" style="background:{m_["Color"]}">{m_["Modelo"]}</span>{best}</td><td>{m_["MSE"]:.6f}</td><td>{m_["RMSE"]:.6f}</td><td>{m_["MAE"]:.6f}</td><td>{m_["R2"]:.6f}</td></tr>\n'
    html+="""    </tbody></table></div>
  <div class="card"><h2>\U0001f4ca Comparaci\u00f3n de M\u00e9tricas</h2>
    <div class="metrics-grid">
      <div id="chart-mse"></div><div id="chart-rmse"></div>
      <div id="chart-mae"></div><div id="chart-r2"></div>
    </div>
  </div>
  <footer>Generado autom\u00e1ticamente \u00b7 main2.py</footer>
</div>
<script>
"""
    html+=f"const zoomX={json.dumps(zoom_x)};\nconst zoomClose={json.dumps(zoom_close)};\n"
    html+=f"const zoomModels={json.dumps(zoom_models)};\nconst metricsData={json.dumps(mp_c)};\n"
    html+="""const dL={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#ccc',family:'Segoe UI,system-ui,sans-serif'},xaxis:{gridcolor:'rgba(255,255,255,0.06)'},yaxis:{gridcolor:'rgba(255,255,255,0.06)'},margin:{t:40,r:30,b:50,l:60},legend:{bgcolor:'rgba(0,0,0,0)',font:{size:11}}};
const zt=[{x:zoomX,y:zoomClose,type:'scatter',mode:'lines',name:'Close (USD)',line:{color:'#fff',width:2}}];
for(const[n,d] of Object.entries(zoomModels))zt.push({x:d.x,y:d.y,type:'scatter',mode:'lines+markers',name:n,line:{color:d.color,width:1.5},marker:{size:4,color:d.color}});
Plotly.newPlot('zoom-chart',zt,{...dL,title:{text:'Precio Close + Predicciones (Zona Test)',font:{size:14,color:'#aaa'}},xaxis:{...dL.xaxis,title:'\u00cdndice temporal'},yaxis:{...dL.yaxis,title:'USD'},hovermode:'x unified'},{responsive:true});
['MSE','RMSE','MAE','R2'].forEach((mn,i)=>{const ids=['chart-mse','chart-rmse','chart-mae','chart-r2'];const titles=['MSE','RMSE','MAE','R\u00b2'];
Plotly.newPlot(ids[i],[{x:metricsData.map(m=>m.Modelo),y:metricsData.map(m=>m[mn]),type:'bar',marker:{color:metricsData.map(m=>m.Color),opacity:.85},text:metricsData.map(m=>m[mn].toFixed(4)),textposition:'outside',textfont:{color:'#ccc',size:11}}],{...dL,title:{text:titles[i],font:{size:14,color:'#aaa'}},xaxis:{...dL.xaxis,tickangle:-20},showlegend:false,margin:{t:50,r:20,b:60,l:60}},{responsive:true,displayModeBar:false});});
</script></body></html>"""
    out_html=os.path.join(out_dir,'report.html')
    with open(out_html,'w',encoding='utf-8') as fh:fh.write(html)
    print(f'Listo: {out_html}')
