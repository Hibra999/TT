import os
import json
import logging
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

def generate_compare_report(token, cp, gi_v, pr_r, preds_p, mp, MDL, zs, ze, out_dir, 
                             ye_vals=None, meta_raw_preds=None, base_raw_preds=None, 
                             met_fn=None, da_fn=None):
    """
    Genera el reporte HTML de comparación para múltiples meta-modelos y bases.
    """
    # 1. Preparar datos para gráficos
    close_x = list(range(zs, ze))
    close_y = [float(v) for v in cp[zs:ze]]
    
    # Datos Meta para tabla general (mp_metas)
    mp_metas = []
    # Usar mp si ya viene con métricas, o calcularlas aquí
    if mp:
        for row in mp:
            mp_metas.append(dict(row))
    
    # zoom_models: predicciones en USD
    zoom_models = {}
    for km, (cl, nm) in MDL.items():
        if km in preds_p:
            pred = preds_p[km]
            m_valid = ~np.isnan(pred)
            if m_valid.any():
                zoom_models[km] = {
                    'x': [int(x) for x in gi_v[m_valid]],
                    'y': [float(y) for y in pred[m_valid]],
                    'name': nm,
                    'color': cl
                }

    # lr_data: predicciones en escala LogReturn_MinMax (raw model scale)
    lr_data = {}
    lr_idx, lr_real = [], []
    if ye_vals is not None:
        lr_idx = [int(i) for i in range(len(ye_vals))]
        lr_real = [float(v) for v in ye_vals]
        
        # Agregar predicciones meta (raw scaled)
        if meta_raw_preds is not None:
            for km in ['MT', 'NC', 'AB', 'SM', 'XGB_META_EXT']:
                p_raw = meta_raw_preds.get(km)
                if p_raw is not None:
                    m_valid = ~np.isnan(p_raw)
                    if m_valid.any():
                        lr_name = MDL[km][1]
                        lr_data[km] = {
                            'idx': [int(i) for i in np.where(m_valid)[0]],
                            'y': [float(p) for p in p_raw[m_valid]],
                            'name': lr_name,
                            'color': MDL[km][0]
                        }
        
        # Agregar predicciones base (raw scaled)
        if base_raw_preds is not None:
            for km in ['LGB', 'CB', 'TX', 'MO', 'BL', 'LSTM_EXT', 'GRU_EXT', 'ARIMA_EXT', 'RF_EXT', 'TRANS_EXT']:
                p_raw = base_raw_preds.get(km)
                if p_raw is not None:
                    m_valid = ~np.isnan(p_raw)
                    if m_valid.any():
                        lr_data[km] = {
                            'idx': [int(i) for i in np.where(m_valid)[0]],
                            'y': [float(p) for p in p_raw[m_valid]],
                            'name': MDL[km][1],
                            'color': MDL[km][0]
                        }
        lr_data['_real'] = {'idx': lr_idx, 'y': lr_real}

    # 2. Construir HTML (masive string heredada de main_compare.py)
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{token} - Comparativa Ensemble</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#ffffff;color:#111;min-height:100vh;padding:24px}}
  .container{{max-width:1400px;margin:0 auto}}
  h1{{text-align:center;font-size:2.4rem;font-weight:700;color:#000;margin-bottom:8px}}
  .subtitle{{text-align:center;color:#666;font-size:1rem;margin-bottom:32px}}
  .card{{background:#fafafa;border:1px solid #ddd;border-radius:16px;padding:26px;margin-bottom:32px;box-shadow:0 3px 10px rgba(0,0,0,.08)}}
  .card h2{{font-size:1.4rem;font-weight:600;margin-bottom:20px;color:#222;letter-spacing:.5px;border-left:5px solid #000;padding-left:15px}}
  .metrics-table{{width:100%;border-collapse:separate;border-spacing:0;border-radius:12px;overflow:hidden}}
  .metrics-table thead th{{background:#f0f0f0;color:#333;padding:16px 20px;font-weight:600;text-align:left;font-size:.9rem;text-transform:uppercase;letter-spacing:1px;border-bottom:2.5px solid #ccc}}
  .metrics-table tbody td{{padding:14px 20px;border-bottom:1px solid #eee;font-size:.95rem;font-variant-numeric:tabular-nums}}
  .metrics-table tbody tr:hover{{background:#f3f3f3}}
  .model-badge{{display:inline-block;padding:5px 14px;border-radius:20px;font-weight:600;font-size:.85rem;color:#fff}}
  .best-badge{{display:inline-block;margin-left:10px;padding:3px 10px;border-radius:10px;background:#e9ecef;color:#495057;font-size:.7rem;font-weight:700;border:1.5px solid #ced4da}}
  .metrics-grid{{display:grid;grid-template-columns:repeat(auto-fit, minmax(400px, 1fr));gap:24px}}
  @media(max-width:768px){{.metrics-grid{{grid-template-columns:1fr}}}}
  footer{{text-align:center;color:#888;font-size:.85rem;margin-top:60px;padding:30px;border-top:1px solid #eee}}
</style>
</head>
<body>
<div class="container">
  <h1>{token} \u2014 Comparativa de Ensembles</h1>
  <p class="subtitle">Evaluaci\u00f3n de Meta-Learners \u00b7 Escalas: USD Price y LogReturn_MinMax</p>

  <div class="card">
    <h2>M\u00e9tricas Generales de los Meta Learners (Price USD)</h2>
    <table class="metrics-table">
      <thead><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R\u00b2</th></tr></thead>
      <tbody>
"""
    # Filas de la tabla (placeholder inicial, ser\u00e1 mejorado)
    for row in mp_metas:
        html += f"<tr><td><span class='model-badge' style='background:{row['Color']}'>{row['Modelo']}</span></td><td>{row['MSE']:.6f}</td><td>{row['RMSE']:.6f}</td><td>{row['MAE']:.6f}</td><td>{row['R2']:.6f}</td></tr>"
    
    html += """
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Comparativa de M\u00e9tricas (Barras)</h2>
    <div class="metrics-grid">
      <div id="chart-mse"></div><div id="chart-rmse"></div>
      <div id="chart-mae"></div><div id="chart-r2"></div>
      <div id="chart-da"></div>
    </div>
  </div>

  <div class="card">
    <h2>Resultados Visuales (USD Price)</h2>
    <div id="zoom-chart-actual" style="height:500px; margin-bottom:40px;"></div>
    <div id="zoom-chart-no-cb" style="height:500px; margin-bottom:40px;"></div>
    <div id="zoom-chart-ablation" style="height:500px; margin-bottom:40px;"></div>
    <div id="zoom-chart-sota" style="height:500px; margin-bottom:40px;"></div>
    <div id="zoom-chart-parker" style="height:500px"></div>
  </div>

  <div class="card">
    <h2>LogReturn_MinMax (Base Scale Correlation)</h2>
    <div id="lr-actual" style="height:450px; margin-bottom:40px;"></div>
    <div id="lr-no-cb" style="height:450px; margin-bottom:40px;"></div>
    <div id="lr-ablation" style="height:450px; margin-bottom:40px;"></div>
    <div id="lr-sota" style="height:450px; margin-bottom:40px;"></div>
    <div id="lr-parker" style="height:450px"></div>
  </div>
  
  <footer>Generado autom\u00e1ticamente por TT Pipeline \u00b7 main_compare.py</footer>
</div>

<script>
"""
    # Inyectar variables JSON
    html += f"const closeX={json.dumps(close_x)};\nconst closeY={json.dumps(close_y)};\n"
    html += f"const zoomModels={json.dumps(zoom_models)};\n"
    html += f"const metricsData={json.dumps(mp_metas)};\n"
    html += f"const lrData={json.dumps(lr_data)};\n"
    
    html += """const dL={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#333',family:'Segoe UI,system-ui,sans-serif'},xaxis:{gridcolor:'#eee',linecolor:'#ccc'},yaxis:{gridcolor:'#eee',linecolor:'#ccc'},margin:{t:40,r:30,b:50,l:60},legend:{bgcolor:'rgba(0,0,0,0)',font:{size:11},orientation:'h',y:-0.2}};

// Wrapper function para graficar
function drawChart(divId, titleTxt, keys, metaKey) {
    const data = [];
    keys.forEach(k => {
        if(k === metaKey) return;
        if(zoomModels[k] && zoomModels[k].x.length > 0) {
            data.push({
                x: zoomModels[k].x, y: zoomModels[k].y,
                type: 'scatter', mode: 'lines',
                name: zoomModels[k].name,
                line: {color: zoomModels[k].color, width: 1.2, dash: 'dot'},
                marker: {size: 2, color: zoomModels[k].color}
            });
        }
    });
    if(zoomModels[metaKey] && zoomModels[metaKey].x.length > 0) {
        data.push({
            x: zoomModels[metaKey].x, y: zoomModels[metaKey].y,
            type: 'scatter', mode: 'lines+markers',
            name: zoomModels[metaKey].name,
            line: {color: zoomModels[metaKey].color, width: 2.5},
            marker: {size: 4, color: zoomModels[metaKey].color}
        });
    }
    data.push({x:closeX, y:closeY, type:'scatter', mode:'lines', name:'Close (USD)', line:{color:'#000', width:2}});
    Plotly.newPlot(divId, data, {...dL, title: {text: titleTxt, font: {size: 14, color: '#333'}}, xaxis: {...dL.xaxis, title: 'Indice'}, yaxis: {...dL.yaxis, title: 'USD'}, hovermode: 'x unified'}, {responsive: true});
}

drawChart('zoom-chart-actual', 'Precio Close vs Actual', ['LGB','CB','TX','MO','MT'], 'MT');
drawChart('zoom-chart-no-cb', 'Precio Close vs Ours (Sin CatBoost)', ['LGB','TX','MO','NC'], 'NC');
drawChart('zoom-chart-ablation', 'Precio Close vs Ours (Sin TimeXer)', ['LGB','CB','MO','AB'], 'AB');
drawChart('zoom-chart-sota', 'Precio Close vs Yu et al. 2025', ['LGB','CB','BL','SM'], 'SM');
drawChart('zoom-chart-parker', 'Precio Close vs Parker et al. 2025', ['LSTM_EXT','GRU_EXT','ARIMA_EXT','RF_EXT','TRANS_EXT','XGB_META_EXT'], 'XGB_META_EXT');

function drawLR(divId, titleTxt, metaKey, baseKeys) {
    if(!lrData['_real']) return;
    const real = lrData['_real'];
    const data = [];
    baseKeys.forEach(bk => {
        if(lrData[bk]) {
            data.push({x: lrData[bk].idx, y: lrData[bk].y, type:'scatter', mode:'lines', name: lrData[bk].name, line:{color: lrData[bk].color, width:1.2, dash:'dot'}});
        }
    });
    if(lrData[metaKey]) {
        const pred = lrData[metaKey];
        data.push({x: pred.idx, y: pred.y, type:'scatter', mode:'lines+markers', name: pred.name, line:{color: pred.color, width:2.5}, marker:{size:3, color: pred.color}});
    }
    data.push({x: real.idx, y: real.y, type:'scatter', mode:'lines', name:'Real (LogReturn_MinMax)', line:{color:'#000', width:2}});
    Plotly.newPlot(divId, data, {...dL, title: {text: titleTxt, font: {size: 14, color: '#333'}}, xaxis: {...dL.xaxis, title: 'Indice Test'}, yaxis: {...dL.yaxis, title: 'LogReturn_MinMax'}, hovermode: 'x unified'}, {responsive: true});
}

drawLR('lr-actual', 'LogReturn_MinMax: Ensamble Actual', 'MT', ['LGB','CB','TX','MO']);
drawLR('lr-no-cb', 'LogReturn_MinMax: Ours (Sin CatBoost)', 'NC', ['LGB','TX','MO']);
drawLR('lr-ablation', 'LogReturn_MinMax: Ours (Sin TimeXer)', 'AB', ['LGB','CB','MO']);
drawLR('lr-sota', 'LogReturn_MinMax: Yu et al. 2025', 'SM', ['LGB','CB','BL']);
drawLR('lr-parker', 'LogReturn_MinMax: Parker et al. 2025', 'XGB_META_EXT', ['LSTM_EXT','GRU_EXT','ARIMA_EXT','RF_EXT','TRANS_EXT']);

// --- Graficas de Metricas con DA y resaltado ---
const metaOrder = ['Ensamble Actual', 'Ours (Sin CatBoost)', 'Ours (Ensamble Actual Sin TimeXer)', 'Yu et al. [44] 2025', 'Parker et al. 2025'];
const orderedMetrics = metaOrder.map(name => metricsData.find(m => m.Modelo === name)).filter(m => m);

function drawMetricBar(divId, mn, titleTxt, bestFn, fmtFn) {
    const vals = orderedMetrics.map(m => m[mn]);
    const bestVal = bestFn(vals);
    const borderColors = vals.map(v => v === bestVal ? '#000' : 'rgba(0,0,0,0)');
    const borderWidths = vals.map(v => v === bestVal ? 2.5 : 0);
    Plotly.newPlot(divId, [{
        x: orderedMetrics.map(m => m.Modelo),
        y: vals,
        type: 'bar',
        marker: {color: orderedMetrics.map(m => m.Color), opacity: 0.85, line: {color: borderColors, width: borderWidths}},
        text: vals.map(v => fmtFn(v)),
        textposition: 'outside',
        textfont: {color: '#333', size: 11}
    }], {...dL, title: {text: titleTxt, font: {size: 14, color: '#333'}}, xaxis: {...dL.xaxis, tickangle: 15}, showlegend: false, margin: {t: 50, r: 20, b: 150, l: 60}}, {responsive: true, displayModeBar: false});
}

const minFn = arr => Math.min(...arr);
const maxFn = arr => Math.max(...arr);
const fmt4 = v => v.toFixed(4);
const fmtDA = v => v.toFixed(2) + '%';

drawMetricBar('chart-mse', 'MSE', 'MSE', minFn, fmt4);
drawMetricBar('chart-rmse', 'RMSE', 'RMSE', minFn, fmt4);
drawMetricBar('chart-mae', 'MAE', 'MAE', minFn, fmt4);
drawMetricBar('chart-r2', 'R2', 'R2', maxFn, fmt4);
drawMetricBar('chart-da', 'DA', 'DA (%)', maxFn, fmtDA);
</script></body></html>"""

    safe_token = token.replace('/', '-').replace('^', '').replace('=', '-')
    out_html = os.path.join(out_dir, f'report_compare_{safe_token}.html')
    with open(out_html, 'w', encoding='utf-8') as fh: fh.write(html)
    return out_html

def enhance_report_with_metrics(html_path, preds_p, MDL, pr_r, met_fn, da_fn):
    """
    Usa BeautifulSoup para inyectar DA, mejores badges y orden correcto en el reporte.
    """
    if not os.path.exists(html_path):
        return
        
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Buscar la tabla de métricas
    meta_table = None
    for h2 in soup.find_all('h2'):
        if 'Metricas Generales' in h2.text:
            meta_table = h2.find_next('table', class_='metrics-table')
            break

    if meta_table:
        # 1. Reconstruir header
        thead_tr = meta_table.find('thead').find('tr')
        thead_tr.clear()
        for col_name in ['Modelo', 'MSE', 'RMSE', 'MAE', 'R2', 'DA (%)']:
            th = soup.new_tag('th')
            th.string = col_name
            thead_tr.append(th)

        # 2. Obtener datos de los meta-modelos principales para la tabla
        meta_keys = ['MT', 'AB', 'SM', 'XGB_META_EXT'] 
        meta_rows_data = []
        for km in meta_keys:
            if km in preds_p:
                v = preds_p[km]
                m = ~np.isnan(v)
                if m.any():
                    metrics = met_fn(pr_r[m], v[m])
                    da_val = da_fn(pr_r[m], v[m])
                    meta_rows_data.append({
                        'name': MDL[km][1], 'color': MDL[km][0],
                        'MSE': metrics['MSE'], 'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE'], 'R2': metrics['R2'],
                        'DA': round(da_val, 2)
                    })

        # 3. Mejores valores
        best_vals = {}
        if meta_rows_data:
            best_vals['MSE'] = min(r['MSE'] for r in meta_rows_data)
            best_vals['RMSE'] = min(r['RMSE'] for r in meta_rows_data)
            best_vals['MAE'] = min(r['MAE'] for r in meta_rows_data)
            best_vals['R2'] = max(r['R2'] for r in meta_rows_data)
            best_vals['DA'] = max(r['DA'] for r in meta_rows_data)

        # 4. Reconstruir tbody
        tbody = meta_table.find('tbody')
        tbody.clear()
        for rd in meta_rows_data:
            tr = soup.new_tag('tr')
            # Modelo
            td_m = soup.new_tag('td')
            badge = soup.new_tag('span', attrs={'class': 'model-badge', 'style': f'background:{rd["color"]}'})
            badge.string = rd['name']
            td_m.append(badge)
            tr.append(td_m)
            # MSE, RMSE, MAE, R2
            for mn in ['MSE', 'RMSE', 'MAE', 'R2']:
                td = soup.new_tag('td')
                td.string = f'{rd[mn]:.6f}'
                if mn in best_vals and rd[mn] == best_vals[mn]:
                    td['style'] = 'background:#d4edda'
                tr.append(td)
            # DA
            td_da = soup.new_tag('td')
            td_da.string = f'{rd["DA"]:.2f}%'
            if 'DA' in best_vals and rd['DA'] == best_vals['DA']:
                td_da['style'] = 'background:#d4edda'
            tr.append(td_da)
            tbody.append(tr)

    # 5. Actualizar metricsData en el script JS para charts de barras
    script_tag = soup.find('script', string=lambda s: s and 'metricsData' in s)
    if script_tag and meta_rows_data:
        new_metrics_json = json.dumps([{
            'Modelo': rd['name'], 'Color': rd['color'],
            'MSE': rd['MSE'], 'RMSE': rd['RMSE'],
            'MAE': rd['MAE'], 'R2': rd['R2'], 'DA': rd['DA']
        } for rd in meta_rows_data])
        script_tag.string = re.sub(r'const metricsData=.*?;', f'const metricsData={new_metrics_json};', script_tag.string, count=1)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))

def inject_dm_results(html_path, bloques):
    """
    Inyecta la tabla de resultados Diebold-Mariano en el reporte HTML.
    """
    if not os.path.exists(html_path) or not bloques:
        return
        
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Eliminar DM existente
    for h2 in soup.find_all('h2'):
        if 'Diebold-Mariano' in h2.text:
            dm_card = h2.find_parent('div', class_='card')
            if dm_card: dm_card.decompose()
            break

    container = soup.find('div', class_='container')
    if container:
        dm_html = """
        <div class="card">
            <h2>Prueba de Diebold-Mariano (Escala USD: Errores Cuadr\u00e1ticos)</h2>
            <p style="color:#666; font-size:0.9rem; margin-bottom:12px;">H0: Los modelos tienen la misma precisi\u00f3n predictiva. p-valor &lt; 0.05 indica diferencia significativa.</p>
            <table class="metrics-table">
                <thead><tr>
                    <th>Modelo A</th><th>Modelo B</th>
                    <th>Estad\u00edstico DM</th><th>p-valor</th>
                    <th>Significativo</th><th>Mejor Modelo</th>
                </tr></thead>
                <tbody>"""

        for b_name, b_rows in bloques:
            if not b_rows: continue
            dm_html += f'<tr style="background:#f0f0f0"><td colspan="6" style="font-weight:600;font-size:0.9rem;padding:10px 18px;color:#333">{b_name}</td></tr>'
            for r in b_rows:
                st = f'<strong>{r["stat"]:.4f}</strong>' if r['sig'] else f'{r["stat"]:.4f}'
                pv = f'<strong>{r["p_value"]:.4f}</strong>' if r['sig'] else f'{r["p_value"]:.4f}'
                better_cell = f'<td><strong>{r["better"]}</strong></td>' if r['sig'] else '<td style="color:#999">Sin diferencia significativa</td>'
                dm_html += f'<tr><td>{r["model_a"]}</td><td>{r["model_b"]}</td><td>{st}</td><td>{pv}</td><td>{"S\u00cd" if r["sig"] else "No"}</td>{better_cell}</tr>'

        dm_html += "</tbody></table></div>"
        container.append(BeautifulSoup(dm_html, 'html.parser'))

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))
