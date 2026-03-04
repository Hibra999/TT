import pickle
import numpy as np

# Let's inspect report_compare.html
import json

with open("report_compare.html", encoding="utf-8") as f:
    text = f.read()

start = text.find("const zoomModels=") + 17
end = text.find(";\nconst metricsData", start)
try:
    data = json.loads(text[start:end])
    for k, v in data.items():
        print(f"Model {k}: {len(v['x'])} points, Valid range: {v['x'][0]} to {v['x'][-1]}")
    
    print("\nTotal range available in chart:")
    zoomx_start = text.find("const zoomX=") + 12
    zoomx_end = text.find(";\nconst zoomClose", zoomx_start)
    zoomx = json.loads(text[zoomx_start:zoomx_end])
    print(f"zoomX len: {len(zoomx)}, range: {zoomx[0]} to {zoomx[-1]}")
except Exception as e:
    print("Error parsing json:", e)
