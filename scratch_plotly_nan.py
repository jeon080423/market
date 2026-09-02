import plotly.graph_objects as go
import numpy as np

fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=50.0,
    number={'suffix': "점", 'font': {'size': 38}},
    delta={'reference': np.nan, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#2ecc71"},
           'suffix': "pt (5일전 대비)", 'font': {'size': 14}},
    gauge={'axis': {'range': [0, 100]}}
))
fig.write_html("test_gauge.html")
