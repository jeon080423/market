import os
import sys
import pandas as pd
import numpy as np

sys.path.append('k:/app/3. market 코스피 전망')
import app

dates = app.ks_s.index[-252:]
for d in dates:
    s_fx = app.get_hist_score_val(app.fx_s, d)
    s_wt = app.get_hist_score_val(app.wt_s, d)
    s_sp = app.get_hist_score_val(app.sp_s, d, True)
    s_vx = app.get_hist_score_val(app.vx_s, d)
    
    t = max(0.0, min(100.0, float(100 - (float(app.ks_s.loc[d]) / float(app.ma20.loc[d]) - 0.9) * 500))) if d in app.ma20.index and pd.notna(app.ma20.loc[d]) and app.ma20.loc[d] != 0 else 50.0
    
    base = (s_fx * app.w_fx + s_wt * app.w_wti + s_sp * app.w_global + s_vx * app.w_fear + t * app.w_tech) / app.total_w
    
    if np.isnan(base):
        print(f"base is NaN at {d}!")
        print(f"s_fx: {s_fx}, s_wt: {s_wt}, s_sp: {s_sp}, s_vx: {s_vx}, t: {t}")
        break

print("Finished checking base.")
