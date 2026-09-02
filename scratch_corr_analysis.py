import yfinance as yf
import pandas as pd
import numpy as np

# Download data for last 2 years
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
start_date = (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
tickers = ['^KS11', '^GSPC', 'KRW=X', '^TNX', 'HG=F', '^VIX', 'EEM']
df = yf.download(tickers, start=start_date, end=end_date)['Close'].ffill()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

ks_s = df['^KS11'].ffill()
sp_s = df['^GSPC'].ffill()
fx_s = df['KRW=X'].ffill()
b10_s = df['^TNX'].ffill()
cp_s = df['HG=F'].ffill()
vx_s = df['^VIX'].ffill()
em_s = df['EEM'].ffill()
ma20 = ks_s.rolling(window=20).mean()

def get_hist_score_val(series, current_idx, inverse=False):
    sub = series.loc[:current_idx].iloc[-252:]
    if len(sub) < 10: return 50.0
    mu, std = float(sub.mean()), float(sub.std())
    if std == 0: return 50.0
    curr_v = float(series.loc[current_idx])
    z = (curr_v - mu) / std
    score = 100 / (1 + np.exp(-z))
    return (100 - score) if inverse else score

dates = ks_s.index[-252:]
base_risks = []
macro_scores = []
tech_scores = []
glob_scores = []
fear_scores = []
peri_scores = []

w_macro, w_tech, w_global, w_fear, w_peri = 0.2, 0.2, 0.2, 0.2, 0.2
tot_w = 1.0
k = 0.5

for d in dates:
    m = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
    t = max(0.0, min(100.0, float(100 - (float(ks_s.loc[d]) / float(ma20.loc[d]) - 0.9) * 500))) if d in ma20.index and pd.notna(ma20.loc[d]) and ma20.loc[d] != 0 else 50.0
    s_sp = get_hist_score_val(sp_s, d, True)
    s_vx = get_hist_score_val(vx_s, d)
    s_em = get_hist_score_val(em_s, d, True)
    
    base = (m * w_macro + t * w_tech + s_sp * w_global + s_vx * w_fear + s_em * w_peri) / tot_w
    convex_risk = ((np.exp(k * base / 100) - 1) / (np.exp(k) - 1)) * 100
    
    macro_scores.append(m)
    tech_scores.append(t)
    glob_scores.append(s_sp)
    fear_scores.append(s_vx)
    peri_scores.append(s_em)
    base_risks.append(convex_risk)

hist_df = pd.DataFrame({
    'KOSPI': ks_s.loc[dates].values,
    'Risk': base_risks,
    'Macro': macro_scores,
    'Tech': tech_scores,
    'Glob': glob_scores,
    'Fear': fear_scores,
    'Peri': peri_scores
})

print("Correlations with KOSPI Level:")
print(hist_df.corr()['KOSPI'])

print("\nCorrelations with KOSPI Future Returns (5 days ahead):")
hist_df['KOSPI_Ret_5d'] = ks_s.pct_change(5).shift(-5).loc[dates].values
print(hist_df.corr()['KOSPI_Ret_5d'])
