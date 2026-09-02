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
em_s = df['EEM'].ffill() if 'EEM' in df.columns else sp_s
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
w_macro, w_tech, w_global, w_fear, w_peri = 0.2, 0.2, 0.2, 0.2, 0.2
tot_w = 1.0

for d in dates:
    m = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
    t = max(0.0, min(100.0, float(100 - (float(ks_s.loc[d]) / float(ma20.loc[d]) - 0.9) * 500))) if d in ma20.index and pd.notna(ma20.loc[d]) and ma20.loc[d] != 0 else 50.0
    s_sp = get_hist_score_val(sp_s, d, True)
    s_vx = get_hist_score_val(vx_s, d)
    s_em = get_hist_score_val(em_s, d, True)
    
    base = (m * w_macro + t * w_tech + s_sp * w_global + s_vx * w_fear + s_em * w_peri) / tot_w
    
    k = 0.5
    convex_risk = ((np.exp(k * base / 100) - 1) / (np.exp(k) - 1)) * 100
    base_risks.append(convex_risk)

hist_df = pd.DataFrame({
    'KOSPI': ks_s.loc[dates].values,
    'Risk': base_risks
}, index=dates)

# Define Prediction logic
# If Risk > 60 today, KOSPI should be lower N days from now.
N_days = 10
hist_df['KOSPI_Future'] = hist_df['KOSPI'].shift(-N_days)
hist_df['Future_Return'] = (hist_df['KOSPI_Future'] - hist_df['KOSPI']) / hist_df['KOSPI']

# Hit Condition: Risk > 60 AND Future_Return < 0
high_risk_days = hist_df[hist_df['Risk'] >= 60]
valid_high_risk = high_risk_days.dropna(subset=['Future_Return'])

hits = valid_high_risk[valid_high_risk['Future_Return'] < 0]
hit_rate = len(hits) / len(valid_high_risk) * 100 if len(valid_high_risk) > 0 else 0

print(f"Total valid days analyzed: {len(hist_df.dropna())}")
print(f"Days with Risk >= 60: {len(valid_high_risk)}")
print(f"Days where KOSPI dropped after {N_days} days: {len(hits)}")
print(f"Hit Rate (Accuracy for High Risk): {hit_rate:.1f}%")

# Safe Condition: Risk < 40 AND Future_Return > 0
low_risk_days = hist_df[hist_df['Risk'] <= 40]
valid_low_risk = low_risk_days.dropna(subset=['Future_Return'])
safe_hits = valid_low_risk[valid_low_risk['Future_Return'] > 0]
safe_hit_rate = len(safe_hits) / len(valid_low_risk) * 100 if len(valid_low_risk) > 0 else 0

print(f"Days with Risk <= 40: {len(valid_low_risk)}")
print(f"Days where KOSPI rose after {N_days} days: {len(safe_hits)}")
print(f"Hit Rate (Accuracy for Safe Days): {safe_hit_rate:.1f}%")

total_signals = len(valid_high_risk) + len(valid_low_risk)
total_hits = len(hits) + len(safe_hits)
overall_acc = total_hits / total_signals * 100 if total_signals > 0 else 0
print(f"Overall Accuracy: {overall_acc:.1f}%")
