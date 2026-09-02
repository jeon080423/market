import yfinance as yf
import pandas as pd
import numpy as np

start_date = "2010-01-01"
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
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

# Too slow to calculate day by day for 10 years, let's vectorize rolling z-score.
def calc_rolling_z_score(s, inverse=False):
    mu = s.rolling(window=252).mean()
    std = s.rolling(window=252).std()
    z = (s - mu) / std
    score = 100 / (1 + np.exp(-z))
    return 100 - score if inverse else score

m = (calc_rolling_z_score(fx_s) + calc_rolling_z_score(b10_s) + calc_rolling_z_score(cp_s, True)) / 3
t = np.clip(100 - (ks_s / ma20 - 0.9) * 500, 0, 100)
s_sp = calc_rolling_z_score(sp_s, True)
s_vx = calc_rolling_z_score(vx_s)
s_em = calc_rolling_z_score(em_s, True)

w_macro, w_tech, w_global, w_fear, w_peri = 0.2, 0.2, 0.2, 0.2, 0.2
tot_w = 1.0

base = (m * w_macro + t * w_tech + s_sp * w_global + s_vx * w_fear + s_em * w_peri) / tot_w
k = 0.5
convex_risk = ((np.exp(k * base / 100) - 1) / (np.exp(k) - 1)) * 100

hist_df = pd.DataFrame({
    'KOSPI': ks_s,
    'Risk': convex_risk
}).dropna()

N_days = 20 # Predict 1 month out
hist_df['KOSPI_Future'] = hist_df['KOSPI'].shift(-N_days)
hist_df['Future_Return'] = (hist_df['KOSPI_Future'] - hist_df['KOSPI']) / hist_df['KOSPI']

high_risk_days = hist_df[hist_df['Risk'] >= 60]
valid_high_risk = high_risk_days.dropna(subset=['Future_Return'])
hits = valid_high_risk[valid_high_risk['Future_Return'] < 0]
hit_rate = len(hits) / len(valid_high_risk) * 100 if len(valid_high_risk) > 0 else 0

print(f"Prediction Window: {N_days} days")
print(f"Total valid days analyzed: {len(hist_df.dropna())}")
print(f"Days with Risk >= 60: {len(valid_high_risk)}")
print(f"Days where KOSPI dropped after {N_days} days: {len(hits)}")
print(f"Hit Rate (Accuracy for High Risk): {hit_rate:.1f}%")

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
