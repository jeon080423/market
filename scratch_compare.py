import yfinance as yf
import pandas as pd
import numpy as np

start_date = "2010-01-01"
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
tickers = ['^KS11', '^GSPC', 'KRW=X', '^TNX', 'HG=F', '^VIX', 'EEM']
df = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close'].ffill()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

_ks_s = df['^KS11'].ffill()
_sp_s = df['^GSPC'].ffill()
_fx_s = df['KRW=X'].ffill()
_b10_s = df['^TNX'].ffill()
_cp_s = df['HG=F'].ffill()
_vx_s = df['^VIX'].ffill()
_em_s = df['EEM'].ffill() if 'EEM' in df.columns else _sp_s
_ma20 = _ks_s.rolling(window=20).mean()

def calc_rolling_z_score(s, inverse=False):
    mu = s.rolling(window=252).mean()
    std = s.rolling(window=252).std().replace(0, 1e-9)
    z = (s - mu) / std
    score = 100 / (1 + np.exp(-z))
    return 100 - score if inverse else score

m = (calc_rolling_z_score(_fx_s) + calc_rolling_z_score(_b10_s) + calc_rolling_z_score(_cp_s, True)) / 3.0
t = np.clip(100.0 - (_ks_s / _ma20 - 0.9) * 500.0, 0, 100.0)
s_sp = calc_rolling_z_score(_sp_s, True)
s_vx = calc_rolling_z_score(_vx_s)
s_em = calc_rolling_z_score(_em_s, True)

def get_performance(w_m, w_t, w_g, w_f, w_p, name):
    tot_w = w_m + w_t + w_g + w_f + w_p
    base = (m * w_m + t * w_t + s_sp * w_g + s_vx * w_f + s_em * w_p) / tot_w
    k_val = 0.5
    convex_risk = ((np.exp(k_val * base / 100.0) - 1.0) / (np.exp(k_val) - 1.0)) * 100.0
    
    eval_df = pd.DataFrame({'KOSPI': _ks_s, 'Risk': convex_risk}).dropna()
    
    best_hit_rate = 0
    best_n = 20
    best_signal_count = 0
    
    for n in range(5, 31):
        temp_df = eval_df.copy()
        temp_df['Min_Future_KOSPI'] = temp_df['KOSPI'].rolling(window=n).min().shift(-n)
        temp_df['Max_Drawdown'] = (temp_df['Min_Future_KOSPI'] - temp_df['KOSPI']) / temp_df['KOSPI']
        
        high_risk_days = temp_df[temp_df['Risk'] >= 60]
        valid_high_risk = high_risk_days.dropna(subset=['Max_Drawdown'])
        hits = valid_high_risk[valid_high_risk['Max_Drawdown'] <= -0.05]
        
        hr = len(hits) / len(valid_high_risk) * 100 if len(valid_high_risk) > 0 else 0
        if hr > best_hit_rate:
            best_hit_rate = hr
            best_n = n
            best_signal_count = len(valid_high_risk)
            
    # 최근 1년 상관계수
    last_1y = eval_df.iloc[-252:]
    corr = last_1y['Risk'].corr(last_1y['KOSPI'])
    
    print(f"[{name}]")
    print(f"  - 단순 상관계수(최근 1년): {corr:.3f}")
    print(f"  - 최적 예측 기간: {best_n}일")
    print(f"  - 경보 횟수: {best_signal_count}회")
    print(f"  - 패닉 적중률: {best_hit_rate:.1f}%\n")

# 1. 기존 모델 (EEM 제외, 4개 지표 동일 비중)
get_performance(0.25, 0.25, 0.25, 0.25, 0.0, "과거 모델 (김효진 박사 의견 미반영 / EEM 제외)")

# 2. 현재 모델 (EEM 포함, 5개 지표 동일 비중)
get_performance(0.2, 0.2, 0.2, 0.2, 0.2, "현재 모델 (김효진 박사 의견 반영 / EEM 포함)")
