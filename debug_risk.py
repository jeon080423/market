"""
KOSPI 위험 지수 컴포넌트 분석 스크립트
- 각 시그널별 현재 점수와 가중치를 산출
- 최종 위험 지수가 어떻게 41.3이 되었는지 역추적
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 데이터 로드 (app.py load_data와 동일 로직)
end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

tickers = {
    "kospi": "^KS11", "sp500": "^GSPC", "fx": "KRW=X",
    "us10y": "^TNX", "us2y": "^IRX", "vix": "^VIX",
    "copper": "HG=F", "gold": "GC=F", "jpy_krw": "JPYKRW=X",
    "usd_chf": "CHF=X", "vvix": "^VVIX"
}

print("=" * 70)
print("🔍 KOSPI 위험 지수 컴포넌트 분석 보고서")
print("=" * 70)
print(f"분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 데이터 다운로드
all_tickers = list(tickers.values())
data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.ffill()

def get_s(ticker):
    if ticker in data.columns:
        return data[ticker].dropna()
    return pd.Series(dtype=float)

ks_s = get_s("^KS11")
sp_s = get_s("^GSPC").reindex(ks_s.index).ffill()
fx_s = get_s("KRW=X").reindex(ks_s.index).ffill()
b10_s = get_s("^TNX").reindex(ks_s.index).ffill()
vx_s = get_s("^VIX").reindex(ks_s.index).ffill()
cp_s = get_s("HG=F").reindex(ks_s.index).ffill()
gd_s = get_s("GC=F").reindex(ks_s.index).ffill()
jk_s = get_s("JPYKRW=X").reindex(ks_s.index).ffill()
uf_s = get_s("CHF=X").reindex(ks_s.index).ffill()
vv_s = get_s("^VVIX").reindex(ks_s.index).ffill()

ma20 = ks_s.rolling(window=20).mean()

# 2. calculate_score 함수 (app.py와 동일)
def calculate_score(current_series, full_series, inverse=False):
    recent = full_series[full_series.index >= (full_series.index.max() - pd.Timedelta(days=365))]
    mu, std = float(recent.mean()), float(recent.std())
    curr_v = float(current_series.iloc[-1])
    if std == 0: return 50.0
    z = (curr_v - mu) / std
    score = 100 / (1 + np.exp(-z))
    return float(max(0, min(100, (100 - score) if inverse else score)))

# 3. 각 컴포넌트 계산
s_fx = calculate_score(fx_s, fx_s)
s_b10 = calculate_score(b10_s, b10_s)
s_cp = calculate_score(cp_s, cp_s, True)
m_now = (s_fx + s_b10 + s_cp) / 3

ks_val = float(ks_s.iloc[-1])
ma20_val = float(ma20.iloc[-1])
t_now = max(0.0, min(100.0, float(100 - (ks_val / ma20_val - 0.9) * 500)))

s_sp = calculate_score(sp_s, sp_s, True)
s_vx = calculate_score(vx_s, vx_s)

print("━" * 70)
print("📊 [1단계] 개별 시그널 점수 (0=안전, 100=위험)")
print("━" * 70)

# 최신 데이터값 출력
print(f"\n  📈 KOSPI:     {ks_val:,.1f} (20일 이평: {ma20_val:,.1f})")
print(f"  💵 USD/KRW:   {float(fx_s.iloc[-1]):,.1f}")
print(f"  📉 S&P 500:   {float(sp_s.iloc[-1]):,.1f}")
print(f"  🏦 미국 10Y:  {float(b10_s.iloc[-1]):.2f}%")
print(f"  🔴 VIX:       {float(vx_s.iloc[-1]):.2f}")
print(f"  🟠 구리:      {float(cp_s.iloc[-1]):,.1f}")
print()

print(f"  {'시그널':<30} {'현재 점수':>10}   설명")
print(f"  {'-'*60}")
print(f"  🌍 매크로 합산 (환율+금리+구리):  {m_now:>6.1f}점")
print(f"     ├─ 환율 위험 (↑=위험):        {s_fx:>6.1f}점")
print(f"     ├─ 금리 위험 (↑=위험):        {s_b10:>6.1f}점")
print(f"     └─ 구리 위험 (↓=위험):        {s_cp:>6.1f}점")
print(f"  📈 글로벌 시장 (S&P 역방향):     {s_sp:>6.1f}점   (S&P↑=안전, ↓=위험)")
print(f"  😱 시장 공포 (VIX):              {s_vx:>6.1f}점   (VIX↑=위험)")
print(f"  📉 기술적 과매수:                {t_now:>6.1f}점   (KOSPI > MA20 = 위험)")

# 4. ML 가중치 (기본 균등 가중치 사용해서 비교)
w_macro = 0.25
w_tech = 0.25
w_global = 0.25
w_fear = 0.25
total_w = w_macro + w_tech + w_global + w_fear

base_risk = (m_now * w_macro + t_now * w_tech + s_sp * w_global + s_vx * w_fear) / total_w

print(f"\n{'━' * 70}")
print(f"📊 [2단계] 가중 평균 → 기초 위험 지수 (균등 가중 25%씩 가정)")
print(f"{'━' * 70}")
print(f"  ({m_now:.1f} × {w_macro:.2f} + {t_now:.1f} × {w_tech:.2f} + {s_sp:.1f} × {w_global:.2f} + {s_vx:.1f} × {w_fear:.2f}) / {total_w:.2f}")
print(f"  → 기초 위험 지수 (base_risk) = {base_risk:.1f}점")

# 5. 패닉 감지
def get_panic_score(series):
    if series.empty or len(series) < 10: return 0.0
    try:
        recent_5d_mean = series.iloc[-5:].mean()
        past_1y = series[series.index >= (series.index.max() - pd.Timedelta(days=365))]
        mu = past_1y.mean()
        std = past_1y.std()
        if std == 0: return 0.0
        z_score = (recent_5d_mean - mu) / std
        if z_score <= 1.0: return 0.0
        return min(100.0, max(0.0, (z_score - 1.0) * 40))
    except: return 0.0

panic_gold = get_panic_score(gd_s)
panic_jpy = get_panic_score(jk_s)
panic_chf = get_panic_score(uf_s)
panic_vvix = get_panic_score(vv_s)
active_panics = sum([1 for p in [panic_gold, panic_jpy, panic_chf, panic_vvix] if p > 30])
raw_panic_avg = (panic_gold + panic_jpy + panic_chf + panic_vvix) / 4.0
panic_multiplier = 1.0 + (active_panics * 0.5)
final_panic_score = min(100.0, raw_panic_avg * panic_multiplier)

print(f"\n{'━' * 70}")
print(f"🚨 [3단계] 패닉 이벤트 탐지")
print(f"{'━' * 70}")
print(f"  금(Gold):    {panic_gold:>6.1f}점")
print(f"  엔/원(JPY):  {panic_jpy:>6.1f}점")
print(f"  스위스프랑:  {panic_chf:>6.1f}점")
print(f"  VVIX:        {panic_vvix:>6.1f}점")
print(f"  → 활성 패닉 수: {active_panics}개, 최종 패닉 점수: {final_panic_score:.1f}점")

applied_base_risk = max(base_risk, final_panic_score) if final_panic_score > 60 else base_risk
print(f"  → 패닉 오버라이드 적용 여부: {'✅ 예' if final_panic_score > 60 else '❌ 아니오 (60점 미만)'}")
print(f"  → 적용된 위험 점수 (applied_base_risk): {applied_base_risk:.1f}점")

# 6. 비선형 Convexity 변환
k = 0.5
total_risk_index = ((np.exp(k * applied_base_risk / 100) - 1) / (np.exp(k) - 1)) * 100

print(f"\n{'━' * 70}")
print(f"📊 [4단계] 최종 위험 지수 (비선형 Convexity 변환)")
print(f"{'━' * 70}")
print(f"  Convexity 변환 (k={k}): base {applied_base_risk:.1f} → 최종 {total_risk_index:.1f}")
print(f"\n  ★★★ 최종 KOSPI 위험 지수: {total_risk_index:.1f}점 ★★★")

# 7. 1주/1개월 전 비교
print(f"\n{'━' * 70}")
print(f"📊 [5단계] 최근 데이터 변동 추이 (점수 변동 원인 분석)")
print(f"{'━' * 70}")

# 1주 전 데이터
one_week_ago = ks_s.index.max() - pd.Timedelta(days=7)
closest_1w = ks_s.index[ks_s.index <= one_week_ago].max()

print(f"\n  {'지표':<20} {'1주 전':>12} {'현재':>12} {'변동':>12}")
print(f"  {'-'*56}")

for name, series, ticker in [
    ("KOSPI", ks_s, "^KS11"),
    ("S&P 500", sp_s, "^GSPC"),
    ("USD/KRW", fx_s, "KRW=X"),
    ("미국 10Y금리", b10_s, "^TNX"),
    ("VIX", vx_s, "^VIX"),
    ("구리", cp_s, "HG=F"),
]:
    if closest_1w in series.index:
        prev = float(series.loc[closest_1w])
        curr = float(series.iloc[-1])
        chg = curr - prev
        pct = (chg / prev * 100) if prev != 0 else 0
        arrow = "🔺" if chg > 0 else "🔻" if chg < 0 else "⬜"
        print(f"  {name:<20} {prev:>12,.2f} {curr:>12,.2f} {arrow}{chg:>+10.2f} ({pct:+.1f}%)")

print(f"\n{'=' * 70}")
print("분석 완료")
