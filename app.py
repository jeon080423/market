import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os

# [자동 업데이트] 5분
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [세션 상태]
if 'spike_logs' not in st.session_state:
    st.session_state.spike_logs = []

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 정밀 진단 시스템 v2.0", layout="wide")

# [데이터 수집 및 보정]
@st.cache_data(ttl=300)
def load_expert_data():
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
    hist_data = yf.download(list(tickers.keys()), start=start_date, interval='1d', progress=False)['Close']
    
    current_prices = {}
    for t in tickers.keys():
        try:
            ticker_obj = yf.Ticker(t)
            rt_data = ticker_obj.history(period='1d', interval='1m')
            if not rt_data.empty:
                val = rt_data['Close'].iloc[-1]
                prev_val = hist_data[t].dropna().iloc[-1]
                if abs((val - prev_val) / prev_val) < 0.1:
                    current_prices[t] = val
                else:
                    current_prices[t] = prev_val
            else:
                current_prices[t] = hist_data[t].dropna().iloc[-1]
        except:
            current_prices[t] = hist_data[t].dropna().iloc[-1]

    df = hist_data.copy()
    today_ts = pd.Timestamp(datetime.now().date())
    if df.index[-1].date() == today_ts.date():
        for t, price in current_prices.items(): df.at[df.index[-1], t] = price
    else:
        new_row = pd.Series(current_prices, name=pd.Timestamp(datetime.now()))
        df = pd.concat([df, pd.DataFrame([new_row])])

    df = df.rename(columns=tickers).ffill().interpolate(method='linear')
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = (df['US10Y'] - df['US2Y']) * 100 # BP 단위
    return df.dropna().tail(250)

# [분석] 가중치 기반 영향도 산출 (합계 100%)
def get_contribution_analysis(df):
    returns = np.log(df / df.shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y']
    y = returns['KOSPI']
    X = returns[features]
    X_scaled = (X - X.mean()) / X.std()
    X_scaled = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_scaled).fit()
    
    # 기여도 계산: 계수의 절대값 비중
    abs_coeffs = np.abs(model.params.drop('const'))
    contributions = (abs_coeffs / abs_coeffs.sum()) * 100
    return model, contributions

# [UI 구현]
st.title("🏛️ KOSPI 8대 지표 정밀 진단 시스템 v2.0")

try:
    df = load_expert_data()
    model, contribution_pct = get_contribution_analysis(df)
    
    # 상단 요약 섹션
    c1, c2 = st.columns([1, 1.2])
    with c1:
        pred_val = model.predict(sm.add_constant((df.pct_change().iloc[-1:] - df.pct_change().mean()) / df.pct_change().std(), has_constant='add'))[0]
        color = "green" if pred_val > 0 else "red"
        st.markdown(f"""
            <div style="padding: 25px; border-radius: 15px; border-left: 10px solid {color}; background-color: #f8f9fa;">
                <h3 style="margin: 0;">오늘의 종합 투자 지수</h3>
                <h1 style="color: {color}; font-size: 50px;">{pred_val:+.2%}</h1>
                <p style="color: #666;">8대 글로벌 거시 지표 가중치 분석 결과</p>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        # 종합 그래프 교체: Donut Chart (기여도 합 100%)
        fig_donut, ax_donut = plt.subplots(figsize=(8, 5))
        wedges, texts, autotexts = ax_donut.pie(
            contribution_pct, labels=contribution_pct.index, autopct='%1.1f%%',
            startangle=140, colors=plt.cm.Pastel1.colors, pctdistance=0.85,
            textprops={'fontproperties': fprop}
        )
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig_donut.gca().add_artist(centre_circle)
        ax_donut.set_title("지표별 KOSPI 영향력 비중 (Total 100%)", fontproperties=fprop, pad=20)
        st.pyplot(fig_donut)

    st.divider()

    # 2행 4열 개별 지표 상세 진단
    fig, axes = plt.subplots(2, 4, figsize=(24, 16))
    plt.subplots_adjust(hspace=0.6, wspace=0.3) # 간격 조정

    items = [
        ('KOSPI', '1. KOSPI 본체', 'MA250 - 1σ', '장기 추세 붕괴'),
        ('Exchange', '2. 원/달러 환율', 'MA250 + 1.5σ', '외인 자금 이탈'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', '기술주 공급망 위기'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '글로벌 심리 위축'),
        ('VIX', '5. 공포지수(VIX)', '25.0', '시장 패닉 진입'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '아시아권 경기 침체'),
        ('Yield_Spread', '7. 장단기 금리차', '0.0', '경제 불황 전조'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '유동성 긴축 압박')
    ]

    for i, (col, title, threshold_label, warning_text) in enumerate(items):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(60)
        curr_val = plot_data.iloc[-1]
        
        # 임계값 계산
        ma250 = df[col].rolling(window=250).mean().iloc[-1]
        std250 = df[col].rolling(window=250).std().iloc[-1]
        
        if col == 'Exchange': threshold = ma250 + (1.5 * std250)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(threshold_label)
        elif col in ['US10Y']: threshold = ma250 + std250
        else: threshold = ma250 - std250

        # 상태 판단 및 전문 설명
        is_danger = curr_val > threshold if col in ['Exchange', 'VIX', 'US10Y'] else curr_val < threshold
        status = "🚨 위기" if is_danger else "✅ 안정"
        status_color = "red" if is_danger else "blue"
        
        # 거리 계산 (전문적 분석)
        dist_pct = abs(curr_val - threshold) / threshold
        analysis_text = f"현재 지표는 위험선과 약 {dist_pct:.1%} 거리로 [{status}] 상태입니다.\n"
        
        if col in ['Exchange', 'VIX', 'US10Y']:
            analysis_text += f"그래프가 빨간선 위로 올라갈 경우\n[{warning_text}]으로 판단합니다."
        else:
            analysis_text += f"그래프가 빨간선 아래로 내려갈 경우\n[{warning_text}]으로 판단합니다."

        # 시각화
        ax.plot(plot_data, color='#1f77b4', lw=3)
        ax.axhline(y=threshold, color='crimson', ls='--', lw=2)
        ax.set_title(title, fontproperties=fprop, fontsize=18, fontweight='bold', pad=15)
        
        # 하단 텍스트 박스 (전문 설명 추가)
        ax.text(0.5, -0.25, analysis_text, transform=ax.transAxes, 
                ha='center', va='center', fontproperties=fprop, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f1f3f5", ec="#ced4da", lw=1))

    st.pyplot(fig)

except Exception as e:
    st.error(f"전문 진단 시스템 가동 중 오류 발생: {e}")
