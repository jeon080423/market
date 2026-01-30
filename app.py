import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os
from streamlit_autorefresh import st_autorefresh # 추가 설치 필요: pip install streamlit-autorefresh

# [자동 업데이트 설정] 5분(300,000ms)마다 새로고침
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

# [설정] 페이지 설정
st.set_page_config(page_title="KOSPI 8대 지표 실시간 예측", layout="wide")

# [데이터 수집] 실시간 데이터 반영 로직
@st.cache_data(ttl=300) # 5분간 캐시 유지
def load_market_data():
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    
    # 1. 과거 데이터 (최근 1000일 일봉)
    start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    hist_data = yf.download(list(tickers.keys()), start=start_date, interval='1d')['Close']
    
    # 2. 실시간 데이터 (오늘 최신가 강제 결합)
    # yfinance의 period='1d'와 interval='1m'을 사용하여 가장 최근 체결가를 가져옴
    current_data = {}
    for t in tickers.keys():
        tmp = yf.Ticker(t).history(period='1d', interval='1m')
        if not tmp.empty:
            current_data[t] = tmp['Close'].iloc[-1]
        else:
            current_data[t] = hist_data[t].iloc[-1] # 데이터가 없을 경우 마지막 종가 사용

    # 데이터 결합 및 전처리
    data = hist_data.copy()
    data.loc[datetime.now()] = pd.Series(current_data)
    data = data.rename(columns=tickers).ffill().bfill()
    
    data['SOX_lag1'] = data['SOX'].shift(1)
    data['Yield_Spread'] = data['US10Y'] - data['US2Y']
    
    return data.dropna()

# [분석] 회귀 모델링
def perform_analysis(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI 구현]
st.title("🛡️ KOSPI 8대 지표 실시간 예측 대시보드 (5분 자동 갱신)")
st.caption(f"최근 데이터 확인 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    df = load_market_data()
    model, latest_x = perform_analysis(df)
    
    # --- 1. 종합 예측 신호 섹션 ---
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    if pred < -0.003:
        signal_color, signal_icon, signal_text = "red", "🚨", "하락 경계 (Risk Off)"
        strategy_guide = "장중 실시간 데이터가 부정적입니다. 현금 비중을 방어적으로 유지하세요."
    elif pred < 0.001:
        signal_color, signal_icon, signal_text = "orange", "⏳", "중립 (Neutral / Watch)"
        strategy_guide = "현재 지표들이 팽팽하게 맞서고 있습니다. 무리한 장중 대응보다는 관망을 권장합니다."
    else:
        signal_color, signal_icon, signal_text = "green", "🚀", "상승 기대 (Risk On)"
        strategy_guide = "글로벌 지표가 우호적으로 변하고 있습니다. 매수 관점의 접근이 유리합니다."

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border: 2px solid {signal_color}; background-color: rgba(0,0,0,0.05); text-align: center;">
                <h1 style="font-size: 60px; margin: 0;">{signal_icon}</h1>
                <h2 style="color: {signal_color}; margin: 10px 0;">{signal_text}</h2>
                <p style="font-size: 18px;">실시간 예측 수익률: <b>{pred:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.subheader("💡 실시간 투자 행동 가이드")
        st.info(strategy_guide)
        st.write(f"**통계적 신뢰도:** 모델 설명력(R²) **{model.rsquared:.2%}** | 5분 전 데이터와 비교하여 실시간 변화를 분석 중입니다.")

    st.divider()

    # --- 2. 8대 지표 시각화 (2행 4열) ---
    st.subheader("⚠️ 지표별 실시간 변동 및 통계적 위험선")
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    plt.rcParams['axes.unicode_minus'] = False

    plot_items = [
        ('KOSPI', '1. KOSPI (실시간)', 'MA250 - 1σ', '평균 대비 저평가'),
        ('Exchange', '2. 환율 (실시간)', 'MA250 + 1.5σ', '급등 경계'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', 'AI 업황 저점'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '추세 훼손 주의'),
        ('VIX', '5. 공포지수(VIX)', '20.0 (Fixed)', '패닉 임계점'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '중국 경기 침체'),
        ('Yield_Spread', '7. 장단기 금리차', '0.00 (Fixed)', '불황 전조'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '금리 압박')
    ]

    for i, (col, title, threshold_label, desc) in enumerate(plot_items):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(120)
        ma250 = df[col].rolling(window=250).mean().iloc[-1]
        std250 = df[col].rolling(window=250).std().iloc[-1]
        
        if col == 'Exchange': threshold = ma250 + (1.5 * std250)
        elif col in ['VIX', 'Yield_Spread']: threshold = 20.0 if col == 'VIX' else 0.0
        elif col in ['US10Y']: threshold = ma250 + std250
        else: threshold = ma250 - std250
        
        ax.plot(plot_data, color='#1f77b4', lw=2.5)
        ax.axhline(y=threshold, color='crimson', linestyle='--', alpha=0.9, lw=2)
        ax.text(plot_data.index[5], threshold, f" 위험 기준: {threshold_label}", 
                fontproperties=fprop, fontsize=11, color='crimson', 
                verticalalignment='bottom', backgroundcolor='white')

        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold', pad=15)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)
        ax.annotate(f"[{desc}]", xy=(0.5, -0.18), xycoords='axes fraction', 
                    ha='center', fontproperties=fprop, fontsize=12, color='#444444')

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"실시간 데이터 연동 중 오류 발생: {e}")
