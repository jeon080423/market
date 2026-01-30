import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os

# [폰트 설정] 
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

# [설정] 
st.set_page_config(page_title="KOSPI 8대 지표 복합 분석", layout="wide")

# [데이터 수집] 8대 지표 (KOSPI 포함)
@st.cache_data(ttl=3600)
def load_market_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    # 8대 요인 티커 매핑
    tickers = {
        '^KS11': 'KOSPI',        # 1. 국내 지수
        '^SOX': 'SOX',           # 2. 미 반도체
        '^GSPC': 'SP500',        # 3. 미 대형주
        '^VIX': 'VIX',           # 4. 공포지수
        'USDKRW=X': 'Exchange',  # 5. 환율
        '^TNX': 'US10Y',         # 6. 미 장기금리
        '^IRX': 'US2Y',          # 7. 미 단기금리
        '000001.SS': 'China'     # 8. 중국 경기(상하이)
    }
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Close']
    data = data.rename(columns=tickers).ffill().bfill()
    
    # 파생 변수 처리
    data['SOX_lag1'] = data['SOX'].shift(1)  # 시차 반영
    data['Yield_Spread'] = data['US10Y'] - data['US2Y'] # 금리차
    
    return data.dropna()

# [분석] 회귀 분석
def perform_analysis(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    # 8대 복합 요인 구성
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI]
st.title("🛡️ KOSPI 8대 핵심 요인 복합 분석 시스템")
st.markdown("8개 핵심 지표의 상관관계를 통계적으로 검토하여 시장의 위험 수준을 판단합니다.")

try:
    df = load_market_data()
    model, latest_x = perform_analysis(df)
    
    # 요약 메트릭
    st.sidebar.subheader(f"📊 모델 설명력 (R²): {model.rsquared:.2%}")
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("예측 기대수익률", f"{pred:.2%}")
    with col_b:
        status = "위험" if pred < -0.003 else "경계" if pred < 0 else "안정"
        st.subheader(f"종합 진단: {status}")
    with col_c:
        st.write(f"데이터 갱신: {df.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # [8대 지표 시각화] 4x2 레이아웃
    st.subheader("⚠️ 8대 요인 실시간 모니터링")
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    plt.rcParams['axes.unicode_minus'] = False 

    # 1. KOSPI
    axes[0, 0].plot(df['KOSPI'].tail(100), color='black', lw=2)
    axes[0, 0].set_title("1. 코스피 지수 (KOSPI)", fontproperties=fprop, fontsize=12)

    # 2. 환율 (임계점 1,380)
    axes[0, 1].plot(df['Exchange'].tail(100), color='tab:blue')
    axes[0, 1].axhline(y=1380, color='red', linestyle='--')
    axes[0, 1].set_title("2. 원/달러 환율 (위험선: 1,380)", fontproperties=fprop, fontsize=12)

    # 3. 미 반도체 (시차)
    axes[1, 0].plot(df['SOX_lag1'].tail(100), color='tab:green')
    axes[1, 0].set_title("3. 필라델피아 반도체 (SOX Lag)", fontproperties=fprop, fontsize=12)

    # 4. S&P 500
    axes[1, 1].plot(df['SP500'].tail(100), color='tab:cyan')
    axes[1, 1].set_title("4. 미 S&P 500 지수", fontproperties=fprop, fontsize=12)

    # 5. VIX (임계점 20)
    axes[2, 0].plot(df['VIX'].tail(100), color='tab:purple')
    axes[2, 0].axhline(y=20, color='red', linestyle='--')
    axes[2, 0].set_title("5. 공포지수 (VIX)", fontproperties=fprop, fontsize=12)

    # 6. 중국 상하이 지수
    axes[2, 1].plot(df['China'].tail(100), color='tab:red')
    axes[2, 1].set_title("6. 중국 상하이 종합지수", fontproperties=fprop, fontsize=12)

    # 7. 장단기 금리차
    axes[3, 0].plot(df['Yield_Spread'].tail(100), color='tab:orange')
    axes[3, 0].axhline(y=0, color='gray', linestyle='-')
    axes[3, 0].set_title("7. 미 장단기 금리차 (10Y-2Y)", fontproperties=fprop, fontsize=12)

    # 8. 미 10년물 금리
    axes[3, 1].plot(df['US10Y'].tail(100), color='tab:brown')
    axes[3, 1].set_title("8. 미 국채 10년물 금리", fontproperties=fprop, fontsize=12)

    # 폰트 일괄 적용
    for ax in axes.flat:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)

    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("**8대 요인 복합 가이드:** 본 시스템은 위 8가지 지표의 변화율을 다중 회귀 분석하여 코스피에 미치는 순영향을 산출합니다. 환율 1,380원 상회나 금리차의 급격한 변화를 유의 깊게 살펴야 합니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")
