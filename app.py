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
st.set_page_config(page_title="KOSPI 위험 분석 (업데이트)", layout="wide")

# [데이터 수집]
@st.cache_data(ttl=3600)
def load_market_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Close']
    data = data.rename(columns=tickers).ffill().bfill()
    data['SOX_lag1'] = data['SOX'].shift(1) 
    data['Yield_Spread'] = data['US10Y'] - data['US2Y'] 
    return data.dropna()

# [회귀 분석]
def perform_analysis(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI]
st.title("🛡️ KOSPI 8대 지표 위험 분석 (환율 기준 업데이트)")

try:
    df = load_market_data()
    model, latest_x = perform_analysis(df)
    
    st.sidebar.subheader(f"📊 모델 설명력: {model.rsquared:.2%}")
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("예측 수익률", f"{pred:.2%}")
    with col2:
        status = "위험" if pred < -0.003 else "경계" if pred < 0 else "안정"
        st.subheader(f"시장 진단: {status}")
    with col3:
        st.write(f"최근 데이터 업데이트: {df.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    st.subheader("⚠️ 주요 지표별 위험 모니터링 (최근 데이터 반영)")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.rcParams['axes.unicode_minus'] = False 

    # 1. 환율 - 최근 시장 변동성을 반영하여 1,380원으로 임계점 상향
    axes[0, 0].plot(df['Exchange'].tail(60), color='tab:blue')
    axes[0, 0].axhline(y=1380, color='red', linestyle='--') # 1350 -> 1380 수정
    axes[0, 0].set_title("원/달러 환율 (Risk Threshold: 1,380)", fontproperties=fprop, fontsize=14)
    if fprop: axes[0, 0].legend(["환율", "최근 위험선(1,380)"], prop=fprop)

    # 2. VIX
    axes[0, 1].plot(df['VIX'].tail(60), color='tab:purple')
    axes[0, 1].axhline(y=20, color='red', linestyle='--')
    axes[0, 1].set_title("공포지수 (VIX Index)", fontproperties=fprop, fontsize=14)
    if fprop: axes[0, 1].legend(["VIX", "위험(20)"], prop=fprop)

    # 3. 반도체 지수
    axes[1, 0].plot(df['SOX_lag1'].tail(60), color='tab:green')
    axes[1, 0].set_title("전일 미 반도체지수 (SOX Index)", fontproperties=fprop, fontsize=14)

    # 4. 장단기 금리차
    axes[1, 1].plot(df['Yield_Spread'].tail(60), color='tab:orange')
    axes[1, 1].axhline(y=0, color='black')
    axes[1, 1].set_title("장단기 금리차 (US 10Y-2Y)", fontproperties=fprop, fontsize=14)

    for ax in axes.flat:
        for label in ax.get_xticklabels(): label.set_fontproperties(fprop)
        for label in ax.get_yticklabels(): label.set_fontproperties(fprop)

    plt.tight_layout()
    st.pyplot(fig)
    
    # 설명 텍스트 업데이트
    st.info("**최근 데이터 기반 분석 가이드:** 환율 1,380원 돌파는 외국인 자금 이탈의 강력한 신호로 작동합니다. 과거의 1,350원 기준보다 최근의 환율 상단 뉴노멀을 반영한 1,380~1,400원 선을 실질적인 위험 구간으로 판단합니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")
