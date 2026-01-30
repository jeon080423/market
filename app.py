import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os

# [설정] 페이지 설정
st.set_page_config(page_title="KOSPI 위험 지수 분석", layout="wide")

# [폰트 설정] 폰트 객체를 생성하여 모든 텍스트 요소에 직접 주입
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        # 폰트 매니저에 등록하고 객체 반환
        return fm.FontProperties(fname=font_path)
    else:
        # 파일이 없을 경우 경고창을 띄우고 None 반환
        st.error(f"폰트 파일을 찾을 수 없습니다: {font_path}")
        return None

# 폰트 속성 객체 생성
fprop = get_korean_font()

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

# [메인 화면]
st.title("🛡️ KOSPI 8대 지표 위험 분석 시스템")

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
        st.write(f"업데이트: {df.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # [그래프 섹션] 모든 텍스트 요소에 fprop 주입
    st.subheader("⚠️ 주요 지표별 위험 임계점")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 깨짐 방지

    # 1. 환율
    axes[0, 0].plot(df['Exchange'].tail(60), color='tab:blue')
    axes[0, 0].axhline(y=1350, color='red', linestyle='--')
    axes[0, 0].set_title("원/달러 환율 (USD/KRW)", fontproperties=fprop, fontsize=14)
    axes[0, 0].set_xlabel("날짜", fontproperties=fprop)
    axes[0, 0].set_ylabel("가격", fontproperties=fprop)
    if fprop: axes[0, 0].legend(["환율", "위험(1350)"], prop=fprop)

    # 2. VIX
    axes[0, 1].plot(df['VIX'].tail(60), color='tab:purple')
    axes[0, 1].axhline(y=20, color='red', linestyle='--')
    axes[0, 1].set_title("공포지수 (VIX)", fontproperties=fprop, fontsize=14)
    if fprop: axes[0, 1].legend(["VIX", "위험(20)"], prop=fprop)

    # 3. 반도체 지수
    axes[1, 0].plot(df['SOX_lag1'].tail(60), color='tab:green')
    axes[1, 0].set_title("전일 미 반도체지수 (SOX)", fontproperties=fprop, fontsize=14)

    # 4. 장단기 금리차
    axes[1, 1].plot(df['Yield_Spread'].tail(60), color='tab:orange')
    axes[1, 1].axhline(y=0, color='black')
    axes[1, 1].set_title("장단기 금리차 (10Y-2Y)", fontproperties=fprop, fontsize=14)

    # X축 눈금(Tick) 폰트 처리
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_fontproperties(fprop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(fprop)

    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("**분석 가이드:** 환율 1350원과 VIX 20은 지수 하락의 임계점입니다. SOX 지수는 국내 증시 방향성의 핵심 선행 지표입니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")
