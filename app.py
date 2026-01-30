import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# [환경설정] 페이지 레이아웃 및 타임존
st.set_page_config(page_title="KOSPI 위험 지수 분석", layout="wide")

# [데이터 수집] FinanceDataReader 단일화 (설치 에러 최소화)
@st.cache_data(ttl=3600)
def load_data():
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # 8대 지표 선정 (KOSPI, SOX, SP500, VIX, 환율, 10년물금리, 2년물금리, 상하이종합)
    tickers = {
        'KS11': 'KOSPI',         # 코스피
        'SOX': 'SOX',            # 필라델피아 반도체
        'US500': 'SP500',        # S&P 500
        'VIX': 'VIX',            # 공포지수
        'USD/KRW': 'Exchange',   # 원/달러 환율
        'US10YT=X': 'US10Y',     # 미 10년물 금리
        'US2YT=X': 'US2Y',       # 미 2년물 금리
        'SSEC': 'China'          # 상하이 종합 (중국 실물 대용)
    }
    
    combined = []
    for t, name in tickers.items():
        try:
            df = fdr.DataReader(t, start_date, end_date)['Close']
            combined.append(df.rename(name))
        except:
            continue
            
    all_data = pd.concat(combined, axis=1).ffill().bfill()
    
    # 선행 지표 변환: 반도체 지수의 시차(t-1) 적용
    all_data['SOX_lag1'] = all_data['SOX'].shift(1)
    # 장단기 금리차 생성
    all_data['Spread'] = all_data['US10Y'] - all_data['US2Y']
    
    return all_data.dropna()

# [회귀 분석] 8대 지표 기반 위험도 산출
def analyze_market(df):
    # 수익률 기반 분석
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    # 8대 독립변수 (Foreign_NetBuy는 FDR에서 지원 안되므로 실물 지표로 대체 보완)
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [메인 화면]
st.title("🛡️ KOSPI 8대 핵심 지표 위험 분석")
st.markdown("글로벌 주요 지표를 통합 분석하여 코스피의 위험 수준을 진단합니다.")

try:
    df = load_data()
    model, latest_x = analyze_market(df)
    
    # 1. 상단 요약 정보
    st.sidebar.subheader(f"모델 설명력: {model.rsquared:.2%}")
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("예측 수익률", f"{pred:.2%}")
    with c2:
        risk_status = "위험" if pred < -0.003 else "주의" if pred < 0 else "안정"
        st.subheader(f"시장 진단: {risk_status}")
    with c3:
        st.write(f"최종 업데이트: {df.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # 2. 위험 임계점 시각화 (그래프 설명 포함)
    st.subheader("⚠️ 주요 지표별 위험 모니터링")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    
    # 환율 (위험선: 1350)
    axes[0, 0].plot(df['Exchange'].tail(60))
    axes[0, 0].axhline(y=1350, color='r', linestyle='--', label='위험(1350)')
    axes[0, 0].set_title("환율 (USD/KRW)")
    axes[0, 0].legend()
    
    # VIX (위험선: 20)
    axes[0, 1].plot(df['VIX'].tail(60), color='purple')
    axes[0, 1].axhline(y=20, color='r', linestyle='--', label='위험(20)')
    axes[0, 1].set_title("공포지수 (VIX)")
    axes[0, 1].legend()
    
    # 반도체 시차 데이터
    axes[1, 0].plot(df['SOX_lag1'].tail(60), color='green')
    axes[1, 0].set_title("전일 미 반도체지수(SOX)")
    
    # 장단기 금리차
    axes[1, 1].plot(df['Spread'].tail(60), color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='-')
    axes[1, 1].set_title("장단기 금리차 (10Y-2Y)")

    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("**분석 가이드:** 환율이 1350원 위로 치솟거나 VIX가 20을 넘으면 코스피 하락 위험이 매우 큽니다. 반도체 지수는 익일 코스피 시가 결정에 가장 큰 영향을 줍니다.")

except Exception as e:
    st.error(f"데이터 분석 중 오류 발생: {e}")
