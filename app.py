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
import pandas_datareader.data as web

# [자동 업데이트] 5분 주기
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 정밀 진단 시스템", layout="wide")

# [데이터 수집] 한국 실업률 삭제
@st.cache_data(ttl=300)
def load_all_market_data():
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China',
        'BDRY': 'Freight'
    }
    
    start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    
    # 1. 금융 데이터 수집
    try:
        raw_data = yf.download(list(tickers.keys()), start=start_date, interval='1d', progress=False)
        if isinstance(raw_data.columns, pd.MultiIndex):
            df = raw_data.xs('Close', axis=1, level=0 if 'Close' in raw_data.columns.levels[0] else 1).copy()
        else:
            df = raw_data.copy()
        df = df.rename(columns=tickers)
    except Exception as e:
        st.error(f"금융 데이터 수집 실패: {e}")
        df = pd.DataFrame()

    # 2. 미국 고용 지표만 수집
    us_unemployment = pd.DataFrame()
    try:
        us_unemployment = web.DataReader('ICSA', 'fred', start_date)
    except:
        pass

    if not df.empty:
        df = df.ffill().bfill()
        df['SOX_lag1'] = df['SOX'].shift(1)
        df['Yield_Spread'] = df['US10Y'] - df['US2Y']
        df = df.dropna()
    
    return df, us_unemployment

# [UI 구현]
st.title("🛡️ KOSPI 정밀 진단 및 실물 경제 모니터링")
st.caption(f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    df, us_job = load_all_market_data()
    
    if df.empty or len(df) < 50:
        st.warning("데이터 수집 중입니다. 잠시만 기다려 주세요.")
        st.stop()

    # --- 회귀 분석 로직 ---
    returns_df = np.log(df / df.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    y = returns_df['KOSPI']
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y']
    X = returns_df[features].fillna(0)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    # 예측값 계산
    latest_pct = df[features].pct_change().iloc[-1].replace([np.inf, -np.inf], 0).fillna(0)
    pred_input = np.array([1.0] + [latest_pct[f] for f in features]).reshape(1, -1)
    pred = model.predict(pred_input)[0]

    # 상단 요약 신호
    s_color = "red" if pred < -0.003 else "orange" if pred < 0.001 else "green"
    status_msg = "하락 경계" if s_color=="red" else "중립/관망" if s_color=="orange" else "상승 기대"
    
    st.markdown(f"""<div style="padding:15px; border-radius:10px; border:2px solid {s_color}; text-align:center;">
                <h3 style="color:{s_color}; margin:0;">종합 예측 신호: {status_msg} (내일 예측치: {pred:.2%})</h3>
                </div>""", unsafe_allow_html=True)

    st.divider()

    # 섹션 1: 금융 지표 시각화
    st.subheader("🔍 8대 핵심 금융 지표")
    fig1, axes1 = plt.subplots(2, 4, figsize=(24, 10))
    items = [
        ('KOSPI', 'KOSPI'), ('Exchange', '환율'), ('SOX_lag1', '미 반도체(SOX)'), ('SP500', '미 S&P 500'),
        ('VIX', '공포지수(VIX)'), ('China', '상하이 종합'), ('Yield_Spread', '금리차'), ('US10Y', '미 국채 10Y')
    ]
    for i, (col, title) in enumerate(items):
        ax = axes1[i // 4, i % 4]
        ax.plot(df[col].tail(120), color='#1f77b4', lw=2)
        ax.set_title(title, fontproperties=fprop, fontsize=14)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontproperties(fprop)
    st.pyplot(fig1)

    st.divider()

    # 섹션 2: 실물 경제 지표 (한국 실업률 삭제 후 2열 구성)
    st.subheader("💼 실물 경제 모니터링")
    col1, col2 = st.columns(2)
    
    with col1:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df['Freight'].tail(120), color='green', lw=2)
        ax2.set_title("글로벌 물동량 (BDRY)", fontproperties=fprop, fontsize=15)
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()): label.set_fontproperties(fprop)
        st.pyplot(fig2)
        st.info("글로벌 물동량(BDRY)은 해상 운임 기반 경기 선행 지표입니다.")

    with col2:
        if not us_job.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(us_job.tail(52), color='red', lw=2)
            ax3.set_title("미국 신규 실업수당 청구 (ICSA)", fontproperties=fprop, fontsize=15)
            for label in (ax3.get_xticklabels() + ax3.get_yticklabels()): label.set_fontproperties(fprop)
            st.pyplot(fig3)
            st.info("미국 실업수당 청구 건수는 고용 시장의 건강도를 나타내는 주간 지표입니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")
