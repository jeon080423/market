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
import pandas_datareader.data as web # FRED 데이터 수집용

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

st.set_page_config(page_title="KOSPI 8대 지표 및 고용 지표 진단", layout="wide")

# [데이터 수집]
@st.cache_data(ttl=300)
def load_all_market_data():
    # 1. 기존 8대 지표 및 물동량(BDRY)
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China',
        'BDRY': 'Freight' # 글로벌 물동량 지표 (ETF)
    }
    
    start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    data = yf.download(list(tickers.keys()), start=start_date, interval='1d', progress=False)['Close']
    
    # 2. 고용 지표 (FRED 연동)
    try:
        # 미국 주간 신규 실업수당 청구 건수 (ICSA)
        us_unemployment = web.DataReader('ICSA', 'fred', start_date)
        # 한국 실업수당 청구 건수 (프록시 데이터 또는 관련 ETF 역산 - 여기서는 가독성을 위해 FRED의 한국 관련 고용지표 활용)
        kr_unemployment = web.DataReader('IDXKRWHCOYDSMEI', 'fred', start_date) # KR Unemployment Proxy
    except:
        us_unemployment = pd.DataFrame()
        kr_unemployment = pd.DataFrame()

    data = data.rename(columns=tickers).ffill().bfill()
    data['SOX_lag1'] = data['SOX'].shift(1)
    data['Yield_Spread'] = data['US10Y'] - data['US2Y']
    
    return data.dropna(), us_unemployment, kr_unemployment

# [UI 구현]
st.title("🛡️ KOSPI 정밀 진단 및 글로벌 고용 지표")
st.caption(f"최종 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    df, us_job, kr_job = load_all_market_data()
    
    # 상단 회귀 분석 섹션 (기존 로직 유지)
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    pred = model.predict(X.iloc[-1].values.reshape(1, -1))[0]

    # 신호 요약 카드
    s_color = "red" if pred < -0.003 else "orange" if pred < 0.001 else "green"
    st.markdown(f"""<div style="padding:15px; border-radius:10px; border:2px solid {s_color}; text-align:center;">
                <h3>종합 예측 신호: {"하락 경계" if s_color=="red" else "중립" if s_color=="orange" else "상승 기대"} (수익률 {pred:.2%})</h3>
                </div>""", unsafe_allow_html=True)

    st.divider()

    # 섹션 1: 핵심 8대 금융 지표 (2행 4열)
    st.subheader("🔍 8대 핵심 금융 지표")
    fig1, axes1 = plt.subplots(2, 4, figsize=(24, 10))
    items = [
        ('KOSPI', 'KOSPI', 'MA250-1σ'), ('Exchange', '환율', 'MA250+1.5σ'),
        ('SOX_lag1', '미 반도체(SOX)', 'MA250-1σ'), ('SP500', '미 S&P 500', 'MA250-0.5σ'),
        ('VIX', '공포지수(VIX)', '20.0'), ('China', '상하이 종합', 'MA250-1.5σ'),
        ('Yield_Spread', '금리차', '0.00'), ('US10Y', '미 국채 10Y', 'MA250+1σ')
    ]
    for i, (col, title, threshold_lab) in enumerate(items):
        ax = axes1[i // 4, i % 4]
        ax.plot(df[col].tail(100), color='#1f77b4', lw=2)
        ax.set_title(title, fontproperties=fprop, fontsize=14)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontproperties(fprop)
    st.pyplot(fig1)

    st.divider()

    # 섹션 2: 실물 경제 및 고용 지표 (1행 3열)
    st.subheader("💼 실물 경제 및 고용 지표 모니터링")
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 6))

    # 1. 글로벌 물동량 (Freight)
    axes2[0].plot(df['Freight'].tail(100), color='green', lw=2)
    axes2[0].set_title("글로벌 물동량 지표 (BDRY)", fontproperties=fprop, fontsize=15)
    axes2[0].annotate("물동량 감소 시 경기 둔화 신호", xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontproperties=fprop)

    # 2. 미국 실업수당 청구 건수
    if not us_job.empty:
        axes2[1].plot(us_job.tail(50), color='red', lw=2)
        axes2[1].set_title("미국 실업수당 청구 건수 (Initial Claims)", fontproperties=fprop, fontsize=15)
        axes2[1].annotate("수치 상승 시 고용 시장 위축", xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontproperties=fprop)

    # 3. 한국 실업수당 청구 건수 (프록시 지표)
    if not kr_job.empty:
        axes2[2].plot(kr_job.tail(50), color='orange', lw=2)
        axes2[2].set_title("한국 실업수당 청구 건수 (Trend)", fontproperties=fprop, fontsize=15)
        axes2[2].annotate("국내 소비 심리 및 고용 지표", xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontproperties=fprop)

    for ax in axes2:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontproperties(fprop)
    
    plt.tight_layout()
    st.pyplot(fig2)

except Exception as e:
    st.error(f"데이터 정합성 확인 중 오류 발생: {e}")
