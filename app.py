import streamlit as st
import os
import sys

# [중요] 최상단에서 라이브러리 체크
try:
    import yfinance as yf
    from pykrx import stock
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
except ImportError as e:
    st.error(f"라이브러리 로드 실패: {e}")
    st.info("이 에러는 GitHub의 'requirements.txt' 파일이 없거나 잘못되었을 때 발생합니다.")
    st.markdown("### 현재 서버에 인식된 파일 목록")
    st.write(os.listdir('.')) # 서버 루트의 파일 목록을 화면에 출력하여 확인
    st.stop()

# 타임존 설정
os.environ['TZ'] = 'Asia/Seoul'
st.set_page_config(page_title="KOSPI 하락 전조 분석", layout="wide")

# [보안] 비밀번호
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]: return True
    st.title("🔐 접속 보안")
    password = st.text_input("비밀번호", type="password")
    if st.button("접속"):
        if password == "1234":
            st.session_state["password_correct"] = True
            st.rerun()
        else: st.error("비밀번호 오류")
    return False

if not check_password(): st.stop()

# 이하 데이터 수집 및 회귀 분석 로직 (이전 코드와 동일)
@st.cache_data(ttl=3600)
def get_data():
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
    df_kospi = stock.get_market_ohlcv(start, end, "KOSPI", adjusted=False)['종지'
        df_inv = stock.get_market_mkt_purchases_of_equities_by_ticker(start, end, "KOSPI", adjusted=False)[['일자']]
    
    tickers = {'^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX', 'USDKRW=X': 'USD_KRW', '^TNX': 'US10Y', '^IRX': 'US2Y'}
    df_global = yf.download(list(tickers.keys()), start=pd.to_datetime(start), end=pd.to_datetime(end))['Close']
    df_global = df_global.rename(columns=tickers)
    
    df = pd.concat([df_kospi, df_inv, df_global], axis=1).ffill().bfill()
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    return df.dropna()

# 모델링 및 시각화 (중략 - 이전 코드와 동일한 구조 유지)
data = get_data()
st.success("데이터 로드 성공!")

st.line_chart(data[['종가', 'USD_KRW']]) # 테스트용 차트



