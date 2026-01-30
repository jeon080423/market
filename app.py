import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# [중요] 최상단에서 라이브러리 체크
try:
    import yfinance as yf
    from pykrx import stock
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
except ImportError as e:
    st.error(f"라이브러리 로드 실패: {e}")
    st.info("이 에러는 GitHub의 'requirements.txt' 파일이 없거나 잘못되었을 때 발생합니다.")
    st.markdown("### 현재 서버에 인식된 파일 목록")
    st.write(os.listdir('.')) 
    st.stop()

# 타임존 및 페이지 설정
os.environ['TZ'] = 'Asia/Seoul'
st.set_page_config(page_title="KOSPI 하락 전조 분석", layout="wide")

# [보안] 비밀번호 함수
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if st.session_state["password_correct"]:
        return True
        
    st.title("🔐 접속 보안")
    password = st.text_input("비밀번호", type="password")
    if st.button("접속"):
        if password == "1234":  # 실제 서비스 시 환경변수 권장
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("비밀번호 오류")
    return False

if not check_password():
    st.stop()

# 데이터 수집 및 회귀 분석 로직
@st.cache_data(ttl=3600)
def get_data():
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
    
    # 1. KOSPI 종가 데이터 (pykrx의 컬럼명은 '종가'임에 유의)
    df_kospi = stock.get_market_ohlcv(start, end, "KOSPI")[['종가']]
    
    # 2. 투자자별 순매수 데이터 (필요한 경우 사용, 여기서는 예시로 로드)
    # 주의: get_market_net_purchases_of_equities_by_ticker 등을 주로 사용함
    df_inv = stock.get_market_net_purchases_of_equities_by_ticker(start, end, "KOSPI")
    
    # 3. 글로벌 지수 데이터 (yfinance)
    tickers = {
        '^SOX': 'SOX', 
        '^GSPC': 'SP500', 
        '^VIX': 'VIX', 
        'USDKRW=X': 'USD_KRW', 
        '^TNX': 'US10Y', 
        '^IRX': 'US2Y'
    }
    df_global = yf.download(list(tickers.keys()), start=pd.to_datetime(start), end=pd.to_datetime(end))['Close']
    df_global = df_global.rename(columns=tickers)
    
    # 데이터 병합 (시계열 기준)
    df = pd.concat([df_kospi, df_global], axis=1).ffill().bfill()
    
    # 파생 변수 생성
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    
    return df.dropna()

# 실행 및 시각화
try:
    data = get_data()
    st.success("✅ 데이터 로드 및 분석 완료!")

    # 데이터 확인용 차트 (KOSPI 종가와 환율)
    st.subheader("📊 주요 지표 트렌드")
    # '종가' 컬럼이 KOSPI 데이터임
    st.line_chart(data[['종가', 'USD_KRW']])
    
    st.write("최근 데이터 요약", data.tail())

except Exception as e:
    st.error(f"데이터 처리 중 오류 발생: {e}")
