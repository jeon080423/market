import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import time

# [자동 업데이트] 15분 주기
st_autorefresh(interval=15 * 60 * 1000, key="datarefresh")

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path): return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()
st.set_page_config(page_title="KOSPI 인텔리전스 진단 v3.0 (Lite)", layout="wide")

# [데이터 수집]
@st.cache_data(ttl=900)
def load_expert_data():
    tickers = {
        '^KS11': 'KOSPI', 'USDKRW=X': 'Exchange', '^SOX': 'SOX', '^GSPC': 'SP500', 
        '^VIX': 'VIX', '000001.SS': 'China', '^TNX': 'US10Y', '^IRX': 'US2Y',
        '005930.KS': 'Samsung', '000660.KS': 'Hynix', '005380.KS': 'Hyundai', '373220.KS': 'LG_Energy'
    }
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    combined_df = pd.DataFrame()
    
    for ticker, name in tickers.items():
        for _ in range(3): 
            try:
                raw = yf.download(ticker, start=start_date, interval='1d', progress=False)
                if not raw.empty:
                    try:
                        rt = yf.download(ticker, period='1d', interval='1m', progress=False)
                        val = rt['Close'].iloc[-1] if not rt.empty else raw['Close'].iloc[-1]
                        series = raw['Close'].copy()
                        series.iloc[-1] = val
                    except:
                        series = raw['Close']
                    combined_df[name] = series
                    break 
                time.sleep(1) 
            except: continue
                
    if combined_df.empty or 'KOSPI' not in combined_df.columns: 
        raise Exception("데이터 수집 실패. 네트워크를 확인하세요.")
        
    df = combined_df.ffill().interpolate()
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    return df.dropna().tail(300)

def custom_date_formatter(x, pos):
    dt = mdates.num2date(x)
    return dt.strftime('%Y/%m') if dt.month == 1 else dt.strftime('%m')

try:
    df = load_expert_data()
    
    # --- 데이터 분석 (단순 통계 기반) ---
    # 최근 5일 등락률로 시장 분위기 파악
    kospi_ret_1w = (df['KOSPI'].iloc[-1] / df['KOSPI'].iloc[-6] - 1) * 100
    sox_ret_1w = (df['SOX'].iloc[-1] / df['SOX'].iloc[-6] - 1) * 100
    
    # 시장 분위기 판단
    if kospi_ret_1w > 1.5: market_mood = "강세장 (Bullish)"
    elif kospi_ret_1w < -1.5: market_mood = "약세장 (Bearish)"
    else: market_mood = "보합세 (Neutral)"

    # AI 브리핑 문구 생성
    ai_summary = f"최근 5일간 KOSPI는 **{kospi_ret_1w:+.2f}%** 변동했으며, 반도체 지수(SOX)는 **{sox_ret_1w:+.2f}%** 움직였습니다. 현재 시장은 **{market_mood}** 흐름을 보이고 있습니다."

    # --- 레이아웃 ---
    st.markdown(f"## 🏛️ KOSPI 인텔리전스 진단 시스템 <small>v3.0 (Lite)</small>", unsafe_allow_html=True)
    
    # 1행: AI 요약 및 현금 비중 가이드
    h1, h2 = st.columns([3, 1])
    with h1:
        st.info(f"🤖 **AI 마켓 브리핑:** {ai_summary}")
    with h2:
        # VIX 지수 기반 현금 비중 제안 (단순화된 로직)
        current_vix = df['VIX'].iloc[-1]
        cash = 20 if current_vix < 15 else 40 if current_vix < 20 else 60 if current_vix < 25 else 80
        st.metric("권장 현금 비중", f"{cash}%", f"VIX: {current_vix:.2f}")

    st.divider()

    # 2행: 주도 업종 분석 (기능 유지)
    st.subheader("🔄 주도 업종 수익률 모멘텀 (최근 5일)")
    sector_rets = df[['Samsung', 'Hynix', 'Hyundai', 'LG_Energy']].pct_change(5).iloc[-1] * 100
    st.bar_chart(sector_rets)

    st.divider()

    # 3행: 8대 지표 그래프 (기능 유지)
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    plt.subplots_adjust(hspace=0.4)
    config = [
        ('KOSPI', '1. KOSPI 본체', 'MA250 - 1σ', '선 아래로 하향 시 [추세 붕괴]'),
        ('Exchange', '2. 원/달러 환율', 'MA250 + 1.5σ', '선 위로 상향 시 [외인 자금 이탈]'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', '선 아래로 하향 시 [IT 공급망 위기]'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '선 아래로 하향 시 [글로벌 심리 위축]'),
        ('VIX', '5. 공포지수(VIX)', '20.0', '선 위로 상향 시 [시장 패닉 진입]'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '선 아래로 하향 시 [아시아권 경기 침체]'),
        ('Yield_Spread', '7. 장단기 금리차', '0.0', '선 아래로 하향 시 [경제 불황 전조]'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '선 위로 상향 시 [유동성 긴축 압박]')
    ]

    for i, (col, title, th_label, warn_text) in enumerate(config):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(100)
        
        # 단순 이동평균선 계산
        ma = df[col].rolling(window=250).mean().iloc[-1]
        std = df[col].rolling(window=250).std().iloc[-1]
        
        # 임계치 설정 (단순화)
        if col == 'Exchange': threshold = ma + (1.5 * std)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(th_label)
        elif col in ['US10Y']: threshold = ma + std
        else: threshold = ma - std
        
        ax.plot(plot_data, color='#34495e', lw=2.5)
        ax.axhline(y=threshold, color='#e74c3c', ls='--', lw=2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold')
        
        # 현재가 표시
        curr_val = plot_data.iloc[-1]
        ax.text(plot_data.index[-1], curr_val, f"{curr_val:.2f}", 
                fontproperties=fprop, color='blue', va='bottom', ha='left', fontsize=10)

        ax.set_xlabel(f"기준선: {threshold:.2f} | {warn_text}", fontproperties=fprop, fontsize=11, color='#c0392b')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontproperties(fprop)

    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ 시스템 오류: {e}")
