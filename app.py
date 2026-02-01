import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
from io import StringIO
import google.generativeai as genai

# 1. 페이지 설정
st.set_page_config(page_title="주식 시장 하락 전조 신호 모니터링", layout="wide")

# 자동 새로고침 설정 (10분 간격)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=600000, key="datarefresh")
except ImportError:
    pass

# 2. 고정 NewsAPI Key 및 Gemini API Key 설정
NEWS_API_KEY = "13cfedc9823541c488732fb27b02fa25"
GEMINI_API_KEY = "AIzaSyBZT8GHuD9E9TuhbsZxlRPXxoQfAXNCnV8"

# Gemini 설정 및 모델 초기화 (에러 해결을 위해 모델명 명시적 지정)
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # 404 에러 방지를 위해 가장 안정적인 모델명 사용
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gemini 설정 중 오류 발생: {e}")

# AI 분석 함수 정의
def get_ai_analysis(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # 오류 발생 시 구체적인 메시지 반환
        return f"AI 분석을 가져오는 중 오류가 발생했습니다: {str(e)}"

# 코로나19 폭락 기점 날짜 정의 (S&P 500 고점 기준)
COVID_EVENT_DATE = "2020-02-19"

# 관리자 설정 (보안 강화: st.secrets 사용)
try:
    ADMIN_ID = st.secrets["admin"]["id"]
    ADMIN_PW = st.secrets["admin"]["pw"]
except FileNotFoundError:
    ADMIN_ID = "admin_temp" 
    ADMIN_PW = "temp_pass" 
except KeyError:
    ADMIN_ID = "admin_temp"
    ADMIN_PW = "temp_pass"

# 구글 시트 설정
SHEET_ID = "1eu_AeA54pL0Y0axkhpbf5_Ejx0eqdT0oFM3WIepuisU"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
GSHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbyli4kg7O_pxUOLAOFRCCiyswB5TXrA0RUMvjlTirSxLi4yz3tXH1YoGtNUyjztpDsb/exec" 

# CSS 주입
st.markdown("""
    <style>
    h1 { font-size: clamp(24px, 4vw, 48px) !important; }
    .guide-header { font-size: clamp(18px, 2.5vw, 28px) !important; font-weight: 600; margin-bottom: 45px !important; margin-top: 60px !important; padding-top: 10px !important; }
    .guide-text { font-size: clamp(14px, 1.2vw, 20px) !important; line-height: 1.8 !important; }
    div[data-testid="stMarkdownContainer"] table { width: 100% !important; table-layout: auto !important; margin-bottom: 10px !important; }
    div[data-testid="stMarkdownContainer"] table th, div[data-testid="stMarkdownContainer"] table td { font-size: clamp(12px, 1.1vw, 16px) !important; word-wrap: break-word !important; padding: 12px 4px !important; }
    hr { margin-top: 1rem !important; margin-bottom: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

def get_kst_now():
    return datetime.now() + timedelta(hours=9)

st.title("KOSPI 위험 모니터링 (KOSPI Market Risk Index)")
st.markdown(f"이 대시보드는 **향후 1주일(5거래일) 내외**의 시장 변동 위험을 포착하는데 최적화 되어 있습니다. (업데이트: {get_kst_now().strftime('%m월 %d일 %H시 %M분')})")
st.markdown("---")

with st.expander("📖 지수 가이드북"):
    st.subheader("1. 지수 산출 핵심 지표 (Core Indicators)")
    st.write("미국 S&P 500, 원/달러 환율, VIX, 구리 가격, 장단기 금리차 등 하락 선행성을 가진 지표를 사용합니다.")
    st.divider()
    st.subheader("2. 선행성 분석 범위 (Lag Analysis)")
    st.write("단기 선행성(1~5일) 분석을 통해 향후 1주일 내의 변동 위험을 포착합니다.")
    st.divider()
    st.subheader("3. 수리적 산출 공식")
    st.latex(r"\rho(k) = \frac{Cov(X_{t-k}, Y_t)}{\sigma_{X_{t-k}} \sigma_{Y_t}}")

@st.cache_data(ttl=600)
def load_data():
    end_date = datetime.now()
    start_date = "2019-01-01"
    kospi = yf.download("^KS11", start=start_date, end=end_date)
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)
    fx = yf.download("KRW=X", start=start_date, end=end_date)
    b10 = yf.download("^TNX", start=start_date, end=end_date)
    b2 = yf.download("^IRX", start=start_date, end=end_date)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    cop = yf.download("HG=F", start=start_date, end=end_date)
    dry = yf.download("BDRY", start=start_date, end=end_date)
    oil = yf.download("CL=F", start=start_date, end=end_date)
    dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date)
    sector_map = {"반도체": "005930.KS", "자동차": "005380.KS", "바이오": "207940.KS"}
    sector_raw = yf.download(list(sector_map.values()), period="5d")['Close']
    return kospi, sp500, fx, b10, b2, vix, cop, dry, oil, dxy, sector_raw, sector_map

try:
    with st.spinner('데이터 분석 중...'):
        kospi, sp500, fx, bond10, bond2, vix_data, copper_data, freight_data, wti_data, dxy_data, sector_raw, sector_map = load_data()

    def get_clean_series(df):
        if df.empty: return pd.Series()
        df = df[~df.index.duplicated(keep='first')]
        return df['Close'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df['Close']

    ks_s = get_clean_series(kospi)
    sp_s = get_clean_series(sp500).reindex(ks_s.index).ffill()
    fx_s = get_clean_series(fx).reindex(ks_s.index).ffill()
    b10_s = get_clean_series(bond10).reindex(ks_s.index).ffill()
    b2_s = get_clean_series(bond2).reindex(ks_s.index).ffill()
    vx_s = get_clean_series(vix_data).reindex(ks_s.index).ffill()
    cp_s = get_clean_series(copper_data).reindex(ks_s.index).ffill()
    fr_s = get_clean_series(freight_data).reindex(ks_s.index).ffill()
    wt_s = get_clean_series(wti_data).reindex(ks_s.index).ffill()
    dx_s = get_clean_series(dxy_data).reindex(ks_s.index).ffill()
    yield_curve = b10_s - b2_s
    ma20 = ks_s.rolling(window=20).mean()

    def get_hist_score_val(series, current_idx, inverse=False):
        sub = series.loc[:current_idx].iloc[-252:]
        if len(sub) < 10 or sub.max() == sub.min(): return 50.0
        curr = series.loc[current_idx]
        score = ((sub.max() - curr) / (sub.max() - sub.min())) * 100 if inverse else ((curr - sub.min()) / (sub.max() - sub.min())) * 100
        return float(score)

    # 가중치 계산 및 사이드바 설정
    st.sidebar.header("⚙️ 가중치 설정")
    w_macro = st.sidebar.slider("매크로", 0.0, 1.0, 0.25)
    w_global = st.sidebar.slider("글로벌 리스크", 0.0, 1.0, 0.25)
    w_fear = st.sidebar.slider("시장 공포", 0.0, 1.0, 0.25)
    w_tech = st.sidebar.slider("기술적 지표", 0.0, 1.0, 0.25)
    total_w = w_macro + w_global + w_fear + w_tech
    if total_w == 0: st.stop()

    # 현재 위험 지수 산출
    m_now = (get_hist_score_val(fx_s, ks_s.index[-1]) + get_hist_score_val(b10_s, ks_s.index[-1]) + get_hist_score_val(cp_s, ks_s.index[-1], True)) / 3
    t_now = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    total_risk_index = (m_now * w_macro + t_now * w_tech + get_hist_score_val(sp_s, ks_s.index[-1], True) * w_global + get_hist_score_val(vx_s, ks_s.index[-1]) * w_fear) / total_w

    # 메인 게이지 표시
    c_gauge, c_guide = st.columns([1, 1.6])
    with c_guide:
        st.markdown('<p class="guide-header">💡 지수 해석 가이드</p>', unsafe_allow_html=True)
        st.markdown('<div class="guide-text">0-40 (Safe): 적극적 수익 추구 / 60-80 (Danger): 리스크 관리 필수</div>', unsafe_allow_html=True)
    with c_gauge:
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=total_risk_index, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"}, 'steps': [{'range': [0, 40], 'color': "green"}, {'range': [80, 100], 'color': "red"}]}))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 뉴스 및 AI 분석 섹션
    st.markdown("---")
    cn, cr = st.columns(2)
    with cn:
        st.subheader("📰 글로벌 경제 뉴스 (AI 분석)")
        news_items = get_market_news()
        all_titles = ". ".join([n['title'] for n in news_items])
        for n in news_items: st.markdown(f"- [{n['title']}]({n['link']})")
        if news_items:
            with st.spinner("AI 분석 중..."):
                prompt = f"다음 뉴스들을 종합하여 시장 리스크를 한국어 두 문장으로 요약해줘: {all_titles}"
                st.info(f"🔎 **AI 뉴스 통합 분석:** {get_ai_analysis(prompt)}")

    # 지표 분석 및 AI 종합 진단
    st.markdown("---")
    st.subheader("🔍 주요 상관관계 지표 분석 (AI 해설)")
    latest_data = f"S&P500: {sp_s.iloc[-1]:.2f}, 환율: {fx_s.iloc[-1]:.1f}, VIX: {vx_s.iloc[-1]:.2f}"
    with st.expander("🤖 Gemini AI 현재 시장 종합 진단", expanded=True):
        st.write(get_ai_analysis(f"다음 지표를 바탕으로 한국 증시 영향을 3문장으로 설명해줘: {latest_data}"))

    # 그래프 생성 함수
    def create_chart(series, title, threshold, desc):
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, name=title))
        fig.add_hline(y=threshold, line_color="red")
        fig.add_vline(x=COVID_EVENT_DATE, line_dash="dash", line_color="blue")
        return fig

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    r1_c1.plotly_chart(create_chart(sp_s, "S&P 500", sp_s.mean()*0.9, "하락"), use_container_width=True)
    r1_c2.plotly_chart(create_chart(fx_s, "환율", fx_s.mean()*1.02, "상승"), use_container_width=True)
    r1_c3.plotly_chart(create_chart(cp_s, "Copper", cp_s.mean()*0.9, "수요 위축"), use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {e}")
