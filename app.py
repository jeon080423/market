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
from groq import Groq

# 1. 페이지 설정
st.set_page_config(page_title="주식 시장 하락 전조 신호 모니터링", layout="wide")

# 자동 새로고침 설정 (10분 간격)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=600000, key="datarefresh")
except ImportError:
    pass

# 2. Secrets에서 API Key 불러오기
try:
    NEWS_API_KEY = st.secrets["news_api"]["api_key"]
    GROQ_API_KEY = st.secrets["groq"]["api_key"]
except KeyError:
    st.error("Secrets 설정(API Key)이 누락되었습니다. 설정을 확인해 주세요.")
    st.stop()

# Groq 설정 및 모델 초기화
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Groq 설정 중 오류 발생: {e}")

# AI 분석 함수 정의 (할당량 보호를 위해 캐시 적용)
@st.cache_data(ttl=3600)
def get_ai_analysis(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"AI 분석을 가져오는 중 오류가 발생했습니다: {str(e)}"

# 코로나19 폭락 기점 날짜 정의
COVID_EVENT_DATE = "2020-02-19"

# 관리자 설정
try:
    ADMIN_ID = st.secrets["auth"]["admin_id"]
    ADMIN_PW = st.secrets["auth"]["admin_pw"]
except KeyError:
    ADMIN_ID = "admin_temp" 
    ADMIN_PW = "temp_pass"

# 구글 시트 설정
SHEET_ID = "1eu_AeA54pL0Y0axkhpbf5_Ejx0eqdT0oFM3WIepuisU"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
GSHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbyli4kg7O_pxUOLAOFRCCiyswB5TXrA0RUMvjlTirSxLi4yz3tXH1YoGtNUyjztpDsb/exec" 

# CSS 주입: 레이아웃 시인성 및 박스 높이 최적화
st.markdown("""
    <style>
    h1 { font-size: clamp(24px, 4vw, 48px) !important; }
    .guide-header {
        font-size: clamp(18px, 2.5vw, 28px) !important;
        font-weight: 600;
        margin-bottom: 20px !important; 
        margin-top: 30px !important;    
    }
    .guide-text {
        font-size: clamp(14px, 1.2vw, 18px) !important;
        line-height: 1.6 !important;
    }
    .ai-analysis-box {
        background-color: #ffffff;
        padding: 12px 18px !important;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        border-left: 6px solid #007bff;
        line-height: 1.5 !important;
        font-size: 0.95rem !important;
        margin-bottom: 5px !important;
    }
    /* 사이드바 가중치 알고리즘 박스 스타일 */
    [data-testid="stExpander"] div[role="button"] p { font-weight: bold; }
    hr { margin: 1rem 0 !important; }
    </style>
    """, unsafe_allow_html=True)

def get_kst_now():
    return datetime.now() + timedelta(hours=9)

# 3. 제목 및 설명
st.title("KOSPI 위험 모니터링 (KOSPI Market Risk Index)")
st.markdown(f"향후 1주일 내외의 시장 변동 위험 포착용 대시보드 (업데이트: {get_kst_now().strftime('%m월 %d일 %H시 %M분')})")
st.markdown("---")

# --- [안내서 섹션] ---
with st.expander("📖 지수 가이드북"):
    st.subheader("1. 지수 산출 핵심 지표")
    st.write("본 모델의 지표들은 KOSPI와의 통계적 상관관계 및 하락 선행성을 기준으로 선정되었습니다.")
    st.divider()
    st.subheader("2. 선행성 분석 범위 및 효과")
    st.info("본 위험 지수는 향후 1주일(5거래일) 내외의 단기 변동 위험 포착에 최적화되어 있습니다.")
    st.divider()
    st.subheader("3. 수리적 산출 공식")
    @st.cache_data
    def get_math_formulas():
        st.markdown("#### ① 시차 상관관계")
        st.latex(r"\rho(k) = \frac{Cov(X_{t-k}, Y_t)}{\sigma_{X_{t-k}} \sigma_{Y_t}}")
        st.markdown("#### ② 통계적 변동 기여도 분석")
        st.latex(r"Importance_i = |\beta_i| \times \sigma_{X_i}")
    get_math_formulas()

# 4. 데이터 수집
@st.cache_data(ttl=900)
def load_data():
    end_date = datetime.now()
    start_date = "2019-01-01"
    tickers = {
        "kospi": "^KS11", "sp500": "^GSPC", "fx": "KRW=X", 
        "us10y": "^TNX", "us2y": "^IRX", "vix": "^VIX", 
        "copper": "HG=F", "freight": "BDRY", "wti": "CL=F", "dxy": "DX-Y.NYB"
    }
    data = yf.download(list(tickers.values()), start=start_date, end=end_date)['Close']
    sector_tickers = {
        "반도체": "005930.KS", "자동차": "005380.KS", "2차전지": "051910.KS",
        "바이오": "207940.KS", "인터넷": "035420.KS", "금융": "055550.KS"
    }
    sector_raw = yf.download(list(sector_tickers.values()), period="5d")['Close']
    return (data[[tickers["kospi"]]], data[[tickers["sp500"]]], data[[tickers["fx"]]], 
            data[[tickers["us10y"]]], data[[tickers["us2y"]]], data[[tickers["vix"]]], 
            data[[tickers["copper"]]], data[[tickers["freight"]]], data[[tickers["wti"]]], 
            data[[tickers["dxy"]]], sector_raw, sector_tickers)

# 4.5 뉴스 수집
@st.cache_data(ttl=1800)
def get_market_news():
    api_url = f"https://newsapi.org/v2/everything?q=stock+market+risk&sortBy=publishedAt&language=en&pageSize=5&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(api_url, timeout=10).json()
        return [{"title": a["title"], "link": a["url"]} for a in res.get("articles", [])]
    except: return []

# 4.6 게시판 로직
@st.cache_data(ttl=10) 
def load_board_data():
    try:
        res = requests.get(f"{GSHEET_CSV_URL}&cache_bust={datetime.now().timestamp()}", timeout=10)
        res.encoding = 'utf-8' 
        return pd.read_csv(StringIO(res.text), dtype=str).fillna("").to_dict('records')
    except: return []

def save_to_gsheet(date, author, content, password, action="append"):
    try:
        payload = {"date": str(date), "author": str(author), "content": str(content), "password": str(password), "action": action}
        if requests.post(GSHEET_WEBAPP_URL, data=json.dumps(payload), timeout=15).status_code == 200:
            st.cache_data.clear(); return True
        return False
    except: return False

try:
    with st.spinner('데이터 분석 중...'):
        kospi, sp500, fx, bond10, bond2, vix_data, copper_data, freight_data, wti_data, dxy_data, sector_raw, sector_map = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series(dtype='float64')
        if isinstance(df, pd.DataFrame): df = df.iloc[:, 0]
        return df[~df.index.duplicated(keep='first')]

    ks_s = get_clean_series(kospi).ffill()
    sp_s = get_clean_series(sp500).reindex(ks_s.index).ffill()
    fx_s = get_clean_series(fx).reindex(ks_s.index).ffill()
    b10_s = get_clean_series(bond10).reindex(ks_s.index).ffill()
    b2_s = get_clean_series(bond2).reindex(ks_s.index).ffill()
    vx_s = get_clean_series(vix_data).reindex(ks_s.index).ffill()
    cp_s = get_clean_series(copper_data).reindex(ks_s.index).ffill()
    ma20 = ks_s.rolling(window=20).mean()

    def get_hist_score_val(series, current_idx, inverse=False):
        try:
            sub = series.loc[:current_idx].iloc[-252:]
            min_v, max_v = sub.min(), sub.max(); curr_v = series.loc[current_idx]
            if max_v == min_v: return 50.0
            return ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100
        except: return 50.0

    @st.cache_data(ttl=3600)
    def calculate_ml_lagged_weights(_ks_s, _sp_s, _fx_s, _b10_s, _cp_s, _ma20, _vx_s):
        def find_best_lag(feature, target):
            corrs = [abs(feature.shift(lag).corr(target)) for lag in range(6)]
            return np.argmax(corrs)
        best_lags = {'SP': find_best_lag(_sp_s, _ks_s), 'FX': find_best_lag(_fx_s, _ks_s), 'B10': find_best_lag(_b10_s, _ks_s), 'CP': find_best_lag(_cp_s, _ks_s), 'VX': find_best_lag(_vx_s, _ks_s)}
        data_rows = []
        for d in _ks_s.index[-252:]:
            s_sp = get_hist_score_val(_sp_s.shift(best_lags['SP']), d, True)
            s_fx = get_hist_score_val(_fx_s.shift(best_lags['FX']), d)
            s_vx = get_hist_score_val(_vx_s.shift(best_lags['VX']), d)
            s_tech = max(0, min(100, 100 - (float(_ks_s.loc[d]) / float(_ma20.loc[d]) - 0.9) * 500))
            data_rows.append([(s_fx)/1, s_sp, s_vx, s_tech, _ks_s.loc[d]])
        df_reg = pd.DataFrame(data_rows, columns=['Macro', 'Global', 'Fear', 'Tech', 'KOSPI']).dropna()
        X = (df_reg.iloc[:, :4] - df_reg.iloc[:, :4].mean()) / (df_reg.iloc[:, :4].std() + 1e-6)
        Y = (df_reg['KOSPI'] - df_reg['KOSPI'].mean()) / (df_reg['KOSPI'].std() + 1e-6)
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        adj_imp = (np.abs(coeffs) * X.std().values) + 1e-6 
        return adj_imp / np.sum(adj_imp)

    sem_w = calculate_ml_lagged_weights(ks_s, sp_s, fx_s, b10_s, cp_s, ma20, vx_s)

    # 5. 사이드바 설정
    st.sidebar.header("⚙️ 지표별 가중치 설정")
    w_macro = st.sidebar.slider("매크로", 0.0, 1.0, float(round(sem_w[0], 2)), 0.01)
    w_global = st.sidebar.slider("글로벌 리스크", 0.0, 1.0, float(round(sem_w[1], 2)), 0.01)
    w_fear = st.sidebar.slider("시장 공포", 0.0, 1.0, float(round(sem_w[2], 2)), 0.01)
    w_tech = st.sidebar.slider("국내 기술지표", 0.0, 1.0, float(round(sem_w[3], 2)), 0.01)

    with st.sidebar.expander("ℹ️ 가중치 산출 알고리즘"):
        st.caption("선형 회귀(OLS) 통계 기법을 사용하여 과거 데이터상 각 팩터의 영향력을 역산합니다.")

    # 위험 지수 계산
    m_now = calculate_score = lambda s, i: float(max(0, min(100, ((s.last('365D').max() - s.iloc[-1]) / (s.last('365D').max() - s.last('365D').min())) * 100 if i else ((s.iloc[-1] - s.last('365D').min()) / (s.last('365D').max() - s.last('365D').min())) * 100)))
    total_risk = (m_now(fx_s, False) * w_macro + m_now(sp_s, True) * w_global + m_now(vx_s, False) * w_fear + max(0, min(100, 100 - (float(ks_s.iloc[-1])/float(ma20.iloc[-1]) - 0.9)*500)) * w_tech) / (w_macro+w_global+w_fear+w_tech)

    # 게이지 차트
    c_gauge, c_guide = st.columns([1, 1.6])
    with c_gauge:
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=total_risk, title={'text': "시장 위험 지수"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"}, 'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 70], 'color': "orange"}, {'range': [70, 100], 'color': "red"}]}))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20)); st.plotly_chart(fig_gauge, use_container_width=True)
    with c_guide:
        st.markdown('<p class="guide-header">💡 지수 해석 가이드</p>', unsafe_allow_html=True)
        st.markdown('<div class="guide-text">0-40: 안정기 (적극 투자 고려)<br>40-70: 주의보 (비중 조절 시작)<br>70-100: 위험기 (현금 비중 확대)</div>', unsafe_allow_html=True)

    st.markdown("---")
    cn, cr = st.columns(2)
    with cn:
        st.subheader("📰 글로벌 경제 뉴스 (AI 요약)")
        news_data = get_market_news()
        all_titles = ". ".join([n['title'] for n in news_data])
        if news_data:
            with st.spinner("분석 중..."):
                prompt = f"경제 뉴스 제목들: {all_titles}\n핵심 리스크와 유의점을 한국어 문장 2개로 분석해줘. 지침: 1. 한자를 절대 쓰지 마. 2. 별표(**) 같은 강조 기호를 쓰지 마. 3. 문법에 맞는 표준 한국어만 사용해."
                summary = get_ai_analysis(prompt).replace('**', '').strip()
                st.markdown(f'<div class="ai-analysis-box"><strong>🔎 AI 뉴스 통합 분석</strong><br>{summary}</div>', unsafe_allow_html=True)

    with cr:
        st.subheader("🤖 시장 지표 종합 진단")
        latest_summary = f"S&P500: {sp_s.iloc[-1]:.0f}, 환율: {fx_s.iloc[-1]:.1f}, VIX: {vx_s.iloc[-1]:.1f}"
        with st.spinner("진단 중..."):
            ai_desc_prompt = f"데이터: {latest_summary}\n한국 증시 상황을 진단해줘. 지침: 1. 한자를 절대 쓰지 마. 2. [주요 지표 요약]과 [시장 진단 및 전망] 섹션으로 나누되 별표(**) 기호를 절대 쓰지 마. 3. 박스 크기를 고려해 간결하게 작성해."
            analysis = get_ai_analysis(ai_desc_prompt).replace('**', '').strip()
            st.markdown(f'<div class="ai-analysis-box">{analysis}</div>', unsafe_allow_html=True)

    # 7. 백테스팅 및 차트
    st.markdown("---")
    st.subheader("📈 주요 지표 추세 분석")
    col1, col2, col3 = st.columns(3)
    def small_chart(series, title):
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, name=title))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), title=title); return fig
    col1.plotly_chart(small_chart(sp_s.last('90D'), "미국 S&P 500"), use_container_width=True)
    col2.plotly_chart(small_chart(fx_s.last('90D'), "원/달러 환율"), use_container_width=True)
    col3.plotly_chart(small_chart(vx_s.last('90D'), "VIX 공포 지수"), use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {get_kst_now().strftime('%d일 %H시 %M분')} | NewsAPI & Groq AI")
