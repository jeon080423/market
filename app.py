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
    # gemini 대신 groq 키 호출
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
@st.cache_data(ttl=3600)  # 1시간 동안 동일 프롬프트에 대해 API 호출 방지
def get_ai_analysis(prompt):
    try:
        # Groq 클라이언트를 사용한 텍스트 생성
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="openai/gpt-oss-20b",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"AI 분석을 가져오는 중 오류가 발생했습니다: {str(e)}"

# 코로나19 폭락 기점 날짜 정의 (S&P 500 고점 기준)
COVID_EVENT_DATE = "2020-02-19"

# 관리자 설정 (보안 강화: st.secrets 사용)
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

# CSS 주입: 제목 폰트 유동성 및 가이드북 간격/정렬 조정
st.markdown("""
    <style>
    /* 메인 제목 유동적 폰트 크기 설정 */
    h1 {
        font-size: clamp(24px, 4vw, 48px) !important;
    }
    
    /* 지수 가이드북 제목 스타일 */
    .guide-header {
        font-size: clamp(18px, 2.5vw, 28px) !important;
        font-weight: 600;
        margin-bottom: 45px !important; 
        margin-top: 60px !important;    
        padding-top: 10px !important;
    }

    /* 설명글 유동적 폰트 및 줄간격 설정 */
    .guide-text {
        font-size: clamp(14px, 1.2vw, 20px) !important;
        line-height: 1.8 !important;
    }
    
    /* 가이드북 내 테이블 스타일 */
    div[data-testid="stMarkdownContainer"] table {
        width: 100% !important;
        table-layout: auto !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stMarkdownContainer"] table th,
    div[data-testid="stMarkdownContainer"] table td {
        font-size: clamp(12px, 1.1vw, 16px) !important; /* 표 텍스트 유동성 */
        word-wrap: break-word !important;
        padding: 12px 4px !important; 
    }
    
    /* 수평선(hr) 여백 조정 */
    hr {
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* AI 분석 결과 박스 커스텀 (야간 모드 대응 및 시인성 개선) */
    .ai-analysis-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #31333F !important; /* 글자색 강제 고정 */
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        line-height: 1.65;
        font-size: 1.0rem;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 한국 시간(KST) 설정을 위한 함수
def get_kst_now():
    return datetime.now() + timedelta(hours=9)

# 3. 제목 및 설명
st.title("KOSPI 위험 모니터링 (KOSPI Market Risk Index)")
st.markdown(f"""
이 대시보드는 **향후 1주일(5거래일) 내외**의 시장 변동 위험을 포착하는데 최적화 되어 있습니다.  **검증되지 않은 모델** 이기때문에 **참고만** 하세요.
(마지막 업데이트 KST: {get_kst_now().strftime('%m월 %d일 %H시 %M분')})
""")
st.markdown("---")

# --- [안내서 섹션] ---
with st.expander("📖 지수 가이드북"):
    st.subheader("1. 지수 산출 핵심 지표 (Core Indicators)")
    st.write("""
    본 모델의 지표들은 KOSPI와의 **통계적 상관관계** 및 **하락 선행성**을 기준으로 선정되었습니다.
    * **글로벌 리스크**: 미국 **S&P 500 지수**를 활용하며, 한국 증시와의 강력한 동조화 경향을 반영합니다.
    * **통화 및 유동성**: **원/달러 환율** 및 **달러 인덱스(DXY)** 를 통해 외국인 자본 유출 압력을 측정합니다.
    * **시장 심리**: **VIX(공포 지수)** 를 통해 투자자의 불안 심리와 변동성 전조를 파악합니다.
    * **실물 경제**: 경기 선행 지표인 **구리 가격(Copper)** 과 **장단기 금리차**를 포함합니다.
    """)
    
    st.divider()

    st.subheader("2. 선행성 분석 범위 및 효과 (Lag Analysis)")
    st.markdown("#### **① 선행성 분석 범위 (Lag Optimization)**")
    st.write("""
    * **단기 선행성 (1~5일)**: 현재 모델의 `find_best_lag` 함수는 각 지표와 KOSPI 간의 상관계수가 가장 높게 나타나는 지연 시간을 0일에서 5일 사이에서 찾습니다. 이는 매크로 지표의 변화가 국내 증시에 즉각적 혹은 수일 내에 반영되는 단기적 '전조 신호'를 포착하는 데 최적화되어 있습니다.
    * **중장기 선행성 (1~3개월)**: '장단기 금리차'와 같은 특정 지표는 수개월 이상의 시차를 두고 실물 경기에 영향을 주지만, 본 대시보드는 주식 시장의 단기 하락 위험 모니터링에 초점을 맞추고 있어 모델 내부적으로는 최근의 변동 기여도를 우선시합니다.
    """)
    
    st.markdown("#### **② 지표별 특성에 따른 선행 효과**")
    st.write("""
    * **공포 지수(VIX) 및 환율**: 통상적으로 당일 혹은 1~2일 내외의 매우 짧은 선행성을 보이며 시장의 즉각적인 심리를 반영합니다.
    * **구리 가격 및 물동량(BDRY)**: 실물 경기를 반영하므로 주가지수보다 수일에서 수주 앞서 추세적 변화를 보이는 경향이 있습니다.
    * **장단기 금리차**: 실제 경기 침체는 6개월~1년 이상의 시차를 두고 발생할 수 있으나, 금융 시장은 이를 선반영하여 수주 내에 하락 압력을 받기 시작합니다.
    """)
    
    st.markdown("#### **③ 요약**")
    st.info("본 대시보드의 위험 지수는 수개월 단위의 거시적 경제 지표보다는, **향후 1주일(5거래일) 내외**의 시장 변동 위험을 포착하고 대비하는데 최적화되어 설계되었습니다.")

    st.divider()
    
    st.subheader("3. 수리적 산출 공식")
    @st.cache_data
    def get_math_formulas():
        st.markdown("#### **① 시차 상관관계 (Time-Lagged Correlation)**")
        st.latex(r"\rho(k) = \frac{Cov(X_{t-k}, Y_t)}{\sigma_{X_{t-k}} \sigma_{Y_t}} \quad (0 \le k \le 5)")
        st.markdown("#### **② 통계적 변동 기여도 분석 (Feature Importance)**")
        st.latex(r"Importance_i = |\beta_i| \times \sigma_{X_i}")
        st.markdown("#### **③ Z-Score 표준화 (Standardization)**")
        st.latex(r"Z = \frac{x - \mu}{\sigma}")
    get_math_formulas()

# 4. 데이터 수집 함수 (최적화: 일괄 다운로드)
@st.cache_data(ttl=900) # 15분으로 연장
def load_data():
    # 오늘 데이터를 포함하기 위해 내일 날짜로 end_date 설정
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = "2019-01-01"
    
    # 여러 티커를 한 번에 다운로드하여 API 호출 횟수 절약
    tickers = {
        "kospi": "^KS11", "sp500": "^GSPC", "fx": "KRW=X", 
        "us10y": "^TNX", "us2y": "^IRX", "vix": "^VIX", 
        "copper": "HG=F", "freight": "BDRY", "wti": "CL=F", "dxy": "DX-Y.NYB"
    }
    
    data = yf.download(list(tickers.values()), start=start_date, end=end_date)['Close']
    
    sector_tickers = {
        "반도체": "005930.KS", "자동차": "005380.KS", "2차전지": "051910.KS",
        "바이오": "207940.KS", "인터넷": "035420.KS", "금융": "055550.KS",
        "철강": "005490.KS", "방산": "047810.KS", "유틸리티": "015760.KS"
    }
    sector_raw = yf.download(list(sector_tickers.values()), period="5d")['Close']
    
    # 딕셔너리 형태로 반환하기 위해 분리 (기존 리턴 구조 유지)
    return (
        data[[tickers["kospi"]]], data[[tickers["sp500"]]], data[[tickers["fx"]]], 
        data[[tickers["us10y"]]], data[[tickers["us2y"]]], data[[tickers["vix"]]], 
        data[[tickers["copper"]]], data[[tickers["freight"]]], data[[tickers["wti"]]], 
        data[[tickers["dxy"]]], sector_raw, sector_tickers
    )

# 4.5 글로벌 경제 뉴스 수집 함수 (최적화: 캐시 연장)
@st.cache_data(ttl=1800) # 30분으로 연장
def get_market_news():
    api_url = "https://newsapi.org/v2/everything"
    params = {
        "q": "stock market risk OR recession OR inflation",
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    try:
        res = requests.get(api_url, params=params, timeout=10)
        data = res.json()
        if data.get("status") == "ok":
            news_items = []
            for article in data.get("articles", []):
                news_items.append({"title": article["title"], "link": article["url"]})
            return news_items
        return []
    except:
        return []

try:
    with st.spinner('시차 상관관계 및 가중치 분석 중...'):
        kospi, sp500, fx, bond10, bond2, vix_data, copper_data, freight_data, wti_data, dxy_data, sector_raw, sector_map = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series(dtype='float64')
        # 중복 제거 및 단일 열 추출 최적화
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        return df[~df.index.duplicated(keep='first')]

    # 데이터 끊김 현상 방지를 위해 ffill() 적용
    ks_s = get_clean_series(kospi).ffill()
    sp_s = get_clean_series(sp500).reindex(ks_s.index).ffill()
    fx_s = get_clean_series(fx).reindex(ks_s.index).ffill()
    # 금리 데이터는 휴장일 차이로 인해 reindex 시 결측치가 많이 생길 수 있어 보강함
    b10_s = get_clean_series(bond10).reindex(ks_s.index).ffill()
    b2_s = get_clean_series(bond2).reindex(ks_s.index).ffill()
    vx_s = get_clean_series(vix_data).reindex(ks_s.index).ffill()
    cp_s = get_clean_series(copper_data).reindex(ks_s.index).ffill()
    fr_s = get_clean_series(freight_data).reindex(ks_s.index).ffill()
    wt_s = get_clean_series(wti_data).reindex(ks_s.index).ffill()
    dx_s = get_clean_series(dxy_data).reindex(ks_s.index).ffill()
    
    # 금리차 계산: 보정된 데이터를 사용하여 계산
    yield_curve = b10_s - b2_s
    ma20 = ks_s.rolling(window=20).mean() # 전체 데이터 기반 이동평균 계산

    def get_hist_score_val(series, current_idx, inverse=False):
        try:
            sub = series.loc[:current_idx].iloc[-252:]
            if len(sub) < 10: return 50.0
            min_v, max_v = sub.min(), sub.max(); curr_v = series.loc[current_idx]
            if max_v == min_v: return 50.0
            return ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100
        except: return 50.0

    @st.cache_data(ttl=3600)
    def calculate_ml_lagged_weights(_ks_s, _sp_s, _fx_s, _b10_s, _cp_s, _ma20, _vx_s):
        def find_best_lag(feature, target, max_lag=5):
            corrs = [abs(feature.shift(lag).corr(target)) for lag in range(max_lag + 1)]
            return np.argmax(corrs)
        best_lags = {'SP': find_best_lag(_sp_s, _ks_s), 'FX': find_best_lag(_fx_s, _ks_s), 'B10': find_best_lag(_b10_s, _ks_s), 'CP': find_best_lag(_cp_s, _ks_s), 'VX': find_best_lag(_vx_s, _ks_s)}
        data_rows = []
        for d in _ks_s.index[-252:]:
            s_sp = get_hist_score_val(_sp_s.shift(best_lags['SP']), d, True)
            s_fx = get_hist_score_val(_fx_s.shift(best_lags['FX']), d)
            s_b10 = get_hist_score_val(_b10_s.shift(best_lags['B10']), d)
            s_cp = get_hist_score_val(_cp_s.shift(best_lags['CP']), d, True)
            s_vx = get_hist_score_val(_vx_s.shift(best_lags['VX']), d)
            data_rows.append([ (s_fx + s_b10 + s_cp) / 3, s_sp, s_vx, max(0, min(100, 100 - (float(_ks_s.loc[d]) / float(_ma20.loc[d]) - 0.9) * 500)), _ks_s.loc[d] ])
        
        df_reg = pd.DataFrame(data_rows, columns=['Macro', 'Global', 'Fear', 'Tech', 'KOSPI']).replace([np.inf, -np.inf], np.nan).dropna()
        X = (df_reg.iloc[:, :4] - df_reg.iloc[:, :4].mean()) / (df_reg.iloc[:, :4].std() + 1e-6)
        Y = (df_reg['KOSPI'] - df_reg['KOSPI'].mean()) / (df_reg['KOSPI'].std() + 1e-6)
        
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        adjusted_importance = (np.abs(coeffs) * X.std().values) + 1e-6 
        return adjusted_importance / np.sum(adjusted_importance)

    sem_w = calculate_ml_lagged_weights(ks_s, sp_s, fx_s, b10_s, cp_s, ma20, vx_s)

    # 5. 사이드바 - 가중치 설정
    st.sidebar.header("⚙️ 지표별 가중치 설정")
    if 'slider_m' not in st.session_state: st.session_state.slider_m = float(round(sem_w[0], 2))
    if 'slider_g' not in st.session_state: st.session_state.slider_g = float(round(sem_w[1], 2))
    if 'slider_f' not in st.session_state: st.session_state.slider_f = float(round(sem_w[2], 2))
    if 'slider_t' not in st.session_state: st.session_state.slider_t = float(round(sem_w[3], 2))

    if st.sidebar.button("🔄 권장 최적 가중치로 복귀"):
        st.session_state.slider_m = float(round(sem_w[0], 2)); st.session_state.slider_g = float(round(sem_w[1], 2))
        st.session_state.slider_f = float(round(sem_w[2], 2)); st.session_state.slider_t = float(round(sem_w[3], 2))
        st.rerun()

    w_macro = st.sidebar.slider("매크로 (환율/금리/물동량)", 0.0, 1.0, key="slider_m", step=0.01)
    w_global = st.sidebar.slider("글로벌 시장 위험 (미국 지수)", 0.0, 1.0, key="slider_g", step=0.01)
    w_fear = st.sidebar.slider("시장 공포 (VIX 지수)", 0.0, 1.0, key="slider_f", step=0.01)
    w_tech = st.sidebar.slider("국내 기술적 지표 (이동평균선)", 0.0, 1.0, key="slider_t", step=0.01)

    with st.sidebar.expander("ℹ️ 가중치 산출 알고리즘"):
        st.caption("""
        본 모델은 **시차 상관분석**과 **선형 회귀(OLS)** 통계 기법을 결합하여 가중치를 제안합니다.
        
        1. **시차 최적화 (Lag Optimization)**:
            각 지표와 KOSPI 간의 상관계수가 최대가 되는 지연 일수(0~5일)를 자동으로 탐색합니다.
        2. **기여도 역산 (OLS Regression)**:
            `np.linalg.lstsq`를 사용하여 과거 데이터상 각 팩터가 KOSPI 변동에 미친 통계적 영향력을 산출합니다.
        3. **가중치 정규화**:
            산출된 계수값을 바탕으로 변동성 기여도를 계산하고, 이를 확률적으로 정규화하여 권장 가중치로 제시합니다.
        """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 관리자 모드")
    admin_id_input = st.sidebar.text_input("아이디", key="admin_id")
    admin_pw_input = st.sidebar.text_input("비밀번호", type="password", key="admin_pw")
    is_admin = (admin_id_input == ADMIN_ID and admin_pw_input == ADMIN_PW)
    st.sidebar.markdown("---")
    st.sidebar.subheader("자발적 후원으로 운영됩니다.")
    st.sidebar.write("카카오뱅크 3333-23-8667708 (ㅈㅅㅎ)")
    st.sidebar.write("유료API로 정밀한 데이터가 필요합니다.")
    
    total_w = w_macro + w_tech + w_global + w_fear
    if total_w == 0: 
        st.error("가중치 합이 0일 수 없습니다."); st.stop()

    def calculate_score(current_series, full_series, inverse=False):
        recent = full_series.last('365D')
        min_v, max_v = float(recent.min()), float(recent.max()); curr_v = float(current_series.iloc[-1])
        if max_v == min_v: return 50.0
        return float(max(0, min(100, ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100)))

    m_now = (calculate_score(fx_s, fx_s) + calculate_score(b10_s, b10_s) + calculate_score(cp_s, cp_s, True)) / 3
    t_now = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    total_risk_index = (m_now * w_macro + t_now * w_tech + calculate_score(sp_s, sp_s, True) * w_global + calculate_score(vx_s, vx_s) * w_fear) / total_w

    c_gauge, c_guide = st.columns([1, 1.6])
    with c_guide: 
        st.markdown('<p class="guide-header">💡 지수를 더 똑똑하게 보는 법</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="guide-text">
        0-40 (Safe): 적극적 수익 추구. 주식 비중을 확대하고, 주도주 위주의 공격적 포트폴리오 운용.  
        <br>
        40-60 (Watch): 현금 비중 조절 시작. 추가 매수는 지양하고, 수익이 난 종목은 일부 차익 실현 고려.  
        <br>
        60-80 (Danger): 방어적 운용 및 리스크 관리. 주식 비중을 50% 이하로 축소.  
        <br>
        80-100 (panic): 최우선 리스크 관리. 가급적 현금 비중 최소화, 신용/미수 사용 전면 금지 및 손절매 기준 엄격 적용.
        </div>
        """, unsafe_allow_html=True)

        # 좋아요 기능 추가
        if 'likes' not in st.session_state:
            st.session_state.likes = 0
        
        l_col1, l_col2 = st.columns([1, 5])
        if l_col1.button(f"👍 {st.session_state.likes}"):
            st.session_state.likes += 1
            st.rerun()
        l_col2.write("대시보드가 도움이 되었다면 응원해 주세요!")
        
    with c_gauge: 
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=total_risk_index, 
            title={'text': "주식 시장 위험 지수", 'font': {'size': 20}},
            number={'suffix': ""}, 
            gauge={
                'axis': {'range': [0, 100]}, 
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 40], 'color': "green"}, 
                    {'range': [40, 60], 'color': "yellow"}, 
                    {'range': [60, 80], 'color': "orange"}, 
                    {'range': [80, 100], 'color': "red"}
                ]}))
        fig_gauge.update_layout(margin=dict(l=40, r=40, t=80, b=40), height=350, autosize=True)
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    cn, cr = st.columns(2)
    with cn:
        # 제목 텍스트 Groq로 수정
        st.subheader("📰 글로벌 경제 뉴스 (Groq AI 요약)")
        news_data = get_market_news()
        all_titles = ""
        for a in news_data:
            st.markdown(f"- [{a['title']}]({a['link']})")
            all_titles += a['title'] + ". "
        
    with cr:
        # "AI 뉴스 통합 분석" 위치 유지 (오른쪽 열 상단)
        if news_data:
            with st.spinner("AI가 뉴스를 분석 중입니다..."):
                prompt = f"""
                다음은 최근 주요 경제 뉴스 제목들입니다: {all_titles}
                
                이 뉴스별로 번역하고, 뉴스 중 금융 시장의 리스크와 변동성 관련 뉴스가 있다면 투자자가 유의해야 할 점으로 따로 분석해줘.
                지침:
                1. 반드시 표준 한국어 문법을 준수하고, 전문적인 경제 용어를 올바르게 사용해.
                2. 영어 등 외국어 단어를 그대로 사용하지 말고 적절한 한국어로 번역해서 표현해.
                3. 분석 내용을 두 개의 핵심 문장으로 요약하고, 각 문장은 줄바꿈으로 구분해.
                4. 답변에 강조 기호(예: **, ##)를 절대 사용하지 마.
                5. 한자(漢字)를 단 하나도 포함하지 마. '仔細'와 같은 표현 대신 '자세히'를 사용해.
                6. 답변에 'AI 뉴스 통합 분석'이라는 제목성 문구는 포함하지 마.
                """
                summary_text = get_ai_analysis(prompt)
                
                # 야간 모드 시인성 해결을 위한 스타일 적용
                st.markdown(f"""
                <div class="ai-analysis-box">
                    <strong>🔎 AI 뉴스 통합 분석</strong><br><br>
                    {summary_text.replace('🔎 AI 뉴스 통합 분석:', '').strip()}
                </div>
                """, unsafe_allow_html=True)

    # 7. 백테스팅
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 (최근 1년)")
    st.info("과거 데이터를 사용하여 모델의 유효성을 검증합니다.")
    dates = ks_s.index[-252:]
    hist_risks = []
    for d in dates:
        # 데이터 끊김 현상 보정을 위해 ffill된 데이터 사용
        m = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
        hist_risks.append((m * w_macro + max(0, min(100, 100 - (float(ks_s.loc[d]) / float(ma20.iloc[-1]) - 0.9) * 500)) * w_tech + get_hist_score_val(sp_s, d, True) * w_global + get_hist_score_val(vx_s, d) * w_fear) / total_w)
    hist_df = pd.DataFrame({'Date': dates, 'Risk': hist_risks, 'KOSPI': ks_s.loc[dates].values})
    cb1, cb2 = st.columns([3, 1])
    with cb1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Risk'], name="위험 지수", line=dict(color='red'), connectgaps=True)) # connectgaps 추가
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot'), connectgaps=True))
        fig_bt.update_layout(yaxis=dict(title="위험 지수", range=[0, 100]), yaxis2=dict(overlaying="y", side="right"), height=400); st.plotly_chart(fig_bt, use_container_width=True)
    with cb2:
        corr_val = hist_df['Risk'].corr(hist_df['KOSPI'])
        st.metric("상관계수 (Corr)", f"{corr_val:.2f}")
        st.write("- -1.0~-0.7: 우수\n- -0.7~-0.3: 유의미\n- 0.0이상: 모델 왜곡")
        
        # 백테스팅 AI 분석 추가
        with st.spinner("AI가 추세를 분석 중..."):
            bt_prompt = f"""
            최근 1년 시장 위험 지수와 KOSPI의 상관계수는 {corr_val:.2f}이며, 현재 위험 지수는 {hist_risks[-1]:.1f}입니다. 
            과거 대비 현재 상황이 우려되는 상황인지 투자자 관점에서 짧게 진단해줘.
            지침: 한자 금지, 강조기호 금지, 3문장 이내.
            """
            bt_analysis = get_ai_analysis(bt_prompt)
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.85rem; color: #31333F;">
                <strong>🤖 모델 유효성 진단:</strong><br>{bt_analysis.replace('**', '').replace('##', '')}
            </div>
            """, unsafe_allow_html=True)

    # 7.5 블랙스완
    st.markdown("---")
    st.subheader("🦢 블랙스완(Black Swan) 과거 사례 비교 시뮬레이션")
    def get_norm_risk_proxy(t, s, e):
        # 최신 데이터를 위해 end_date 보정
        bs_end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d') if e == datetime.now().strftime('%Y-%m-%d') else e
        d = yf.download(t, start=s, end=bs_end)['Close'].ffill() # ffill 추가
        if isinstance(d, pd.DataFrame): d = d.iloc[:, 0]
        return 100 - ((d - d.min()) / (d.max() - d.min()) * 100)
    col_bs1, col_bs2 = st.columns(2)
    avg_current_risk = np.mean(hist_df['Risk'].iloc[-30:])
    with col_bs1:
        st.info("**2008 금융위기 vs 현재**")
        bs_2008 = get_norm_risk_proxy("^KS11", "2008-01-01", "2009-01-01")
        fig_bs1 = go.Figure()
        fig_bs1.add_trace(go.Scatter(y=hist_df['Risk'].iloc[-120:].values, name="현재 위험 지수", line=dict(color='red', width=3), connectgaps=True))
        fig_bs1.add_trace(go.Scatter(y=bs_2008.values, name="2008년 위기 궤적", line=dict(color='black', dash='dot'), connectgaps=True))
        st.plotly_chart(fig_bs1, use_container_width=True)
        if avg_current_risk > 60: st.warning(f"⚠️ 현재 위험 지수(평균 {avg_current_risk:.1f})가 위기 초기와 유사합니다.")
        else: st.success(f"✅ 현재 위험 지수(평균 {avg_current_risk:.1f})는 금융위기 경로와 거리가 있습니다.")
    with col_bs2:
        st.info("**2020 코로나 폭락 vs 현재**")
        bs_2020 = get_norm_risk_proxy("^KS11", "2020-01-01", "2020-06-01")
        fig_bs2 = go.Figure()
        fig_bs2.add_trace(go.Scatter(y=hist_df['Risk'].iloc[-120:].values, name="현재 위험 지수", line=dict(color='red', width=3), connectgaps=True))
        fig_bs2.add_trace(go.Scatter(y=bs_2020.values, name="2020년 위기 궤적", line=dict(color='blue', dash='dot'), connectgaps=True))
        st.plotly_chart(fig_bs2, use_container_width=True)
        if avg_current_risk > 50: st.error(f"🚨 주의: 현재 위험 지수가 2020년 팬데믹 상승 구간과 유사한 패턴을 보입니다.")
        else: st.info(f"💡 현재 위험 지수 흐름은 2020년 패닉 궤적보다는 안정적입니다.")

    # 9. 지표별 상세 분석 및 AI 설명
    st.markdown("---")
    st.subheader("🔍 실물 경제 및 주요 상관관계 지표 분석 (AI 해설 포함)")
    
    # 지표 데이터를 AI 프롬프트용으로 생성
    latest_data_summary = f"""
    - S&P 500 현재가: {sp_s.iloc[-1]:.2f} (최근 1년 평균 대비 {((sp_s.iloc[-1]/sp_s.last('365D').mean())-1)*100:+.1f}%)
    - 원/달러 환율: {fx_s.iloc[-1]:.1f}원 (전일 대비 {fx_s.iloc[-1]-fx_s.iloc[-2]:+.1f}원)
    - 구리 가격: {cp_s.iloc[-1]:.2f} (최근 추세: {'상승' if cp_s.iloc[-1] > cp_s.iloc[-5] else '하락'})
    - VIX 지수: {vx_s.iloc[-1]:.2f} (위험 수준: {'높음' if vx_s.iloc[-1] > 20 else '낮음'})
    """
    
    # 가독성 높은 레이아웃 조정을 위한 프롬프트 수정
    with st.expander("🤖 현재 시장 지표 종합 진단", expanded=True):
        with st.spinner("지표 데이터를 분석 중..."):
            ai_desc_prompt = f"""
            주식 시장 지표 데이터: {latest_data_summary}
            
            위 데이터를 바탕으로 현재 한국 증시(KOSPI)의 상황을 진단해줘.
            지침:
            1. 반드시 완벽한 한국어 문장을 사용하고, 외국어를 섞지 마.
            2. 한자(漢字)를 단 하나도 포함하지 마. '仔細'와 같은 표현 대신 '자세히'를 사용해.
            3. 답변 내용에 ** 기호나 ## 기호와 같은 마크다운 강조 기호를 절대 사용하지 마.
            4. 가독성을 위해 다음 형식을 엄격히 지켜줘 (강조 기호 없이 텍스트만 출력):
                [주요 지표 요약]: 각 지표의 상태를 불렛 포인트로 설명.
                [시장 진단 및 전망]: 종합적인 분위기와 투자자 주의 사항을 2~3문장으로 설명.
            5. 쉽고 전문적인 톤을 유지해.
            """
            analysis_output = get_ai_analysis(ai_desc_prompt)
            # 야간 모드 시인성을 위해 배경색 및 글자색 고정 적용
            clean_output = analysis_output.replace('**', '').replace('##', '').strip()
            st.markdown(f"""
            <div class="ai-analysis-box" style="background: #ffffff; color: #31333F !important; border: 1px solid #e0e0e0; border-left: 8px solid #007bff; line-height: 1.5; padding: 15px 20px;">
                {clean_output}
            </div>
            """, unsafe_allow_html=True)

    def create_chart(series, title, threshold, desc_text):
        # 데이터가 비어있지 않은지 확인 후 그래프 생성
        if series is not None and not series.empty:
            fig = go.Figure(go.Scatter(x=series.index, y=series.values, name=title, connectgaps=True)) # connectgaps 추가
            fig.add_hline(y=threshold, line_width=2, line_color="red")
            # 주석 위치 계산을 위한 안전장치
            annot_idx = len(series)//2 if len(series) > 0 else 0
            fig.add_annotation(x=series.index[annot_idx], y=threshold, text=desc_text, showarrow=False, font=dict(color="red"), bgcolor="white", yshift=10)
            fig.add_vline(x=COVID_EVENT_DATE, line_width=1.5, line_dash="dash", line_color="blue")
            fig.add_annotation(x=COVID_EVENT_DATE, y=1, yref="paper", text="COVID 지수 폭락 기점", showarrow=False, font=dict(color="blue"), xanchor="left", xshift=5, bgcolor="white")
            return fig
        return go.Figure()

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        st.subheader("미국 S&P 500")
        st.plotly_chart(create_chart(sp_s, "S&P 500", sp_s.last('365D').mean()*0.9, "평균 대비 -10% 하락 시"), use_container_width=True)
        st.info("**미국 지수**: KOSPI와 강한 정(+)의 상관성  \n**빨간선 기준**: 최근 1년 평균 가격 대비 -10% 하락 지점")
    with r1_c2:
        st.subheader("원/달러 환율")
        fx_th = float(fx_s.last('365D').mean() * 1.02)
        st.plotly_chart(create_chart(fx_s, "원/달러 환율", fx_th, f"{fx_th:.1f}원 돌파 시 위험"), use_container_width=True)
        st.info("**환율**: +2% 상회 시 외국인 자본 유출 심화  \n**빨간선 기준**: 최근 1년 평균 환율 대비 +2% 상승 지점")
    with r1_c3:
        st.subheader("실물 경기 지표 (Copper)")
        st.plotly_chart(create_chart(cp_s, "Copper", cp_s.last('365D').mean()*0.9, "수요 위축 시 위험"), use_container_width=True)
        st.info("**실물 경기**: 구리 가격 하락은 수요 둔화 선행 신호  \n**빨간선 기준**: 최근 1년 평균 가격 대비 -10% 하락 지점")

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        st.subheader("장단기 금리차")
        # 금리차 그래프 생성
        st.plotly_chart(create_chart(yield_curve, "금리차", 0.0, "0 이하 역전 시 위험"), use_container_width=True)
        st.info("**금리차**: 금리 역전은 경기 침체 강력 전조  \n**빨간선 기준**: 금리차가 0(수평)이 되는 역전 한계 지점")
    with r2_c2:
        st.subheader("KOSPI 기술적 분석")
        ks_recent = ks_s.last('30D')
        fig_ks = go.Figure()
        # 현재가 그래프: 선 굵기 및 마커 추가로 가독성 향상
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ks_recent.values, name="현재가", line=dict(color='royalblue', width=3), mode='lines+markers', connectgaps=True))
        # 20일 이동평균선: 계산된 ma20을 reindex하여 끊김 없이 시각화
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ma20.reindex(ks_recent.index).values, name="20일선", line=dict(color='orange', width=2, dash='dot'), connectgaps=True))
        # 화살표 위치 수정: y값을 실제 20일선 지수값(ma20.iloc[-1])으로 명확히 지정
        fig_ks.add_annotation(
            x=ks_recent.index[-1], 
            y=ma20.iloc[-1], # 화살표가 0이 아닌 실제 지수 위치를 가리키도록 수정
            text="20일 평균선 하회 시 위험", 
            showarrow=True, 
            arrowhead=2, 
            ax=0, 
            ay=-40, 
            font=dict(color="red", size=12),
            bgcolor="white",
            bordercolor="red"
        )
        fig_ks.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
        st.plotly_chart(fig_ks, use_container_width=True)
        st.info("**기술적 분석**: 20일선 하회 시 단기 추세 하락")
    with r2_c3:
        st.subheader("VIX 공포 지수")
        st.plotly_chart(create_chart(vx_s, "VIX", 30, "30 돌파 시 패닉"), use_container_width=True)
        st.info("**VIX 지수**: 지수 급등은 투매 가능성 시사  \n**빨간선 기준**: 시장의 극단적 공포를 상징하는 지수 30 지점")

    st.markdown("---")
    r3_c1, r3_c2, r3_c3 = st.columns(3)
    with r3_c1:
        st.subheader("글로벌 물동량 지표 (BDRY)")
        fr_th = round(float(fr_s.last('365D').mean() * 0.85), 2)
        st.plotly_chart(create_chart(fr_s, "BDRY", fr_th, "물동량 급감 시 위험"), use_container_width=True)
        st.info("**물동량**: 지지선 하향 돌파 시 경기 수축 신호  \n**빨간선 기준**: 최근 1년 평균 대비 -15% 하락 지점")
    with r3_c2:
        st.subheader("에너지 가격 (WTI 원유)")
        wt_th = round(float(wt_s.last('365D').mean() * 1.2), 2)
        st.plotly_chart(create_chart(wt_s, "WTI", wt_th, "비용 압력 증가"), use_container_width=True)
        st.info("**유가**: 급등 시 생산 비용 상승 및 인플레 압박  \n**빨간선 기준**: 최근 1년 평균 대비 +20% 급등 지점")
    with r3_c3:
        st.subheader("달러 인덱스 (DXY)")
        dx_th = round(float(dx_s.last('365D').mean() * 1.03), 1)
        st.plotly_chart(create_chart(dx_s, "DXY", dx_th, "유동성 위축 위험"), use_container_width=True)
        st.info("**달러 가치**: 달러 상승은 유동성 축소 및 위험자산 회피  \n**빨간선 기준**: 최근 1년 평균 대비 +3% 강세 지점")

    st.markdown("---")
    st.subheader("📊 지수간 동조화 및 섹터 분석")
    sp_norm = (sp_s - sp_s.mean()) / sp_s.std(); fr_norm = (fr_s - fr_s.mean()) / fr_s.std()
    fig_norm = go.Figure(); fig_norm.add_trace(go.Scatter(x=sp_norm.index, y=sp_norm.values, name="S&P 500 (Std)", line=dict(color='blue'), connectgaps=True))
    fig_norm.add_trace(go.Scatter(x=fr_norm.index, y=fr_norm.values, name="BDRY (Std)", line=dict(color='orange'), connectgaps=True))
    fig_norm.update_layout(title="Z-Score 동조화 추세"); st.plotly_chart(fig_norm, use_container_width=True)
    st.info("""
**[현재 상황 상세 해석 가이드]**
* **주가지수(Blue)가 위에 있을 때**: 실물 경기 뒷받침 없이 기대감만으로 지수가 과열된 상태일 수 있습니다. 하락 가능성이 높습니다.
* **지표들이 비슷한 위치일 때**: 주가와 실물 경기가 동조화되어 움직이는 안정적인 추세입니다.
* **글로벌 물동량(Orange)이 위에 있을 때**: 실물 경기는 회복되었으나 주가가 저평가된 상태입니다. 우상향 가능성을 시사합니다.
""")

    sector_perf = []
    for n, t in sector_map.items():
        try:
            cur = sector_raw[t].iloc[-1]; pre = sector_raw[t].iloc[-2]
            sector_perf.append({"섹터": n, "등락률": round(((cur - pre) / pre) * 100, 2)})
        except: pass
    if sector_perf:
        df_p = pd.DataFrame(sector_perf)
        fig_h = px.bar(df_p, x="섹터", y="등락률", color="등락률", color_continuous_scale='RdBu_r', text="등락률", title="금일 섹터별 대표 종목 등락 현황 (%)")
        st.plotly_chart(fig_h, use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

# 하단 캡션 Groq로 수정
st.caption(f"Last updated: {get_kst_now().strftime('%d일 %H시 %M분')} | NewsAPI 및 Groq AI 분석 엔진 가동 중")
