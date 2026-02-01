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

# 2. Secrets에서 API Key 및 설정값 불러오기 (image_4f74fa.png 구조 반영)
try:
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    NEWS_API_KEY = st.secrets["news_api"]["api_key"]
    ADMIN_ID = st.secrets["auth"]["admin_id"]
    ADMIN_PW = st.secrets["auth"]["admin_pw"]
except KeyError as e:
    st.error(f"Secrets 설정이 누락되었습니다: {e}. Streamlit Cloud 설정창을 확인하세요.")
    st.stop()

# Gemini 설정 및 모델 초기화
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gemini 설정 중 오류 발생: {e}")

# AI 분석 함수 정의
def get_ai_analysis(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 분석을 가져오는 중 오류가 발생했습니다: {str(e)}"

# 코로나19 폭락 기점 날짜 정의
COVID_EVENT_DATE = "2020-02-19"

# 구글 시트 설정
SHEET_ID = "1eu_AeA54pL0Y0axkhpbf5_Ejx0eqdT0oFM3WIepuisU"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
GSHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbyli4kg7O_pxUOLAOFRCCiyswB5TXrA0RUMvjlTirSxLi4yz3tXH1YoGtNUyjztpDsb/exec" 

# CSS 주입: 제목 폰트 유동성 및 가이드북 간격/정렬 조정
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
    st.write("본 대시보드의 위험 지수는 향후 1주일(5거래일) 내외의 시장 변동 위험을 포착하고 대비하는데 최적화되어 설계되었습니다.")
    st.divider()
    st.subheader("3. 수리적 산출 공식")
    st.latex(r"\rho(k) = \frac{Cov(X_{t-k}, Y_t)}{\sigma_{X_{t-k}} \sigma_{Y_t}} \quad (0 \le k \le 5)")

# 4. 데이터 수집 함수
@st.cache_data(ttl=600)
def load_data():
    end_date = datetime.now()
    start_date = "2019-01-01"
    kospi = yf.download("^KS11", start=start_date, end=end_date)
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)
    exchange_rate = yf.download("KRW=X", start=start_date, end=end_date)
    us_10y = yf.download("^TNX", start=start_date, end=end_date)
    us_2y = yf.download("^IRX", start=start_date, end=end_date)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    copper = yf.download("HG=F", start=start_date, end=end_date)
    freight = yf.download("BDRY", start=start_date, end=end_date)
    wti = yf.download("CL=F", start=start_date, end=end_date)
    dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date)
    
    sector_tickers = {"반도체": "005930.KS", "자동차": "005380.KS", "2차전지": "051910.KS", "바이오": "207940.KS", "인터넷": "035420.KS", "금융": "055550.KS"}
    sector_raw = yf.download(list(sector_tickers.values()), period="5d")['Close']
    return kospi, sp500, exchange_rate, us_10y, us_2y, vix, copper, freight, wti, dxy, sector_raw, sector_tickers

# 4.5 글로벌 경제 뉴스 수집
@st.cache_data(ttl=600)
def get_market_news():
    api_url = "https://newsapi.org/v2/everything"
    params = {"q": "stock market risk", "sortBy": "publishedAt", "language": "en", "pageSize": 5, "apiKey": NEWS_API_KEY}
    try:
        res = requests.get(api_url, params=params, timeout=10)
        data = res.json()
        if data.get("status") == "ok":
            return [{"title": a["title"], "link": a["url"]} for a in data.get("articles", [])]
        return []
    except: return []

# 4.6 게시판 데이터 로직
@st.cache_data(ttl=10) 
def load_board_data():
    try:
        res = requests.get(f"{GSHEET_CSV_URL}&cache_bust={datetime.now().timestamp()}", timeout=10)
        res.encoding = 'utf-8' 
        if res.status_code == 200:
            df = pd.read_csv(StringIO(res.text), dtype=str).fillna("")
            return df.to_dict('records')
        return []
    except: return []

def save_to_gsheet(date, author, content, password, action="append"):
    try:
        payload = {"date": str(date), "author": str(author), "content": str(content), "password": str(password), "action": action}
        res = requests.post(GSHEET_WEBAPP_URL, data=json.dumps(payload), timeout=15)
        if res.status_code == 200:
            st.cache_data.clear()
            return True
        return False
    except: return False

try:
    with st.spinner('데이터 및 ML 가중치 분석 중...'):
        kospi, sp500, fx, bond10, bond2, vix_data, copper_data, freight_data, wti_data, dxy_data, sector_raw, sector_map = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series()
        df = df[~df.index.duplicated(keep='first')]
        if isinstance(df.columns, pd.MultiIndex): return df['Close'].iloc[:, 0]
        return df['Close']

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
        try:
            sub = series.loc[:current_idx].iloc[-252:]
            if len(sub) < 10: return 50.0
            min_v, max_v = sub.min(), sub.max(); curr_v = series.loc[current_idx]
            if max_v == min_v: return 50.0
            return ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100
        except: return 50.0

    # 5. 사이드바 - 가중치 및 관리자 모드
    st.sidebar.header("⚙️ 지표별 가중치 설정")
    w_macro = st.sidebar.slider("매크로", 0.0, 1.0, 0.25, step=0.01)
    w_global = st.sidebar.slider("글로벌", 0.0, 1.0, 0.25, step=0.01)
    w_fear = st.sidebar.slider("시장 공포", 0.0, 1.0, 0.25, step=0.01)
    w_tech = st.sidebar.slider("기술적 지표", 0.0, 1.0, 0.25, step=0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 관리자 모드")
    admin_id_input = st.sidebar.text_input("아이디")
    admin_pw_input = st.sidebar.text_input("비밀번호", type="password")
    is_admin = (admin_id_input == ADMIN_ID and admin_pw_input == ADMIN_PW)
    
    total_w = w_macro + w_tech + w_global + w_fear
    if total_w == 0: st.error("가중치 합이 0일 수 없습니다."); st.stop()

    # 현재 위험 지수 계산
    m_now = (get_hist_score_val(fx_s, ks_s.index[-1]) + get_hist_score_val(b10_s, ks_s.index[-1]) + get_hist_score_val(cp_s, ks_s.index[-1], True)) / 3
    t_now = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    total_risk_index = (m_now * w_macro + t_now * w_tech + get_hist_score_val(sp_s, ks_s.index[-1], True) * w_global + get_hist_score_val(vx_s, ks_s.index[-1]) * w_fear) / total_w

    c_gauge, c_guide = st.columns([1, 1.6])
    with c_gauge: 
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=total_risk_index, title={'text': "주식 시장 위험 지수", 'font': {'size': 20}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"}, 'steps': [{'range': [0, 40], 'color': "green"}, {'range': [80, 100], 'color': "red"}]}))
        st.plotly_chart(fig_gauge, use_container_width=True)
    with c_guide:
        st.markdown('<p class="guide-header">💡 지수를 더 똑똑하게 보는 법</p>', unsafe_allow_html=True)
        st.markdown('<div class="guide-text">0-40 (Safe), 40-60 (Watch), 60-80 (Danger), 80-100 (Panic)</div>', unsafe_allow_html=True)

    # 뉴스 분석
    st.markdown("---")
    cn, cr = st.columns(2)
    with cn:
        st.subheader("📰 글로벌 경제 뉴스 (Gemini AI 요약)")
        news_data = get_market_news()
        all_titles = ". ".join([a['title'] for a in news_data])
        for a in news_data: st.markdown(f"- [{a['title']}]({a['link']})")
        if news_data:
            with st.spinner("AI 분석 중..."):
                prompt = f"다음 뉴스 제목들을 바탕으로 시장 리스크를 한국어 두 문장 요약해줘: {all_titles}"
                st.info(get_ai_analysis(prompt))

    # 한 줄 의견 (원본 게시판 로직)
    with cr:
        st.subheader("💬 한 줄 의견(익명)")
        st.session_state.board_data = load_board_data()
        board_container = st.container(height=200)
        with board_container:
            if not st.session_state.board_data: st.write("의견이 없습니다.")
            else:
                for post in st.session_state.board_data[::-1]:
                    st.markdown(f"**{post.get('Author','익명')}**: {post.get('Content','')} <small>({post.get('date','')})</small>", unsafe_allow_html=True)
        with st.form("board_form", clear_on_submit=True):
            f1, f2, f3 = st.columns([1, 1, 3])
            u_name = f1.text_input("성함", value="익명")
            u_pw = f2.text_input("비번", type="password")
            u_content = f3.text_input("내용", max_chars=50)
            if st.form_submit_button("등록") and u_content and u_pw:
                save_to_gsheet(get_kst_now().strftime("%Y-%m-%d %H:%M:%S"), u_name, u_content, u_pw)
                st.rerun()

    # 7. 백테스팅 (원본 로직 완벽 복원)
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 (최근 1년)")
    dates = ks_s.index[-252:]
    hist_risks = []
    for d in dates:
        m = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
        hist_risks.append((m * w_macro + max(0, min(100, 100 - (float(ks_s.loc[d]) / float(ma20.loc[d]) - 0.9) * 500)) * w_tech + get_hist_score_val(sp_s, d, True) * w_global + get_hist_score_val(vx_s, d) * w_fear) / total_w)
    hist_df = pd.DataFrame({'Date': dates, 'Risk': hist_risks, 'KOSPI': ks_s.loc[dates].values})
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Risk'], name="위험 지수", line=dict(color='red')))
    fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot')))
    fig_bt.update_layout(yaxis=dict(title="위험 지수", range=[0, 100]), yaxis2=dict(overlaying="y", side="right"), height=400)
    st.plotly_chart(fig_bt, use_container_width=True)

    # 🦢 블랙스완 비교
    st.markdown("---")
    st.subheader("Swan 블랙스완 과거 사례 비교 시뮬레이션")
    def get_norm_risk_proxy(t, s, e):
        d = yf.download(t, start=s, end=e)['Close']
        if isinstance(d, pd.DataFrame): d = d.iloc[:, 0]
        return 100 - ((d - d.min()) / (d.max() - d.min()) * 100)
    col_bs1, col_bs2 = st.columns(2)
    with col_bs1:
        st.info("**2008 금융위기 vs 현재**")
        bs_2008 = get_norm_risk_proxy("^KS11", "2008-01-01", "2009-01-01")
        fig_bs1 = go.Figure()
        fig_bs1.add_trace(go.Scatter(y=hist_df['Risk'].iloc[-120:].values, name="현재 위험", line=dict(color='red', width=3)))
        fig_bs1.add_trace(go.Scatter(y=bs_2008.values, name="2008 위기", line=dict(color='black', dash='dot')))
        st.plotly_chart(fig_bs1, use_container_width=True)

    # 9. 지표 분석 및 AI 해설
    st.markdown("---")
    st.subheader("🔍 주요 상관관계 지표 분석 (AI 해설)")
    latest_data = f"- S&P 500: {sp_s.iloc[-1]:.2f}, - 환율: {fx_s.iloc[-1]:.1f}원, - VIX: {vx_s.iloc[-1]:.2f}"
    with st.expander("🤖 Gemini AI 현재 시장 종합 진단", expanded=True):
        st.write(get_ai_analysis(f"다음 지표를 바탕으로 한국 증시 영향을 전문적으로 3문장 요약해줘: {latest_data}"))

    # 동조화 및 섹터 분석
    st.markdown("---")
    st.subheader("📊 지수간 동조화 및 섹터 분석")
    sp_norm = (sp_s - sp_s.mean()) / sp_s.std(); fr_norm = (fr_s - fr_s.mean()) / fr_s.std()
    fig_norm = go.Figure()
    fig_norm.add_trace(go.Scatter(x=sp_norm.index, y=sp_norm.values, name="S&P 500 (Std)"))
    fig_norm.add_trace(go.Scatter(x=fr_norm.index, y=fr_norm.values, name="BDRY (Std)"))
    st.plotly_chart(fig_norm, use_container_width=True)

    sector_perf = []
    for n, t in sector_map.items():
        try:
            cur = sector_raw[t].iloc[-1]; pre = sector_raw[t].iloc[-2]
            sector_perf.append({"섹터": n, "등락률": round(((cur - pre) / pre) * 100, 2)})
        except: pass
    if sector_perf:
        st.plotly_chart(px.bar(pd.DataFrame(sector_perf), x="섹터", y="등락률", color="등락률", title="섹터별 등락 현황"), use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {get_kst_now().strftime('%d일 %H시 %M분')} | NewsAPI 및 Gemini AI 연동")
