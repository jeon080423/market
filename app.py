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

# AI 분석 함수 정의
@st.cache_data(ttl=3600)
def get_ai_analysis(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"AI 분석을 가져오는 중 오류가 발생했습니다: {str(e)}"

COVID_EVENT_DATE = "2020-02-19"

try:
    ADMIN_ID = st.secrets["auth"]["admin_id"]
    ADMIN_PW = st.secrets["auth"]["admin_pw"]
except KeyError:
    ADMIN_ID = "admin_temp" 
    ADMIN_PW = "temp_pass"

SHEET_ID = "1eu_AeA54pL0Y0axkhpbf5_Ejx0eqdT0oFM3WIepuisU"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
GSHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbyli4kg7O_pxUOLAOFRCCiyswB5TXrA0RUMvjlTirSxLi4yz3tXH1YoGtNUyjztpDsb/exec" 

st.markdown("""
    <style>
    h1 { font-size: clamp(24px, 4vw, 48px) !important; }
    .guide-header { font-size: clamp(18px, 2.5vw, 28px) !important; font-weight: 600; margin-bottom: 45px !important; margin-top: 60px !important; padding-top: 10px !important; }
    .guide-text { font-size: clamp(14px, 1.2vw, 20px) !important; line-height: 1.8 !important; }
    div[data-testid="stMarkdownContainer"] table { width: 100% !important; table-layout: auto !important; margin-bottom: 10px !important; }
    div[data-testid="stMarkdownContainer"] table th, div[data-testid="stMarkdownContainer"] table td { font-size: clamp(12px, 1.1vw, 16px) !important; word-wrap: break-word !important; padding: 12px 4px !important; }
    hr { margin-top: 1rem !important; margin-bottom: 1rem !important; }
    .ai-analysis-box { background-color: #f0f7ff; padding: 15px 20px; border-radius: 10px; border-left: 5px solid #007bff; line-height: 1.65; font-size: 1.0rem; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

def get_kst_now():
    return datetime.now() + timedelta(hours=9)

st.title("KOSPI 위험 모니터링 (KOSPI Market Risk Index)")
st.markdown(f"""
이 대시보드는 **향후 1주일(5거래일) 내외**의 시장 변동 위험을 포착하는데 최적화 되어 있습니다.  **검증되지 않은 모델** 이기때문에 **참고만** 하세요.
(마지막 업데이트 KST: {get_kst_now().strftime('%m월 %d일 %H시 %M분')})
""")
st.markdown("---")

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

@st.cache_data(ttl=60) # 최신 업데이트 확인을 위해 1분으로 단축
def load_data():
    # 종료일 설정: 오늘이 KST 기준 장이 끝났을 수 있으므로 내일 날짜까지 요청
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = "2019-01-01"
    
    tickers = {
        "kospi": "^KS11", "sp500": "^GSPC", "fx": "KRW=X", 
        "us10y": "^TNX", "us2y": "^IRX", "vix": "^VIX", 
        "copper": "HG=F", "freight": "BDRY", "wti": "CL=F", "dxy": "DX-Y.NYB"
    }
    
    # auto_adjust=True 추가하여 종가 데이터 무결성 강화
    data = yf.download(list(tickers.values()), start=start_date, end=end_date, interval='1d', auto_adjust=True)['Close']
    
    sector_tickers = {
        "반도체": "005930.KS", "자동차": "005380.KS", "2차전지": "051910.KS",
        "바이오": "207940.KS", "인터넷": "035420.KS", "금융": "055550.KS",
        "철강": "005490.KS", "방산": "047810.KS", "유틸리티": "015760.KS"
    }
    sector_raw = yf.download(list(sector_tickers.values()), period="5d")['Close']
    
    return (
        data[[tickers["kospi"]]], data[[tickers["sp500"]]], data[[tickers["fx"]]], 
        data[[tickers["us10y"]]], data[[tickers["us2y"]]], data[[tickers["vix"]]], 
        data[[tickers["copper"]]], data[[tickers["freight"]]], data[[tickers["wti"]]], 
        data[[tickers["dxy"]]], sector_raw, sector_tickers
    )

@st.cache_data(ttl=1800)
def get_market_news():
    api_url = "https://newsapi.org/v2/everything"
    params = {"q": "stock market risk OR recession OR inflation", "sortBy": "publishedAt", "language": "en", "pageSize": 5, "apiKey": NEWS_API_KEY}
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

@st.cache_data(ttl=10) 
def load_board_data():
    try:
        res = requests.get(f"{GSHEET_CSV_URL}&cache_bust={datetime.now().timestamp()}", timeout=10)
        res.encoding = 'utf-8' 
        if res.status_code == 200:
            df = pd.read_csv(StringIO(res.text), dtype=str).fillna("")
            return df.to_dict('records')
        return []
    except:
        return []

def save_to_gsheet(date, author, content, password, action="append"):
    try:
        payload = {"date": str(date), "author": str(author), "content": str(content), "password": str(password), "action": action}
        res = requests.post(GSHEET_WEBAPP_URL, data=json.dumps(payload), timeout=15)
        if res.status_code == 200:
            st.cache_data.clear()
            return True
        return False
    except Exception as e:
        st.error(f"연동 에러: {e}")
        return False

try:
    with st.spinner('시차 상관관계 및 가중치 분석 중...'):
        kospi, sp500, fx, bond10, bond2, vix_data, copper_data, freight_data, wti_data, dxy_data, sector_raw, sector_map = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series(dtype='float64')
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        # 타임존 제거 및 중복 제거 (가장 최신 값 유지)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[~df.index.duplicated(keep='last')]

    ks_s = get_clean_series(kospi).ffill()
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

    total_w = w_macro + w_tech + w_global + w_fear
    if total_w == 0: st.error("가중치 합이 0일 수 없습니다."); st.stop()

    def calculate_score(current_series, full_series, inverse=False):
        recent = full_series.tail(252)
        min_v, max_v = float(recent.min()), float(recent.max()); curr_v = float(current_series.iloc[-1])
        if max_v == min_v: return 50.0
        return float(max(0, min(100, ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100)))

    m_now = (calculate_score(fx_s, fx_s) + calculate_score(b10_s, b10_s) + calculate_score(cp_s, cp_s, True)) / 3
    t_now = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    total_risk_index = (m_now * w_macro + t_now * w_tech + calculate_score(sp_s, sp_s, True) * w_global + calculate_score(vx_s, vx_s) * w_fear) / total_w

    c_gauge, c_guide = st.columns([1, 1.6])
    with c_guide: 
        st.markdown('<p class="guide-header">💡 지수를 더 똑똑하게 보는 법</p>', unsafe_allow_html=True)
    with c_gauge: 
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=total_risk_index, title={'text': "주식 시장 위험 지수"}))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 7. 백테스팅 (최근 1년 tail 강제 지정)
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅")
    # ks_s 인덱스를 타임존 없는 KST 날짜로 정렬하여 최신 252일 확보
    dates = ks_s.tail(252).index
    hist_risks = []
    for d in dates:
        m = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
        hist_risks.append((m * w_macro + max(0, min(100, 100 - (float(ks_s.loc[d]) / float(ma20.loc[d]) - 0.9) * 500)) * w_tech + get_hist_score_val(sp_s, d, True) * w_global + get_hist_score_val(vx_s, d) * w_fear) / total_w)
    hist_df = pd.DataFrame({'Date': dates, 'Risk': hist_risks, 'KOSPI': ks_s.loc[dates].values})
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Risk'], name="위험 지수", line=dict(color='red')))
    fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot')))
    fig_bt.update_layout(yaxis2=dict(overlaying="y", side="right"), height=400)
    st.plotly_chart(fig_bt, use_container_width=True)

    # 9. 지표별 상세 분석 (가장 최신 30개 데이터 tail(30) 사용)
    st.markdown("---")
    st.subheader("🔍 주요 상관관계 지표 분석")
    
    def create_chart(series, title, threshold, desc_text):
        # 개별 지표도 최신 252일치만 표시하여 로딩 속도 최적화 및 최신성 확보
        sub_s = series.tail(252)
        fig = go.Figure(go.Scatter(x=sub_s.index, y=sub_s.values, name=title))
        fig.add_hline(y=threshold, line_width=2, line_color="red")
        return fig

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        st.plotly_chart(create_chart(sp_s, "S&P 500", sp_s.tail(252).mean()*0.9, ""), use_container_width=True)
    with r1_c2:
        fx_th = float(fx_s.tail(252).mean() * 1.02)
        st.plotly_chart(create_chart(fx_s, "원/달러 환율", fx_th, ""), use_container_width=True)
    with r1_c3:
        st.plotly_chart(create_chart(cp_s, "Copper", cp_s.tail(252).mean()*0.9, ""), use_container_width=True)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        st.plotly_chart(create_chart(yield_curve, "금리차", 0.0, ""), use_container_width=True)
    with r2_c2:
        st.subheader("KOSPI 기술적 분석")
        # 수정 핵심: last('30D') 대신 무조건 마지막 30개 로우(Row)를 가져옴
        ks_recent = ks_s.tail(30)
        ma20_recent = ma20.reindex(ks_recent.index).ffill()
        
        fig_ks = go.Figure()
        # 선 굵기 강화 및 데이터 포인트 표시
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ks_recent.values, name="현재가", line=dict(color='royalblue', width=4), mode='lines+markers'))
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ma20_recent.values, name="20일선", line=dict(color='orange', width=2, dash='dot')))
        
        # 오늘 날짜와 가격을 그래프 위에 텍스트로 표시
        fig_ks.add_annotation(
            x=ks_recent.index[-1], y=ks_recent.values[-1],
            text=f"오늘: {ks_recent.values[-1]:.2f}",
            showarrow=True, arrowhead=1, ax=0, ay=-40,
            bgcolor="royalblue", font=dict(color="white")
        )
        
        fig_ks.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_ks, use_container_width=True)
        st.info(f"최종 데이터 포인트: {ks_recent.index[-1].strftime('%Y-%m-%d')}")
        
    with r2_c3:
        st.plotly_chart(create_chart(vx_s, "VIX", 30, ""), use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {get_kst_now().strftime('%d일 %H시 %M분')} | KOSPI 데이터 실시간 반영 모드")
