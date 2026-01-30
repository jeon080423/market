import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

# 1. 페이지 설정
st.set_page_config(page_title="주식 시장 하락 전조 신호 모니터링", layout="wide")

# 자동 새로고침 설정
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=600000, key="datarefresh")
except ImportError:
    pass

# 2. 고정 NewsAPI Key 설정
NEWS_API_KEY = "13cfedc9823541c488732fb27b02fa25"

# 3. 제목 및 설명
st.title("📊 종합 시장 위험 지수(Total Market Risk Index) 모니터링")
st.markdown(f"""
이 대시보드는 상관관계 분석을 통해 **환율(40%), 글로벌(30%), 공포(20%), 기술(10%)** 비중으로 위험 지수를 산출합니다.
(마지막 업데이트: {datetime.now().strftime('%H:%M:%S')})
""")

# 4. 사이드바 - 가중치 설정 (분석 기반 최적 비중으로 기본값 세팅)
st.sidebar.header("⚙️ 지표별 가중치 설정")
w_macro = st.sidebar.slider("매크로 (환율/금리/물동량)", 0.0, 1.0, 0.4, 0.1)
w_global = st.sidebar.slider("글로벌 시장 위험 (미국/일본)", 0.0, 1.0, 0.3, 0.1)
w_fear = st.sidebar.slider("시장 공포 (VIX 지수)", 0.0, 1.0, 0.2, 0.1)
w_tech = st.sidebar.slider("국내 기술적 지표 (이동평균선)", 0.0, 1.0, 0.1, 0.1)

total_w = w_macro + w_tech + w_global + w_fear
if total_w == 0:
    st.error("가중치의 합이 0일 수 없습니다.")
    st.stop()

# 5. 데이터 수집 함수
@st.cache_data(ttl=600)
def load_data():
    end_date = datetime.now()
    start_date = "2019-01-01"
    kospi = yf.download("^KS11", start=start_date, end=end_date)
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)
    nikkei = yf.download("^N225", start=start_date, end=end_date)
    exchange_rate = yf.download("KRW=X", start=start_date, end=end_date)
    us_10y = yf.download("^TNX", start=start_date, end=end_date)
    us_2y = yf.download("^IRX", start=start_date, end=end_date)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    copper = yf.download("HG=F", start=start_date, end=end_date) 
    return kospi, sp500, nikkei, exchange_rate, us_10y, us_2y, vix, copper

# 6. 리포트 및 뉴스 함수
def get_analyst_reports():
    url = "http://consensus.hankyung.com/apps.analysis/analysis.list?skinType=business"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        reports = []
        for row in soup.select("tr")[1:11]:
            titles = row.select(".text_l a")
            if titles:
                d = row.select("td")
                reports.append({"제목": titles[0].get_text().strip(), "종목": d[1].get_text().strip(), "출처": f"{d[4].get_text().strip()}({d[3].get_text().strip()})"})
        return reports
    except: return []

@st.cache_data(ttl=600)
def get_market_news():
    url = f"https://newsapi.org/v2/everything?q=stock+market+risk&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        articles = requests.get(url, timeout=10).json().get('articles', [])[:5]
        return [{"title": a['title'], "link": a['url']} for a in articles]
    except: return []

try:
    with st.spinner('최적 비중 기반 리스크 분석 중...'):
        kospi, sp500, nikkei, fx, bond10, bond2, vix_data, copper_data = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series()
        if isinstance(df.columns, pd.MultiIndex): return df['Close'].iloc[:, 0]
        return df['Close']

    ks_s, sp_s, nk_s = get_clean_series(kospi), get_clean_series(sp500), get_clean_series(nikkei)
    fx_s, b10_s, b2_s, vx_s = get_clean_series(fx), get_clean_series(bond10), get_clean_series(bond2), get_clean_series(vix_data)
    cp_s = get_clean_series(copper_data)
    
    yield_curve = b10_s - b2_s
    ma20 = ks_s.rolling(window=20).mean()

    def calculate_score(current_series, full_series, inverse=False):
        recent = full_series.last('365D')
        if recent.empty: return 50.0
        min_v, max_v = float(recent.min()), float(recent.max())
        curr_v = float(current_series.iloc[-1])
        if max_v == min_v: return 0.0
        return float(max(0, min(100, ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100)))

    # 위험 점수 계산
    score_sp = calculate_score(sp_s, sp_s, inverse=True)
    score_nk = calculate_score(nk_s, nk_s, inverse=True)
    global_risk_score = (score_sp * 0.6) + (score_nk * 0.4)

    score_fx = calculate_score(fx_s, fx_s)
    score_bond = calculate_score(b10_s, b10_s)
    score_cp = calculate_score(cp_s, cp_s, inverse=True)
    macro_score = (score_fx + score_bond + score_cp) / 3
    
    tech_score = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    fear_score = calculate_score(vx_s, vx_s)

    total_risk_index = float((macro_score * w_macro + tech_score * w_tech + global_risk_score * w_global + fear_score * w_fear) / total_w)

    # 7. 메인 게이지
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = total_risk_index,
        title = {'text': "종합 시장 위험 지수", 'font': {'size': 24}},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                 'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 60], 'color': "yellow"},
                           {'range': [60, 80], 'color': "orange"}, {'range': [80, 100], 'color': "red"}]}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 8. 뉴스 및 보고서 가로 배치
    st.markdown("---")
    c_news, c_report = st.columns(2)
    with c_news:
        st.subheader("📰 글로벌 마켓 리스크 뉴스")
        for n in get_market_news(): st.markdown(f"- [{n['title']}]({n['link']})")
    with c_report:
        st.subheader("📝 실시간 애널리스트 보고서")
        st.dataframe(pd.DataFrame(get_analyst_reports()), use_container_width=True, hide_index=True)

    # 9. 지표별 상세 분석 (3열 배치)
    st.markdown("---")
    st.subheader("🔍 실물 경제 및 주요 상관관계 지표 분석")
    
    def create_chart(series, title, threshold, mode='above', desc=""):
        if series.empty: return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=title))
        fig.add_hline(y=threshold, line_width=2, line_color="red")
        fig.add_annotation(x=series.index[len(series)//2], y=threshold, text=desc, showarrow=False, font=dict(color="red"), bgcolor="white", yshift=10)
        fig.add_vline(x="2020-03-19", line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=300)
        return fig

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        st.plotly_chart(create_chart(sp_s, "미국 S&P 500 (영향력 60%)", sp_s.last('365D').mean()*0.9, 'below', "평균 대비 -10% 하락 시"), use_container_width=True)
        st.info("**미국 지수**: KOSPI와 가장 강한 정(+)의 상관성을 보입니다.")
    with r1_c2:
        st.plotly_chart(create_chart(fx_s, "원/달러 환율 추이", 1350, 'above', "1,350원 돌파 시 위험"), use_container_width=True)
        st.info("**환율**: 1,400원 이상 지속 시 외국인 자본 유출 위험이 매우 큽니다.")
    with r1_c3:
        st.plotly_chart(create_chart(cp_s, "실물 경기 지표 (Copper)", cp_s.last('365D').mean()*0.9, 'below', "수요 위축 시 위험"), use_container_width=True)
        st.info("**실물 경기**: 원자재 가격 하락은 글로벌 수요 둔화의 선행 신호입니다.")

    st.markdown("---")
    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        st.plotly_chart(create_chart(yield_curve, "장단기 금리차", 0.0, 'below', "0 이하 역전 시 위험"), use_container_width=True)
    with r2_c2:
        ks_recent, ma_recent = ks_s.last('30D'), ma20.last('30D')
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ks_recent.values, name="현재 주가"))
        fig_ks.add_trace(go.Scatter(x=ma_recent.index, y=ma_recent.values, name="20일 평균선", line=dict(dash='dot')))
        fig_ks.add_annotation(x=ks_recent.index[-1], y=ma_recent.iloc[-1], text="평균선 아래 추락 시 위험", showarrow=True, font=dict(color="red"))
        fig_ks.update_layout(title="KOSPI 최근 1개월 집중 분석", height=300)
        st.plotly_chart(fig_ks, use_container_width=True)
    with r2_c3:
        st.plotly_chart(create_chart(vx_s, "VIX 공포 지수", 30, 'above', "30 돌파 시 패닉"), use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 최적 가중치 분석 시스템 가동 중")
