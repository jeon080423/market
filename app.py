import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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

# 4. 사이드바 - 가중치 설정
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
    freight = yf.download("BDRY", start=start_date, end=end_date)
    return kospi, sp500, nikkei, exchange_rate, us_10y, us_2y, vix, copper, freight

# 6. 리포트 및 뉴스 함수
def get_analyst_reports():
    url = "https://finance.naver.com/research/company_list.naver"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'euc-kr' 
        soup = BeautifulSoup(res.text, 'html.parser')
        reports = []
        table = soup.select_one("table.type_1")
        if not table: return []
        rows = table.select("tr")
        for row in rows:
            if len(reports) >= 10: break
            stock_td = row.select_one("td.alpha")
            if stock_td:
                cells = row.select("td")
                if len(cells) >= 3:
                    reports.append({
                        "제목": cells[1].get_text().strip(),
                        "종목": cells[0].get_text().strip(),
                        "출처": cells[2].get_text().strip()
                    })
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
        kospi, sp500, nikkei, fx, bond10, bond2, vix_data, copper_data, freight_data = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series()
        if isinstance(df.columns, pd.MultiIndex): return df['Close'].iloc[:, 0]
        return df['Close']

    ks_s, sp_s, nk_s = get_clean_series(kospi), get_clean_series(sp500), get_clean_series(nikkei)
    fx_s, b10_s, b2_s, vx_s = get_clean_series(fx), get_clean_series(bond10), get_clean_series(bond2), get_clean_series(vix_data)
    cp_s, fr_s = get_clean_series(copper_data), get_clean_series(freight_data)

    # 데이터 정렬 및 결측치 처리
    sp_s = sp_s.reindex(ks_s.index).ffill()
    nk_s = nk_s.reindex(ks_s.index).ffill()
    fx_s = fx_s.reindex(ks_s.index).ffill()
    b10_s = b10_s.reindex(ks_s.index).ffill()
    b2_s = b2_s.reindex(ks_s.index).ffill()
    vx_s = vx_s.reindex(ks_s.index).ffill()
    cp_s = cp_s.reindex(ks_s.index).ffill()
    fr_s = fr_s.reindex(ks_s.index).ffill()
    
    yield_curve = b10_s - b2_s
    ma20 = ks_s.rolling(window=20).mean()

    def calculate_score(current_series, full_series, inverse=False):
        recent = full_series.last('365D')
        if recent.empty: return 50.0
        min_v, max_v = float(recent.min()), float(recent.max())
        curr_v = float(current_series.iloc[-1])
        if max_v == min_v: return 0.0
        return float(max(0, min(100, ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100)))

    # 현재 위험 점수 계산
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

    # 7. 레이아웃: 가이드 및 게이지
    st.markdown("---")
    col_guide, col_gauge = st.columns([1, 1.5])
    with col_guide:
        st.subheader("💡 지수를 더 똑똑하게 보는 법")
        st.markdown("""
        | 점수 구간 | 의미 | 권장 대응 |
        | :--- | :--- | :--- |
        | **0 ~ 40 (Safe)** | 시장 과열 또는 안정기 | 적극적 수익 추구 |
        | **40 ~ 60 (Watch)** | 지표 간 충돌 발생 | 현금 비중 확보 고민 |
        | **60 ~ 80 (Danger)** | 다수 위험 신호 발생 | 방어적 포트폴리오 |
        | **80 ~ 100 (Panic)** | 시스템적 위기 가능성 | 리스크 관리 최우선 |
        """)
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = total_risk_index,
            title = {'text': "종합 시장 위험 지수"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 60], 'color': "yellow"},
                               {'range': [60, 80], 'color': "orange"}, {'range': [80, 100], 'color': "red"}]}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        if total_risk_index >= 60: st.warning("⚠️ 시장 리스크가 높습니다. 지표를 면밀히 관찰하세요.")
        else: st.success("✅ 현재 시장 지표는 대체로 안정적입니다.")

    # 8. 백테스팅 및 설명력 산출
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 및 회귀 분석")
    
    with st.spinner('설명력(R-squared) 산출 중...'):
        lookback = 252
        dates = ks_s.index[-lookback:]
        
        def get_hist_score(series, current_idx, inverse=False):
            sub = series.loc[:current_idx].iloc[-252:]
            if len(sub) < 10: return 50.0
            min_v, max_v = sub.min(), sub.max()
            curr_v = series.loc[current_idx]
            if max_v == min_v: return 0.0
            return ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100

        hist_risks = []
        for d in dates:
            s_sp = get_hist_score(sp_s, d, True)
            s_nk = get_hist_score(nk_s, d, True)
            g_risk = (s_sp * 0.6) + (s_nk * 0.4)
            s_fx = get_hist_score(fx_s, d)
            s_bn = get_hist_score(b10_s, d)
            s_cp = get_hist_score(cp_s, d, True)
            m_score = (s_fx + s_bn + s_cp) / 3
            t_score = max(0, min(100, 100 - (ks_s.loc[d] / ma20.loc[d] - 0.9) * 500))
            f_score = get_hist_score(vx_s, d)
            total_h = (m_score * w_macro + t_score * w_tech + g_risk * w_global + f_score * w_fear) / total_w
            hist_risks.append(total_h)

        hist_df = pd.DataFrame({'Date': dates, 'RiskIndex': hist_risks, 'KOSPI': ks_s.loc[dates].values})
        
        # 상관계수 및 결정계수(R^2) 산출
        corr = hist_df['RiskIndex'].corr(hist_df['KOSPI'])
        r_squared = corr**2  # 단순 선형 회귀에서 R^2는 상관계수의 제곱과 같음

        c1, c2 = st.columns([3, 1])
        with c1:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['RiskIndex'], name="위험 지수", line=dict(color='red')))
            fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot')))
            fig_bt.update_layout(yaxis=dict(title="위험 지수", range=[0, 100]), yaxis2=dict(title="KOSPI", overlaying="y", side="right"), height=400)
            st.plotly_chart(fig_bt, use_container_width=True)
        with c2:
            st.metric("회귀 분석 설명력 (R²)", f"{r_squared*100:.1f}%")
            st.metric("상관계수 (Corr)", f"{corr:.2f}")
            st.write(f"""
            현재 모델은 KOSPI 변동의 **{r_squared*100:.1f}%**를 설명하고 있습니다. 
            설명력이 낮다면 지수가 주가보다 선행하고 있거나, 비선형적인 관계임을 의미합니다.
            """)

    # 9. 뉴스 및 보고서
    st.markdown("---")
    c_news, c_report = st.columns(2)
    with c_news:
        st.subheader("📰 마켓 뉴스"); [st.markdown(f"- [{n['title']}]({n['link']})") for n in get_market_news()]
    with c_report:
        st.subheader("📝 최신 보고서"); st.dataframe(pd.DataFrame(get_analyst_reports()), use_container_width=True, hide_index=True)

    # 10. 지표별 상세 분석
    st.markdown("---")
    st.subheader("🔍 세부 지표 분석")
    def create_chart(series, title, threshold, desc=""):
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, name=title))
        fig.add_hline(y=threshold, line_width=2, line_color="red")
        fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    r1_c1.plotly_chart(create_chart(sp_s, "S&P 500", sp_s.last('365D').mean()*0.9), use_container_width=True)
    fx_th = float(fx_s.last('365D').mean() * 1.02)
    r1_c2.plotly_chart(create_chart(fx_s, "원/달러 환율", fx_th), use_container_width=True)
    r1_c3.plotly_chart(create_chart(cp_s, "Copper (구리)", cp_s.last('365D').mean()*0.9), use_container_width=True)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    r2_c1.plotly_chart(create_chart(yield_curve, "장단기 금리차", 0.0), use_container_width=True)
    r2_c2.plotly_chart(create_chart(ks_s.last('30D'), "KOSPI (1개월)", ma20.iloc[-1]), use_container_width=True)
    r2_c3.plotly_chart(create_chart(vx_s, "VIX 지수", 30), use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
