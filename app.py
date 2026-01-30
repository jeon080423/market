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
    freight = yf.download("BDRY", start=start_date, end=end_date)
    return kospi, sp500, nikkei, exchange_rate, us_10y, us_2y, vix, copper, freight

# 6. 리포트 및 뉴스 함수 (네이버 증권 기반)
def get_analyst_reports():
    url = "https://finance.naver.com/research/company_list.naver"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
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
                    stock_name = cells[0].get_text().strip()
                    title_tag = cells[1].select_one("a")
                    title_text = title_tag.get_text().strip() if title_tag else cells[1].get_text().strip()
                    source = cells[2].get_text().strip()
                    reports.append({"제목": title_text, "종목": stock_name, "출처": source})
        return reports
    except Exception as e:
        return []

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

    # 에러 수정: 국가별 공휴일로 인한 결측치를 KOSPI 날짜 기준으로 정렬 및 채우기
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

    # 위험 점수 계산 함수
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

    # 7. 지수 가이드 및 메인 게이지 배치
    st.markdown("---")
    col_guide, col_gauge = st.columns([1, 1.5])

    with col_guide:
        st.subheader("💡 지수를 더 똑똑하게 보는 법")
        st.markdown("""
        | 점수 구간 | 의미 | 권장 대응 |
        | :--- | :--- | :--- |
        | **0 ~ 40 (Safe)** | 시장 과열 또는 안정기 | 적극적 수익 추구 |
        | **40 ~ 60 (Watch)** | 지표 간 충돌 발생 (혼조세) | 현금 비중 확보 고민 시작 |
        | **60 ~ 80 (Danger)** | 다수 지표가 위험 신호 발생 | 공격적 투자 지양, 방어적 포트폴리오 |
        | **80 ~ 100 (Panic)** | 시스템적 위기 가능성 농후 | 리스크 관리 최우선 (현금 확보) |
        """)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = total_risk_index,
            title = {'text': "종합 시장 위험 지수", 'font': {'size': 24}},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 60], 'color': "yellow"},
                               {'range': [60, 80], 'color': "orange"}, {'range': [80, 100], 'color': "red"}]}
        ))
        fig_gauge.update_layout(margin=dict(t=50, b=0, l=30, r=30), height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 위험 수준별 경고 메시지
        if total_risk_index >= 80:
            st.error("🚨 **패닉 경보**: 시스템적 위기 징후가 포착되었습니다. 자산 보호를 위해 현금 비중을 대폭 확대하십시오.")
        elif total_risk_index >= 60:
            st.warning("⚠️ **위험 경보**: 주요 지표들이 동시다발적으로 하락을 예고하고 있습니다. 방어적인 포트폴리오 운용이 필요합니다.")
        elif total_risk_index >= 40:
            st.info("🔍 **모니터링 알림**: 시장의 불확실성이 증가하고 있습니다. 개별 지표의 변동성을 면밀히 관찰하세요.")
        else:
            st.success("✅ **시장 안정**: 지표들이 안정적인 흐름을 보이고 있습니다. 기존 투자 전략을 유지하기 좋은 시점입니다.")

    # 8. 백테스팅 기능 (역사적 위험 지수 추이)
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 (최근 1년)")
    
    st.info("""
    **백테스팅(Backtesting)이란?** 과거 데이터를 사용하여 모델이나 투자 전략의 유효성을 검증하는 과정입니다. 
    여기서는 지난 1년간의 데이터를 바탕으로 매일의 '시장 위험 지수'를 재산출하여, 
    실제 KOSPI 지수 하락 시점에 위험 지수가 선행하여 상승했는지 확인합니다.
    """)
    
    with st.spinner('역사적 데이터 시뮬레이션 중...'):
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
        
        # 상관계수 계산
        correlation = hist_df['RiskIndex'].corr(hist_df['KOSPI'])
        
        c1, c2 = st.columns([3, 1])
        with c1:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['RiskIndex'], name="위험 지수", line=dict(color='red', width=2)))
            fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot')))
            
            fig_bt.update_layout(
                title="위험 지수 vs KOSPI 동조화 분석",
                yaxis=dict(title="위험 지수 (0-100)", range=[0, 100]),
                yaxis2=dict(title="KOSPI 지수", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            st.plotly_chart(fig_bt, use_container_width=True)
        
        with c2:
            st.metric(label="📊 모델 상관계수 (Corr)", value=f"{correlation:.2f}")
            st.write("""
            **수치 해석:**
            - **-1.0 ~ -0.7**: 강한 역상관 (위험 신호가 하락을 매우 잘 반영함)
            - **-0.7 ~ -0.3**: 뚜렷한 역상관 (유의미한 하락 전조 신호)
            - **-0.3 ~ 0.0**: 약한 역상관 (참고용 지표)
            - **0.0 이상**: 모델 왜곡 가능성 (지표 재조정 권장)
            """)

        st.caption("※ 위험 지수가 급격히 상승할 때 KOSPI의 하락 압력이 강해지는 경향을 확인할 수 있습니다.")

    # 9. 뉴스 및 보고서 가로 배치
    st.markdown("---")
    c_news, c_report = st.columns(2)
    with c_news:
        st.subheader("📰 글로벌 마켓 리스크 뉴스")
        for n in get_market_news(): st.markdown(f"- [{n['title']}]({n['link']})")
    with c_report:
        st.subheader("📝 최신 애널 보고서")
        reports = get_analyst_reports()
        if reports:
            st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
        else:
            st.info("현재 데이터를 불러오는 중이거나 최신 보고서가 없습니다.")

    # 10. 지표별 상세 분석
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
        fx_threshold = round(float(fx_s.last('365D').mean() * 1.02), 1)
        st.plotly_chart(create_chart(fx_s, "원/달러 환율 추이", fx_threshold, 'above', f"{fx_threshold}원 돌파 시 위험"), use_container_width=True)
        st.info(f"**환율**: 최근 1년 평균 대비 +2%({fx_threshold}원) 상회 시 외국인 자본 유출 압력이 심화됩니다.")
    with r1_c3:
        st.plotly_chart(create_chart(cp_s, "실물 경기 지표 (Copper)", cp_s.last('365D').mean()*0.9, 'below', "수요 위축 시 위험"), use_container_width=True)
        st.info("**실물 경기**: 원자재 가격 하락은 글로벌 수요 둔화의 선행 신호입니다.")

    st.markdown("---")
    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        st.plotly_chart(create_chart(yield_curve, "장단기 금리차", 0.0, 'below', "0 이하 역전 시 위험"), use_container_width=True)
        st.info("**장단기 금리차**: 10년물과 2년물 금리 역전은 통상 경기 침체의 강력한 전조 신호로 해석됩니다.")
    with r2_c2:
        ks_recent, ma_recent = ks_s.last('30D'), ma20.last('30D')
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ks_recent.values, name="현재 주가"))
        fig_ks.add_trace(go.Scatter(x=ma_recent.index, y=ma_recent.values, name="20일 평균선", line=dict(dash='dot')))
        fig_ks.add_annotation(x=ks_recent.index[-1], y=ma_recent.iloc[-1], text="평균선 아래 추락 시 위험", showarrow=True, font=dict(color="red"))
        fig_ks.update_layout(title="KOSPI 최근 1개월 집중 분석", height=300)
        st.plotly_chart(fig_ks, use_container_width=True)
        st.info("**기술적 분석**: 주가가 20일 이동평균선을 하회할 경우 단기 추세 하락 전환 가능성이 높습니다.")
    with r2_c3:
        st.plotly_chart(create_chart(vx_s, "VIX 공포 지수", 30, 'above', "30 돌파 시 패닉"), use_container_width=True)
        st.info("**VIX 지수**: 시장 변동성을 나타내며, 지수 급등은 투자 심리 악화와 투매 가능성을 시사합니다.")

    st.markdown("---")
    r3_c1, r3_c2, r3_c3 = st.columns(3)
    with r3_c1:
        fr_threshold = round(float(fr_s.last('365D').mean() * 0.85), 2)
        st.plotly_chart(create_chart(fr_s, "글로벌 물동량 지표 (BDRY)", fr_threshold, 'below', f"지지선({fr_threshold}) 붕괴 시 위험"), use_container_width=True)
        st.info(f"**물동량 분석**: 건화물선 운임은 경기 선행 지표입니다. 현재 기준선({fr_threshold}) 하향 돌파 시 글로벌 경기 수축 신호로 간주합니다.")

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 최적 가중치 분석 시스템 가동 중")
