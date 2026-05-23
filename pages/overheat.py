import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
import time

def render_overheat_page():
    st.title("🔥 시장 과열 국면 시그널 (김효진 박사)")

    # Data fetching
    @st.cache_data(ttl=3600)
    def get_overheat_data():
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        tickers_map = {
            '069500.KS': '069500', # KODEX 200
            '252650.KS': '252650', # KODEX 200 동일가중
            '005930.KS': '005930', # 삼성전자
            '000660.KS': '000660', # SK하이닉스
            '091180.KS': '091180', # KODEX 자동차
            '091220.KS': '091220', # KODEX 은행
            '261240.KS': '261240', # KODEX 바이오
            '091210.KS': '091210', # KODEX 건설
            '228790.KS': '228790', # TIGER 화장품
            '450320.KS': '450320'  # PLUS K방산
        }
        
        try:
            df_list = []
            for col_name, fdr_ticker in tickers_map.items():
                try:
                    temp_df = fdr.DataReader(fdr_ticker, start_date, end_date)
                    if not temp_df.empty:
                        temp_df = temp_df[['Close']].rename(columns={'Close': col_name})
                        df_list.append(temp_df)
                except Exception:
                    pass
                    
            try:
                # Add US Indicators: ^TNX (10Y), CL=F (Oil), HYG (High Yield), IPO (Renaissance IPO ETF), ^GSPC (S&P 500)
                us_tickers = {'^TNX': '^TNX', 'CL=F': 'CL=F', 'HYG': 'HYG', 'IPO': 'IPO', '^GSPC': '^GSPC'}
                for t_name, t_sym in us_tickers.items():
                    raw = yf.download(t_sym, start=start_date, end=end_date)
                    if isinstance(raw.columns, pd.MultiIndex) and 'Close' in raw.columns.levels[0]:
                        data = raw['Close'].iloc[:, 0]
                    elif 'Close' in raw.columns:
                        data = raw['Close']
                    else:
                        data = raw.iloc[:, 0]
                    if isinstance(data, pd.DataFrame): data = data.iloc[:, 0]
                    df_list.append(pd.DataFrame({t_name: data}))
            except Exception:
                pass

            if df_list:
                df = pd.concat(df_list, axis=1)
                all_cols = list(tickers_map.keys()) + ['^TNX', 'CL=F', 'HYG', 'IPO', '^GSPC']
                for col in all_cols:
                    if col not in df.columns:
                        df[col] = pd.NA
                return df
            else:
                return pd.DataFrame(columns=list(tickers_map.keys()) + ['^TNX', 'CL=F', 'HYG', 'IPO', '^GSPC'])
        except Exception as e:
            st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
            return pd.DataFrame()

    with st.spinner("데이터를 불러오는 중입니다..."):
        df = get_overheat_data()

    if df.empty:
        st.warning("데이터를 불러오지 못했습니다.\n\n잠시 후 다시 시도해주세요.")
        return

    # 최근 6개월(180일) 기준 데이터 정규화
    try:
        cutoff_date = df.index.max() - pd.Timedelta(days=180)
        df_6m = df[df.index >= cutoff_date].ffill().dropna(how='all')
    except Exception:
        df_6m = pd.DataFrame()

    if df_6m.empty:
        df_norm = df.ffill().dropna(how='all')
        if not df_norm.empty:
            df_norm = (df_norm / df_norm.iloc[0]) * 100
    else:
        df_norm = (df_6m / df_6m.iloc[0]) * 100

    if df_norm.empty:
        st.warning("분석할 충분한 데이터가 없습니다.")
        return

    st.header("🌡️ 시장 과열 지수 (Market Overheat Index, 0~100)")
    st.markdown("4가지 핵심 리스크 시그널을 **각각 0~100점으로 정규화(Z-Score + Sigmoid)**한 후 가중 평균하여 산출한 단일 복합 과열 지수입니다. 50점이 중립, 70점 이상이면 과열 경보, 85점 이상이면 위험 구간입니다.")

    # ─────────────────────────────────────────────────────────────────
    # [복합 과열 지수 계산 엔진] Z-Score + Sigmoid 0~100 정규화 방식
    # 각 시그널의 단위가 달라도 비교 가능한 0~100 스코어로 변환 후 가중 평균
    # ─────────────────────────────────────────────────────────────────
    import numpy as np

    def sigmoid_score(series, inverse=False):
        """시계열 데이터를 Z-Score → Sigmoid 변환으로 0~100점 정규화"""
        if series is None or len(series.dropna()) < 5:
            return pd.Series(50.0, index=series.index if series is not None else [])
        mu = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(50.0, index=series.index)
        z = (series - mu) / std
        score = 100 / (1 + np.exp(-z))
        return (100 - score) if inverse else score

    # ── 시그널 1: 주도주 쏠림도 (높을수록 과열) ──
    kodex_mkt  = df_norm.get('069500.KS', pd.Series(dtype=float))
    kodex_eq   = df_norm.get('252650.KS', pd.Series(dtype=float))
    if not kodex_mkt.empty and not kodex_eq.empty:
        spread_comp = (kodex_mkt - kodex_eq).reindex(df_norm.index).ffill()
    else:
        spread_comp = pd.Series(0.0, index=df_norm.index)
    s1_raw = sigmoid_score(spread_comp)  # 스프레드 클수록 고점수 = 과열

    # ── 시그널 2: 채권 자경단 (금리+유가 동반 상승, 높을수록 위험) ──
    tnx  = df_norm.get('^TNX', pd.Series(dtype=float)).reindex(df_norm.index).ffill()
    wti  = df_norm.get('CL=F', pd.Series(dtype=float)).reindex(df_norm.index).ffill()
    if not tnx.empty and not wti.empty:
        spread_vigil = ((tnx - 100) + (wti - 100)).reindex(df_norm.index).ffill()
    elif not tnx.empty:
        spread_vigil = (tnx - 100).reindex(df_norm.index).ffill()
    else:
        spread_vigil = pd.Series(0.0, index=df_norm.index)
    s2_raw = sigmoid_score(spread_vigil)

    # ── 시그널 3: 크레딧 스트레스 (HYG 하락 = 위험, inverse) ──
    hyg = df_norm.get('HYG', pd.Series(dtype=float)).reindex(df_norm.index).ffill()
    if not hyg.empty:
        s3_raw = sigmoid_score(hyg, inverse=True)   # HYG 낮을수록 위험 → 역방향
    else:
        s3_raw = pd.Series(50.0, index=df_norm.index)

    # ── 시그널 4: 투기적 과열 (IPO ETF ↑ > S&P500, 높을수록 과열) ──
    ipo  = df_norm.get('IPO',  pd.Series(dtype=float)).reindex(df_norm.index).ffill()
    gspc = df_norm.get('^GSPC', pd.Series(dtype=float)).reindex(df_norm.index).ffill()
    if not ipo.empty and not gspc.empty:
        spread_spec = (ipo - gspc).reindex(df_norm.index).ffill()
    else:
        spread_spec = pd.Series(0.0, index=df_norm.index)
    s4_raw = sigmoid_score(spread_spec)

    # ── 가중 평균 합산 (동일 가중치 25%씩) ──
    W1, W2, W3, W4 = 0.25, 0.25, 0.25, 0.25
    df_overheat = pd.DataFrame({
        '쏠림_리스크':    s1_raw * W1,
        '채권자경단':     s2_raw * W2,
        '크레딧_스트레스': s3_raw * W3,
        '투기적_과열':    s4_raw * W4,
    }, index=df_norm.index)
    df_overheat['과열_지수'] = df_overheat.sum(axis=1)

    latest_idx  = df_overheat['과열_지수'].iloc[-1]
    prev_idx    = df_overheat['과열_지수'].iloc[-6] if len(df_overheat) > 5 else 50.0  # 5일 전 대비
    delta_5d    = latest_idx - prev_idx

    # 등급 판정
    if latest_idx >= 85:
        grade, grade_color, grade_emoji = "위험 (DANGER)", "#c0392b", "🚨"
    elif latest_idx >= 70:
        grade, grade_color, grade_emoji = "과열 (CAUTION)", "#e67e22", "⚠️"
    elif latest_idx >= 55:
        grade, grade_color, grade_emoji = "주의 (WATCH)", "#f1c40f", "👀"
    elif latest_idx >= 35:
        grade, grade_color, grade_emoji = "중립 (NEUTRAL)", "#27ae60", "✅"
    else:
        grade, grade_color, grade_emoji = "냉각 (COOL)", "#2980b9", "❄️"

    # ─── 레이아웃: 게이지 + 카드 ───
    col_g, col_c = st.columns([1.2, 0.8])

    with col_g:
        # 반원형 게이지
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_idx,
            number={'suffix': "점", 'font': {'size': 38}},
            delta={'reference': prev_idx, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#2ecc71"},
                   'suffix': "pt (5일전 대비)", 'font': {'size': 14}},
            title={'text': f"시장 과열 지수 (MOI)  {grade_emoji} {grade}", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#555",
                         'tickvals': [0, 35, 55, 70, 85, 100],
                         'ticktext': ['0', '35', '55', '70', '85', '100']},
                'bar': {'color': grade_color, 'thickness': 0.25},
                'bgcolor': "#f8f9fa",
                'borderwidth': 1,
                'bordercolor': "#dee2e6",
                'steps': [
                    {'range': [0,  35], 'color': '#d6eaf8'},   # 냉각 - 파랑
                    {'range': [35, 55], 'color': '#d5f5e3'},   # 중립 - 초록
                    {'range': [55, 70], 'color': '#fef9e7'},   # 주의 - 노랑
                    {'range': [70, 85], 'color': '#fdebd0'},   # 과열 - 주황
                    {'range': [85,100], 'color': '#fadbd8'},   # 위험 - 빨강
                ],
                'threshold': {
                    'line': {'color': grade_color, 'width': 5},
                    'thickness': 0.8,
                    'value': latest_idx
                }
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_c:
        # 4개 컴포넌트 점수 카드
        st.markdown("**📊 컴포넌트별 현재 점수**")
        comp_data = {
            '🏦 주도주 쏠림': s1_raw.iloc[-1],
            '📈 채권 자경단': s2_raw.iloc[-1],
            '💳 크레딧 스트레스': s3_raw.iloc[-1],
            '🚀 투기적 과열': s4_raw.iloc[-1],
        }
        for label, score in comp_data.items():
            bar_color = "#e74c3c" if score >= 70 else ("#f39c12" if score >= 55 else "#27ae60")
            bar_width = int(score)
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span style="font-size:0.85rem;">{label}</span>
                    <span style="font-size:0.85rem; font-weight:bold; color:{bar_color};">{score:.1f}점</span>
                </div>
                <div style="background:#e9ecef; border-radius:4px; height:10px;">
                    <div style="background:{bar_color}; width:{bar_width}%; border-radius:4px; height:10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 해석 기준표
        st.markdown("""
        <div style="font-size:0.78rem; color:#666; margin-top:12px; line-height:1.7;">
        <b>📋 등급 기준</b><br>
        ❄️ &nbsp;0~35점: 냉각 — 매수 유리<br>
        ✅ 35~55점: 중립 — 관망<br>
        👀 55~70점: 주의 — 비중 축소 검토<br>
        ⚠️ 70~85점: 과열 — 방어적 운용<br>
        🚨 85~100점: 위험 — 최소화
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('''
    신영증권 김효진 애널리스트가 제시한 강세장에서 반드시 점검해야 할 **4가지 핵심 체크포인트**를 모니터링합니다.\n\n이러한 리스크들이 무르익지 않으면 추세가 곧장 꺾이지는 않겠지만, 임계점을 넘는 사건이 발생하면 시장이 빠르게 반전될 수 있으므로 사이드미러와 백미러를 함께 확인하는 신중한 투자가 필요한 시점입니다.\n\n''')

    with st.expander("💡 차트 데이터 표준화(Base 100) 및 읽는 법 안내"):
        st.markdown('''
        본 페이지의 모든 차트는 수만 원대의 주식, 80달러대의 유가, 4%대의 국채 금리 등 **서로 단위가 완전히 다른 지표들을 한 화면에서 직관적으로 비교**하기 위해 **'Base 100' 표준화 기법**을 적용했습니다.\n\n* **Base 100 작동 원리**: 조회 기간(최근 6개월)의 첫날 가격을 무조건 '100'으로 환산하여 출발합니다.\n\n* **해석 방법**: 만약 삼성전자의 선이 '110'에 있고 유가 선이 '90'에 있다면, 이는 6개월 전 대비 삼성전자는 정확히 10% 상승했고 유가는 10% 하락했음을 의미합니다.\n\n* **비교의 타당성**: Z-Score(표준편차) 방식과 달리 누적 수익률(성장률)을 그대로 반영하므로, **"주도주로 얼마나 자금이 쏠리고 있는지(압착)", "시장 지표들이 상대적으로 얼마나 과열되었는지"**를 1:1로 비교하는 데 가장 타당하고 직관적인 방식입니다.\n\n차트의 Y축 수치는 단순한 가격이 아닌 **'누적 수익률'**로 읽어주시면 됩니다.\n\n''')

    st.markdown("---")

    # ─── 6개월 추세 (4개 시그널 + 종합 인덱스) ───
    st.markdown("**📉 과열 지수 6개월 추세 (시그널별 기여도 분해)**")

    fig_trend = go.Figure()

    # 음영 배경 영역 (위험 존)
    fig_trend.add_hrect(y0=85, y1=100, fillcolor="rgba(192,57,43,0.10)", line_width=0, annotation_text="위험", annotation_position="right")
    fig_trend.add_hrect(y0=70,  y1=85,  fillcolor="rgba(230,126,34,0.10)",  line_width=0, annotation_text="과열", annotation_position="right")
    fig_trend.add_hrect(y0=55,  y1=70,  fillcolor="rgba(241,196,15,0.10)",  line_width=0, annotation_text="주의", annotation_position="right")
    fig_trend.add_hrect(y0=35,  y1=55,  fillcolor="rgba(39,174,96,0.10)",   line_width=0, annotation_text="중립", annotation_position="right")
    fig_trend.add_hrect(y0=0,   y1=35,  fillcolor="rgba(41,128,185,0.10)",  line_width=0, annotation_text="냉각", annotation_position="right")

    # 컴포넌트 기여도 (스택 에리어)
    fig_trend.add_trace(go.Scatter(
        x=df_overheat.index, y=df_overheat['쏠림_리스크'],
        name='주도주 쏠림 (×0.25)', mode='lines',
        line=dict(color='#3498db', width=1, dash='dot'), opacity=0.6
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_overheat.index, y=df_overheat['채권자경단'],
        name='채권자경단 (×0.25)', mode='lines',
        line=dict(color='#e74c3c', width=1, dash='dot'), opacity=0.6
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_overheat.index, y=df_overheat['크레딧_스트레스'],
        name='크레딧 스트레스 (×0.25)', mode='lines',
        line=dict(color='#9b59b6', width=1, dash='dot'), opacity=0.6
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_overheat.index, y=df_overheat['투기적_과열'],
        name='투기적 과열 (×0.25)', mode='lines',
        line=dict(color='#f39c12', width=1, dash='dot'), opacity=0.6
    ))
    # 종합 지수 (굵은 선)
    fig_trend.add_trace(go.Scatter(
        x=df_overheat.index, y=df_overheat['과열_지수'],
        name='📊 종합 과열 지수', mode='lines',
        line=dict(color='#2c3e50', width=3), fill='tozeroy',
        fillcolor='rgba(44,62,80,0.08)'
    ))
    # 현재 수준 기준선
    fig_trend.add_hline(y=latest_idx, line_dash="dash", line_color=grade_color,
                        annotation_text=f"현재 {latest_idx:.1f}점", annotation_position="left")

    fig_trend.update_layout(
        height=380,
        margin=dict(l=20, r=80, t=30, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="%m월"),
        yaxis=dict(range=[0, 100], title="과열 지수 (0~100점)"),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.info(f"""💡 **과열 지수(MOI) 해석 가이드**
{grade_emoji} **현재 {latest_idx:.1f}점 ({grade})** | 5일 전 대비 **{delta_5d:+.1f}점**

각 시그널(주도주 쏠림·채권자경단·크레딧·투기자금)을 Z-Score+Sigmoid로 0~100점 정규화 후 균등 가중(25%)으로 합산합니다. 
50점 = 과거 평균 수준, 70점 이상이면 과열 경보, 85점 이상이면 강력한 매도 대기 신호입니다.""")


    st.markdown("---")
    st.header("🤖 실시간 AI 진단 (김효진 박사 관점)")
    
    # AI API 설정
    gemini_key = None
    try:
        if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            gemini_key = st.secrets["gemini"]["api_key"]
        elif "gemini_api_key" in st.secrets:
            gemini_key = st.secrets["gemini_api_key"]
        elif "google_api_key" in st.secrets:
            gemini_key = st.secrets["google_api_key"]
            
        if gemini_key:
            genai.configure(api_key=gemini_key)
    except Exception:
        pass

    @st.cache_data(ttl=3600)
    def get_ai_overheat_analysis(data_text):
        if not gemini_key:
            return "⚠️ Gemini API 키가 설정되지 않아 AI 분석을 수행할 수 없습니다.\n\nSecrets 설정을 확인해주세요."
        
        prompt = f'''
당신은 신영증권 김효진 박사의 관점을 가진 수석 시장 분석가입니다.\n\n현재 코스피 등 증시가 3년 전 대비 세 배 가까이 오른 강세장에서, 시장의 과열과 반전 리스크를 판단하기 위해 다음 4가지 핵심 체크포인트를 분석해야 합니다.\n\n1. 주도주의 압착 현상: 삼성전자, SK하이닉스 등 극소수 주도주만 상승하고 KODEX 200 동일가중 지수 및 비주도 섹터(자동차, 은행, 바이오 등)가 전반적으로 소외/하락하며 자금 쏠림이 극심해지는지 평가하세요.\n\n2. 채권 자경단의 출현: 유가(WTI) 상승과 동반된 미 국채 10년물 금리의 급등이 인플레이션 우려를 자극하고 증시에 금리 부담을 주고 있는지 평가하세요.\n\n3. 사모 크레딧 환매 리스크: 하이일드 ETF(HYG)의 수익률 둔화나 하락을 통해 수면 아래 사모/비우량 신용 시장의 자금 이탈 징후가 있는지 평가하세요.\n\n4. 대형 IPO와 위험 선호도 정점: 글로벌 벤치마크인 S&P 500과 비교하여 미국 IPO ETF가 비정상적으로 가파르게 급등하며 시장의 투기적 과열이 단기 정점에 달했는지 평가하세요.\n\n최근 6개월 데이터 요약 (최초일 Base 100 환산값):
{data_text}

지시사항 (CRITICAL RULES):
1. 당신의 응답은 **무조건** 아래의 출력 템플릿과 100% 동일한 양식으로만 작성되어야 합니다.\n\n2. 프롬프트 내용의 복사, "분석을 시작하겠습니다" 등의 서론, 인사말은 일절 금지합니다.\n\n3. 반드시 한국어로만 작성하세요.\n\n[출력 템플릿]
**[종합 분석 결과]**
(여기에 김효진 박사 관점의 전문가적 분석 내용 2~3문단 작성.\n\n마침표 뒤에는 반드시 줄바꿈 2번 적용할 것)

**[최종 결론]**
(여기에 시장 리스크에 대한 명확하고 단호한 최종 결론 1문단 작성)
'''
        models = [
            "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
            "gemini-3.1-flash", "gemini-3.1-pro", "gemma-4-31b-it", "gemini-pro"
        ]
        for model_name in models:
            for attempt in range(2):
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content(prompt)
                    if res and hasattr(res, 'text'):
                        return res.text
                except Exception:
                    time.sleep(1)
        return "AI 분석을 가져오는 중 오류가 발생했습니다.\n\n잠시 후 다시 시도해주세요."
        
    if st.button("실시간 AI 진단 실행", use_container_width=True):
        with st.spinner("AI가 최근 6개월 시장 데이터를 바탕으로 과열 시그널을 분석 중입니다..."):
            recent_data = {}
            ticker_names = {
                '069500.KS': 'KODEX 200(시총가중)', '252650.KS': 'KODEX 200 동일가중', 
                '005930.KS': '삼성전자(주도주)', '000660.KS': 'SK하이닉스(주도주)', 
                '091180.KS': 'KODEX 자동차(비주도주)', '091220.KS': 'KODEX 은행(비주도주)', '261240.KS': 'KODEX 바이오(비주도주)',
                '091210.KS': 'KODEX 건설(비주도주)', '228790.KS': 'TIGER 화장품(비주도주)', '450320.KS': 'PLUS K방산(비주도주)',
                '^TNX': '미 국채 10년물 금리', 'CL=F': 'WTI 유가', 'HYG': '하이일드 ETF (사모/크레딧 대용)', 'IPO': 'IPO ETF (위험선호 대용)', '^GSPC': 'S&P 500'
            }
            for col, name in ticker_names.items():
                if col in df_norm.columns and not pd.isna(df_norm[col].iloc[-1]):
                    recent_data[name] = f"{df_norm[col].iloc[-1]:.2f} (Base 100)"
            
            data_text = "\n".join([f"- {k}: {v}" for k, v in recent_data.items()])
            analysis_result = get_ai_overheat_analysis(data_text)
            analysis_result = analysis_result.replace("```markdown", "").replace("```", "").strip()
            
            # AI가 지시를 무시하고 뱉은 영어 메타데이터, 서론 등을 강제 절단 (정규식 필터링)
            import re
            match = re.search(r'(\*\*\[종합 분석 결과\]\*\*|\[종합 분석 결과\])(.*)', analysis_result, re.DOTALL | re.IGNORECASE)
            if match:
                analysis_result = match.group(1) + match.group(2)
            
            # 모든 마침표 뒤에 강제 줄바꿈(엔터 2번) 추가하여 가독성 개선
            analysis_result = analysis_result.replace(". ", ".\n\n")
            
            st.success("분석 완료")
            with st.container(border=True):
                st.markdown(analysis_result)

    st.markdown("---")
    
    # 1. 주도주의 압착 현상
    st.header("1. 주도주의 압착 현상")
    st.markdown("소수 주도주로 자금이 지나치게 쏠리는 현상이 심화되는지 모니터링합니다.\n\n시총가중 지수(KODEX 200) 대비 동일가중 지수가 하락하거나, AI 관련주 외 나머지 비주도 섹터 종목들(자동차, 은행, 바이오, 건설, 뷰티, 방산 등)이 전반적으로 하락하면서 주도주만 독주하는 압착(Compression) 상황이 더 심화될 가능성을 경계해야 합니다.")
    
    fig1 = go.Figure()
    if '005930.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['005930.KS'], name="삼성전자", line=dict(color='#ff4b4b', width=2)))
    if '000660.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['000660.KS'], name="SK하이닉스", line=dict(color='#ff7f0e', width=2)))
    if '091180.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['091180.KS'], name="KODEX 자동차 (비주도주)", line=dict(color='#2ca02c', width=2, dash='dash')))
    if '091220.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['091220.KS'], name="KODEX 은행 (비주도주)", line=dict(color='#9467bd', width=2, dash='dash')))
    if '261240.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['261240.KS'], name="KODEX 바이오 (비주도주)", line=dict(color='#e377c2', width=2, dash='dash')))
    if '091210.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['091210.KS'], name="KODEX 건설 (비주도주)", line=dict(color='#8c564b', width=2, dash='dash')))
    if '228790.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['228790.KS'], name="TIGER 화장품 (비주도주)", line=dict(color='#bcbd22', width=2, dash='dash')))
    if '450320.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['450320.KS'], name="PLUS K방산 (비주도주)", line=dict(color='#17becf', width=2, dash='dash')))
    if '069500.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['069500.KS'], name="KODEX 200", line=dict(color='#1f77b4', width=2)))
    if '252650.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['252650.KS'], name="KODEX 200 동일가중", line=dict(color='#7f7f7f', dash='dash')))
    fig1.update_layout(
        title="주도주 압착 현상: AI 주도주 vs 후발주 및 동일가중 지수 (Base 100)", 
        height=450, 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="%m월")
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.info("💡 **차트 읽는 법:** 주도주(삼성전자, SK하이닉스) 선과 나머지 비주도 섹터/동일가중 선의 위아래 격차가 무섭게 벌어질수록 극소수 종목으로 자금이 몰리는 '압착(Compression)'이 극심함을 의미합니다.\n\n📊 **데이터 가공 기준:** 6개월 전(조회 기간 첫 날)의 가격을 '100'으로 고정(Base 100)하여, 현재까지 누적해서 몇 % 상승/하락했는지 직관적으로 비교합니다.")

    # 2. 채권 자경단의 출현
    st.header("2. 채권 자경단의 출현")
    st.markdown("미 재정부채 누적, 인플레이션 통제력 약화, 중앙은행 독립성 훼손이라는 세 가지 조건이 충족된 상황에서 미국 국채 시장으로 불똥이 튈 위험을 모니터링합니다.\n\n유가 상승과 함께 국채 금리가 급등하는 현상은 채권 자경단의 활동을 암시합니다.")
    
    fig2 = go.Figure()
    
    # 좌측 Y축: KODEX 200, WTI 유가 (Base 100)
    if '069500.KS' in df_norm.columns:
        fig2.add_trace(
            go.Scatter(
                x=df_norm.index, 
                y=df_norm['069500.KS'], 
                name="KODEX 200 (좌축, Base 100)", 
                line=dict(color='#1f77b4', dash='dash', width=2),
                yaxis='y1'
            )
        )
    if 'CL=F' in df_norm.columns:
        fig2.add_trace(
            go.Scatter(
                x=df_norm.index, 
                y=df_norm['CL=F'], 
                name="WTI 유가 (좌축, Base 100)", 
                line=dict(color='#8c564b', width=2),
                yaxis='y1'
            )
        )

    # 우측 Y축: 미 국채 10년물 실제 금리(%) - df 원본 데이터 사용
    if '^TNX' in df.columns:
        tnx_6m = df[df.index >= (df.index.max() - pd.Timedelta(days=180))]['^TNX'].ffill()
        fig2.add_trace(
            go.Scatter(
                x=tnx_6m.index, 
                y=tnx_6m.values, 
                name="미 국채 10년물 금리 (우축, 실제 %)", 
                line=dict(color='#d62728', width=2.5),
                yaxis='y2'
            )
        )
        
    fig2.update_layout(
        title="채권 자경단 모니터링: 10년물 국채 금리(실제 %) vs 유가 및 주가 (Base 100)", 
        height=430, 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="%m월"),
        yaxis=dict(title="누적 수익률 (Base 100)", side='left'),
        yaxis2=dict(title="국채 금리 (%)", side='right', overlaying='y', showgrid=False)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    st.info("💡 **차트 읽는 법:** 좌측 Y축(Base 100)은 KODEX 200과 WTI 유가의 누적 수익률을, 우측 Y축은 미 국채 10년물 금리의 실제 수치(%)를 표시합니다. KODEX 200이 고점을 높이는데도 불구하고, 금리(우축)가 4%→5% 같이 가파르게 오른다면 '채권 자경단'의 출현을 암시하는 강력한 위험 신호입니다.\n\n📊 **데이터 가공 기준:** KODEX 200/WTI는 6개월 전 = 100 기준(Base 100), 국채 금리는 실제 만기수익률(%)을 우측 Y축에 직접 표시합니다.")

    # 3. 사모 크레딧 환매 리스크
    st.header("3. 사모 크레딧 환매 리스크")
    st.markdown("데이터센터 대출 등에 집중된 사모 크레딧은 공시 의무가 없어 위험이 가려져 있습니다.\n\n시장 불안 시 환매가 도미노처럼 발생할 수 있으며, 하이일드 채권(HYG) 성과나 스프레드를 통해 비우량 신용 시장의 불안 조짐을 간접적으로 트래킹합니다.")
    
    fig3 = go.Figure()
    
    # 좌측 Y축: KODEX 200 (Base 100)
    if '069500.KS' in df_norm.columns:
        fig3.add_trace(
            go.Scatter(
                x=df_norm.index, 
                y=df_norm['069500.KS'], 
                name="KODEX 200 (좌축, Base 100)", 
                line=dict(color='#1f77b4', dash='dash', width=2),
                yaxis='y1'
            )
        )

    # 우측 Y축: HYG 실제 달러 가격 - df 원본 데이터 사용
    if 'HYG' in df.columns:
        hyg_6m = df[df.index >= (df.index.max() - pd.Timedelta(days=180))]['HYG'].ffill()
        fig3.add_trace(
            go.Scatter(
                x=hyg_6m.index, 
                y=hyg_6m.values, 
                name="하이일드 ETF HYG (우축, 실제 $)", 
                line=dict(color='#e377c2', width=2.5),
                yaxis='y2'
            )
        )
        
    fig3.update_layout(
        title="크레딧 리스크 대용 지표: 하이일드 ETF(실제 $) vs 주가 지수 (Base 100)", 
        height=430, 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="%m월"),
        yaxis=dict(title="누적 수익률 (Base 100)", side='left'),
        yaxis2=dict(title="HYG 실제 가격 ($)", side='right', overlaying='y', showgrid=False)
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    st.info("💡 **차트 읽는 법:** 좌측 Y축(Base 100)은 KODEX 200 누적 수익률을, 우측 Y축은 하이일드 ETF(HYG)의 실제 달러 가격을 표시합니다. 증시(좌축)는 평온해 보여도 HYG(우축)가 $80→$79처럼 미세하게 하락하기 시작하면, 수면 아래 비우량 신용 시장의 자금 이탈이 시작된 것으로 해석합니다.\n\n📊 **데이터 가공 기준:** KODEX 200은 Base 100 누적 수익률(좌축), HYG는 실제 달러($) 가격(우축)으로 직접 표시합니다.")

    # 4. 대형 IPO와 위험 선호도
    st.header("4. 대형 IPO와 위험 선호도")
    st.markdown("스페이스X, 앤트로픽, 오픈AI 등 대형 IPO의 성공은 시장 위험 선호도의 정점 신호가 될 수 있습니다.\n\n신규 상장 주식들의 성과를 대변하는 IPO ETF의 자금 유입 및 수익률 동향을 통해 시장의 투기적 과열 분위기가 어느 정도인지 가늠할 수 있습니다.")
    
    fig4 = go.Figure()
    
    # 좌측 Y축: KODEX 200 (Base 100) - 스케일이 크므로 좌측 분리
    if '069500.KS' in df_norm.columns:
        fig4.add_trace(
            go.Scatter(
                x=df_norm.index, 
                y=df_norm['069500.KS'], 
                name="KODEX 200 (좌축, Base 100)", 
                line=dict(color='#1f77b4', dash='dash', width=2),
                yaxis='y1'
            )
        )

    # 우측 Y축: 미국 IPO ETF와 S&P 500 (Base 100) - 두 미국 지표를 동일 스케일로 우측에 배치
    if 'IPO' in df_norm.columns:
        fig4.add_trace(
            go.Scatter(
                x=df_norm.index, 
                y=df_norm['IPO'], 
                name="미국 IPO ETF (우축, Base 100)", 
                line=dict(color='#bcbd22', width=2.5),
                yaxis='y2'
            )
        )
    if '^GSPC' in df_norm.columns:
        fig4.add_trace(
            go.Scatter(
                x=df_norm.index, 
                y=df_norm['^GSPC'], 
                name="S&P 500 (우축, Base 100)", 
                line=dict(color='#ff7f0e', dash='dot', width=2),
                yaxis='y2'
            )
        )
        
    fig4.update_layout(
        title="위험 선호도 정점 징후: IPO ETF vs S&P 500 (우측) / 국내 증시 (좌측)", 
        height=430, 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="%m월"),
        yaxis=dict(title="KODEX 200 누적 수익률 (Base 100)", side='left'),
        yaxis2=dict(title="미국 지표 누적 수익률 (Base 100)", side='right', overlaying='y', showgrid=False)
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    st.info("💡 **차트 읽는 법:** 좌측 Y축은 KODEX 200(국내 증시), 우측 Y축은 미국 IPO ETF와 S&P 500을 각각 Base 100으로 표시합니다. 우측 축 안에서 IPO ETF(연두)가 S&P 500(주황)보다 가파르게 벌어진다면 투기적 과열의 신호입니다. 국내 주가의 큰 스케일에 미국 지표가 묻히지 않도록 좌우 축을 분리했습니다.\n\n📊 **데이터 가공 기준:** 세 지표 모두 6개월 전 첫 날 = 100 기준(Base 100). KODEX 200은 좌축, IPO ETF / S&P 500은 우축으로 정밀 비교합니다.")

    # 하단 여백 추가 (화면 잘림 방지)
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

