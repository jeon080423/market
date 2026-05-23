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
    st.markdown("""
    신영증권 김효진 박사의 분석에 기반하여, 시장이 단순 '주도주 상승 국면'을 넘어 
    **'주도주만 좋은(쏠림이 극단화된) 과열 국면'**에 진입했는지 판단하는 3가지 시그널을 모니터링합니다.
    """)

    # Data fetching
    @st.cache_data(ttl=3600)
    def get_overheat_data():
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # KODEX 200 (시총가중) vs KODEX 200 동일가중
        # 반도체 주도주 (삼성전자, SK하이닉스) vs 타 섹터 (KODEX 자동차)
        tickers_map = {
            '069500.KS': '069500',
            '252650.KS': '252650',
            '005930.KS': '005930',
            '000660.KS': '000660',
            '091180.KS': '091180'
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
                    
            if df_list:
                df = pd.concat(df_list, axis=1)
                for col in tickers_map.keys():
                    if col not in df.columns:
                        df[col] = pd.NA
                return df
            else:
                return pd.DataFrame(columns=list(tickers_map.keys()))
        except Exception as e:
            st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
            return pd.DataFrame(columns=list(tickers_map.keys()))

    with st.spinner("데이터를 불러오는 중입니다..."):
        df = get_overheat_data()

    if df.empty:
        st.warning("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
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

    st.markdown("---")

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
            return "⚠️ Gemini API 키가 설정되지 않아 AI 분석을 수행할 수 없습니다. Secrets 설정을 확인해주세요."
        
        prompt = f"""
당신은 신영증권 김효진 박사의 관점을 가진 수석 시장 분석가입니다.
김효진 박사는 시장이 '주도주 상승 국면'을 넘어 '주도주만 좋은 극단화된 과열 국면'에 진입했는지 판단하기 위해 3가지 시그널을 중시합니다.
1. 주도주와 비주도주 간의 극단적인 수익률 격차 (시장 전체 지수는 오르나 나머지 종목은 하락하는지)
2. 시가총액 가중 지수(KODEX 200 등) 대비 동일 가중 지수의 하락 또는 정체 
3. 확산의 부재 (반도체 외 자동차 등 후발 주자들이 동참하는지)

최근 6개월(180일) 수익률 데이터 요약(최초일을 Base 100으로 환산한 현재값):
{data_text}

위 데이터를 바탕으로 현재 시장이 건강한 상승 국면인지, 아니면 과열 징후가 뚜렷한 쏠림 국면인지 김효진 박사의 어조로 분석해주세요.
단, 한두 달의 짧은 이격만으로 시장을 섣불리 과열로 단정 짓지 말고 균형 잡힌 시각을 유지하며, 결론을 3~4문장으로 요약해주세요.
마크다운 포맷을 사용하여 핵심 내용을 강조하되, 불필요한 인사말이나 서론은 생략하고 곧바로 분석 결과만 제시해주세요.
절대로 코드 블록(```)이나 표(Table)를 사용하지 마세요. (화면 레이아웃이 깨집니다)
"""
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
        return "AI 분석을 가져오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        
    if st.button("실시간 AI 진단 실행", use_container_width=True):
        with st.spinner("AI가 최근 6개월 시장 데이터를 바탕으로 과열 시그널을 분석 중입니다..."):
            # 데이터 추출
            recent_data = {}
            ticker_names = {'069500.KS': 'KODEX 200(시총가중)', '252650.KS': 'KODEX 200 동일가중', '005930.KS': '삼성전자(주도주)', '000660.KS': 'SK하이닉스(주도주)', '091180.KS': 'KODEX 자동차(후발주)'}
            for col, name in ticker_names.items():
                if col in df_norm.columns:
                    recent_data[name] = f"{df_norm[col].iloc[-1]:.2f} (Base 100)"
            
            data_text = "\n".join([f"- {k}: {v}" for k, v in recent_data.items()])
            analysis_result = get_ai_overheat_analysis(data_text)
            analysis_result = analysis_result.replace("```markdown", "").replace("```", "").strip()
            
            st.success("분석 완료")
            with st.container(border=True):
                st.markdown(analysis_result)

    st.markdown("---")
    # 1. 주도주와 비주도주 간의 극단적인 수익률 격차
    st.header("1. 주도주와 비주도주 간의 극단적인 수익률 격차")
    st.markdown("과거 닷컴 버블 당시의 사례처럼, 시장 전체의 지수는 상승하지만 주도주를 제외한 대다수 종목들의 수익률이 마이너스로 돌아설 때입니다. 이는 시장에 새로운 유동성이 유입되는 것이 아니라, 기존 자금이 힘없는 종목을 팔아 주도주로만 몰리는 상황임을 의미하며, 이는 상승 사이클의 고점이 다가왔다는 강력한 신호로 해석될 수 있습니다.")
    
    fig1 = go.Figure()
    if '005930.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['005930.KS'], name="삼성전자 (주도주)", line=dict(color='#ff4b4b', width=2)))
    if '000660.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['000660.KS'], name="SK하이닉스 (주도주)", line=dict(color='#ff7f0e', width=2)))
    if '069500.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['069500.KS'], name="KODEX 200 (시총가중)", line=dict(color='#1f77b4', width=2)))
    if '252650.KS' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['252650.KS'], name="KODEX 200 동일가중", line=dict(color='#7f7f7f', dash='dash')))
    fig1.update_layout(title="최근 6개월 주도주 vs 시장 지수 수익률 변화 (Base 100)", height=450, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. 동일 가중 지수의 하락 또는 정체
    st.header("2. 동일 가중 지수(Equal-Weighted Index)의 하락 또는 정체")
    st.markdown("시가총액 가중 지수는 주도주의 비중이 커서 시장의 건강성을 왜곡할 수 있습니다. 반면, 모든 종목을 동일한 비중(1/n)으로 계산하는 동일 가중 지수가 하락하거나 전고점을 회복하지 못하고 있다면, 이는 주도주 이외의 나머지 종목들이 힘을 잃고 있다는 징후입니다.")
    
    if '069500.KS' in df_norm.columns and '252650.KS' in df_norm.columns:
        spread = df_norm['069500.KS'] - df_norm['252650.KS']
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=spread.index, y=spread, name="시총가중 - 동일가중 격차", fill='tozeroy', line=dict(color='#9467bd', width=2)))
        fig2.update_layout(title="KODEX 200 (시총가중) vs KODEX 200 동일가중 수익률 격차 (격차가 클수록 쏠림 심화)", height=400, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    # 3. 확산의 부재
    st.header("3. '확산'의 부재")
    st.markdown("주도주 외의 후발 주자들이 함께 성장하는 '확산'의 모습이 나타나지 않고, 오직 주도주만 독식하는 구조가 지속될 때 경계심을 가져야 합니다.")
    
    fig3 = go.Figure()
    if '069500.KS' in df_norm.columns:
        fig3.add_trace(go.Scatter(x=df_norm.index, y=df_norm['069500.KS'], name="KODEX 200", line=dict(color='#1f77b4', width=2)))
    if '005930.KS' in df_norm.columns:
        fig3.add_trace(go.Scatter(x=df_norm.index, y=df_norm['005930.KS'], name="삼성전자 (주도주)", line=dict(color='#ff4b4b', width=2)))
    if '000660.KS' in df_norm.columns:
        fig3.add_trace(go.Scatter(x=df_norm.index, y=df_norm['000660.KS'], name="SK하이닉스 (주도주)", line=dict(color='#ff7f0e', width=2)))
    if '091180.KS' in df_norm.columns:
        fig3.add_trace(go.Scatter(x=df_norm.index, y=df_norm['091180.KS'], name="KODEX 자동차 (후발주)", line=dict(color='#2ca02c', width=2, dash='dash')))
    fig3.update_layout(title="확산 여부 확인: 주도주(반도체) vs 후발주(자동차) 동향 (Base 100)", height=450, hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

