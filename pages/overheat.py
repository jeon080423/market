import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
        
        # S&P 500 (SPY) vs Equal-Weight (RSP)
        # Leading (NVDA) vs Market
        # KOSPI (KS11) vs Samsung/SK
        tickers = ['SPY', 'RSP', 'NVDA', '^KS11', '005930.KS', '000660.KS']
        try:
            raw_data = yf.download(tickers, start=start_date, end=end_date)
            if isinstance(raw_data.columns, pd.MultiIndex):
                if 'Close' in raw_data.columns.levels[0]:
                    df = raw_data['Close']
                else:
                    df = raw_data.xs('Close', axis=1, level=0)
            elif 'Close' in raw_data.columns:
                df = raw_data[['Close']]
            else:
                df = raw_data
            return df
        except Exception as e:
            st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
            return pd.DataFrame(columns=tickers)

    with st.spinner("데이터를 불러오는 중입니다..."):
        df = get_overheat_data()

    if df.empty:
        st.warning("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        return

    # 최근 6개월(180일) 기준 데이터 정규화
    df_6m = df.last('180D').ffill().dropna(how='all')
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

    # 1. 주도주와 비주도주 간의 극단적인 수익률 격차
    st.header("1. 주도주와 비주도주 간의 극단적인 수익률 격차")
    st.markdown("과거 닷컴 버블 당시의 사례처럼, 시장 전체의 지수는 상승하지만 주도주를 제외한 대다수 종목들의 수익률이 마이너스로 돌아설 때입니다. 이는 시장에 새로운 유동성이 유입되는 것이 아니라, 기존 자금이 힘없는 종목을 팔아 주도주로만 몰리는 상황임을 의미하며, 이는 상승 사이클의 고점이 다가왔다는 강력한 신호로 해석될 수 있습니다.")
    
    fig1 = go.Figure()
    if 'NVDA' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['NVDA'], name="주도주 (NVDA)", line=dict(color='#ff4b4b', width=3)))
    if 'SPY' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['SPY'], name="S&P 500 (SPY)", line=dict(color='#1f77b4', width=2)))
    if 'RSP' in df_norm.columns:
        fig1.add_trace(go.Scatter(x=df_norm.index, y=df_norm['RSP'], name="동일가중 (RSP)", line=dict(color='#7f7f7f', dash='dash')))
    fig1.update_layout(title="최근 6개월 주도주 vs 시장 지수 수익률 변화 (Base 100)", height=450, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. 동일 가중 지수의 하락 또는 정체
    st.header("2. 동일 가중 지수(Equal-Weighted Index)의 하락 또는 정체")
    st.markdown("시가총액 가중 지수는 주도주의 비중이 커서 시장의 건강성을 왜곡할 수 있습니다. 반면, 모든 종목을 동일한 비중(1/n)으로 계산하는 동일 가중 지수가 하락하거나 전고점을 회복하지 못하고 있다면, 이는 주도주 이외의 나머지 종목들이 힘을 잃고 있다는 징후입니다.")
    
    if 'SPY' in df_norm.columns and 'RSP' in df_norm.columns:
        spread = df_norm['SPY'] - df_norm['RSP']
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=spread.index, y=spread, name="SPY - RSP 격차", fill='tozeroy', line=dict(color='#9467bd', width=2)))
        fig2.update_layout(title="시가총액 가중(SPY) vs 동일 가중(RSP) 수익률 격차 (격차가 클수록 쏠림 심화)", height=400, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    # 3. 확산의 부재
    st.header("3. '확산'의 부재")
    st.markdown("주도주 외의 후발 주자들이 함께 성장하는 '확산'의 모습이 나타나지 않고, 오직 주도주만 독식하는 구조가 지속될 때 경계심을 가져야 합니다.")
    
    fig3 = go.Figure()
    if '^KS11' in df_norm.columns:
        fig3.add_trace(go.Scatter(x=df_norm.index, y=df_norm['^KS11'], name="KOSPI 지수", line=dict(color='#1f77b4', width=2)))
    if '005930.KS' in df_norm.columns:
        fig3.add_trace(go.Scatter(x=df_norm.index, y=df_norm['005930.KS'], name="삼성전자", line=dict(color='#2ca02c', width=2)))
    if '000660.KS' in df_norm.columns:
        fig3.add_trace(go.Scatter(x=df_norm.index, y=df_norm['000660.KS'], name="SK하이닉스", line=dict(color='#ff7f0e', width=2)))
    fig3.update_layout(title="국내 시장: KOSPI vs 반도체 주도주 동향 (Base 100)", height=450, hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.info("""
    💡 **결론**: 
    김효진 박사는 이러한 시그널들을 종합할 때, 단순히 한두 달의 짧은 이격만으로 시장을 과열로 단정 짓기는 어려우며, 
    과거의 경험상 이러한 쏠림 현상이 사이클의 종결을 의미하기까지는 상당한 시간(약 1년 이상)이 소요될 수 있다고 분석했습니다.
    """)
