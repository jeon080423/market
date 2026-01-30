import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os

# [자동 업데이트] 5분
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 8대 지표 정밀 분석", layout="wide")

# [데이터 수집] 수직 튀기(Spike) 방지 로직 적용
@st.cache_data(ttl=300)
def load_clean_data():
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    
    # 1. 안정적인 과거 일봉 데이터 (최근 250일치에 집중하여 속도와 정확도 확보)
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
    hist_data = yf.download(list(tickers.keys()), start=start_date, interval='1d', progress=False)['Close']
    
    # 2. 실시간 데이터 수집 및 유효성 검사
    current_prices = {}
    for t in tickers.keys():
        try:
            # 장중 1분봉 데이터의 마지막 유효값 추출
            ticker_obj = yf.Ticker(t)
            rt_data = ticker_obj.history(period='1d', interval='1m')
            
            if not rt_data.empty and pd.notnull(rt_data['Close'].iloc[-1]):
                val = rt_data['Close'].iloc[-1]
                # 직전 종가 대비 극단적 변화(±10% 이상)는 노이즈로 판단하여 제거
                prev_val = hist_data[t].dropna().iloc[-1]
                if abs((val - prev_val) / prev_val) < 0.1:
                    current_prices[t] = val
                else:
                    current_prices[t] = prev_val
            else:
                current_prices[t] = hist_data[t].dropna().iloc[-1]
        except:
            current_prices[t] = hist_data[t].dropna().iloc[-1]

    # 3. 데이터 결합 및 스파이크 제거 (가장 중요)
    df = hist_data.copy()
    
    # 오늘 날짜 행 생성 (시간 제외한 날짜만)
    today = pd.Timestamp(datetime.now().date())
    
    # 마지막 행이 오늘 날짜인 경우 업데이트, 아니면 추가
    if df.index[-1].date() == today.date():
        for t, price in current_prices.items():
            df.at[df.index[-1], t] = price
    else:
        new_row = pd.Series(current_prices, name=pd.Timestamp(datetime.now()))
        df = pd.concat([df, pd.DataFrame([new_row])])

    # 모든 지표가 동일한 행을 갖도록 처리하고, 결측치를 선형적으로 메워 수직선을 방지함
    df = df.rename(columns=tickers).ffill().interpolate(method='linear')
    
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    
    return df.dropna().tail(250) # 분석 및 시각화용 1년치 데이터

# [분석] 회귀 모델링
def perform_analysis(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI 구현]
st.title("📊 KOSPI 8대 지표 예측 대시보드 (데이터 보정형)")
st.caption(f"최근 데이터 확인 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (5분 자동 갱신)")

try:
    df = load_clean_data()
    model, latest_x = perform_analysis(df)
    
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    if pred < -0.003: s_color, s_icon, s_text = "red", "🚨", "하락 경계"
    elif pred < 0.001: s_color, s_icon, s_text = "orange", "⏳", "중립 / 관망"
    else: s_color, s_icon, s_text = "green", "🚀", "상승 기대"

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border: 2px solid {s_color}; text-align: center;">
                <h1 style="font-size: 55px; margin: 0;">{s_icon}</h1>
                <h2 style="color: {s_color};">{s_text}</h2>
                <p>예측 수익률: <b>{pred:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.subheader("💡 실시간 투자 전략 가이드")
        st.info(f"방향성이 모호한 구간에서는 무리한 매매보다 관망을 권장합니다.")
        st.write(f"**데이터 무결성 점검:** 모든 지표의 수직 튀기 현상을 보정하였으며, 현재 설명력은 **{model.rsquared:.2%}**입니다.")

    st.divider()

    # 2행 4열 그래프
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    plt.rcParams['axes.unicode_minus'] = False

    items = [
        ('KOSPI', '1. KOSPI (보정완료)', 'MA250 - 1σ', '평균 대비 저평가'),
        ('Exchange', '2. 환율 (실시간)', 'MA250 + 1.5σ', '급등 경계'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', '단기 저점'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '추세 주의'),
        ('VIX', '5. 공포지수(VIX)', '20.0 (Fix)', '패닉 구간'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '경기 침체'),
        ('Yield_Spread', '7. 금리차', '0.00 (Fix)', '불황 전조'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '금리 압박')
    ]

    for i, (col, title, threshold_label, desc) in enumerate(items):
        ax = axes[i // 4, i % 4]
        # 최근 60일 데이터로 시각화하여 튀는 값 여부를 더 명확히 확인
        plot_data = df[col].tail(60)
        ma250 = df[col].rolling(window=250).mean().iloc[-1]
        std250 = df[col].rolling(window=250).std().iloc[-1]
        
        if col == 'Exchange': threshold = ma250 + (1.5 * std250)
        elif col in ['VIX', 'Yield_Spread']: threshold = 20.0 if col == 'VIX' else 0.0
        elif col in ['US10Y']: threshold = ma250 + std250
        else: threshold = ma250 - std250
        
        ax.plot(plot_data, color='#1f77b4', lw=3)
        ax.axhline(y=threshold, color='crimson', linestyle='--', alpha=0.9, lw=2)
        
        # 위험선 설명 (그래프 위에 표시)
        ax.text(plot_data.index[2], threshold, f" {threshold_label}", 
                fontproperties=fprop, fontsize=11, color='crimson', 
                verticalalignment='bottom', backgroundcolor='white')

        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold', pad=15)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)
        ax.annotate(f"[{desc}]", xy=(0.5, -0.18), xycoords='axes fraction', 
                    ha='center', fontproperties=fprop, fontsize=12, color='#444444')

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"데이터 정밀 보정 중 오류 발생: {e}")
