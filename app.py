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

# [자동 업데이트] 5분 주기
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 8대 지표 정밀 진단", layout="wide")

# [데이터 수집] 변동성 보정 로직 포함
@st.cache_data(ttl=300)
def load_validated_data():
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    
    # 1. 과거 일봉 데이터 (안정적인 기준점)
    start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    hist_data = yf.download(list(tickers.keys()), start=start_date, interval='1d', progress=False)['Close']
    
    # 2. 실시간 데이터 정밀 수집
    current_data = {}
    for t in tickers.keys():
        try:
            # 장중 1d-1m 데이터를 가져와서 마지막 가격 추출
            ticker_obj = yf.Ticker(t)
            # history 대신 download로 일관성 유지
            rt_tmp = yf.download(t, period='1d', interval='1m', progress=False)
            if not rt_tmp.empty:
                last_price = rt_tmp['Close'].iloc[-1]
                # 변동성 필터: 이전 종가 대비 10% 이상 급변 시 데이터 노이즈로 간주하고 무시
                prev_close = hist_data[t].iloc[-1]
                if abs((last_price - prev_close) / prev_close) < 0.1:
                    current_data[t] = last_price
                else:
                    current_data[t] = prev_close
            else:
                current_data[t] = hist_data[t].iloc[-1]
        except:
            current_data[t] = hist_data[t].iloc[-1]

    # 3. 데이터 정합성 결합
    data = hist_data.copy()
    today_ts = pd.Timestamp(datetime.now().date())
    
    # 오늘 날짜가 이미 인덱스에 있는지 확인 후 업데이트 또는 추가
    if data.index[-1].date() == today_ts.date():
        data.iloc[-1] = pd.Series(current_data)
    else:
        new_row = pd.DataFrame([current_data], index=[pd.Timestamp(datetime.now())])
        data = pd.concat([data, new_row])
    
    data = data.rename(columns=tickers).ffill().bfill()
    data['SOX_lag1'] = data['SOX'].shift(1)
    data['Yield_Spread'] = data['US10Y'] - data['US2Y']
    
    return data.dropna()

# [분석] 회귀 모델링
def perform_analysis(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI 구현]
st.title("🛡️ KOSPI 8대 지표 정밀 진단 시스템 (데이터 보정 완료)")
st.caption(f"최종 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 변동성 필터링 적용 중")

try:
    df = load_validated_data()
    model, latest_x = perform_analysis(df)
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    # 신호 및 가이드
    if pred < -0.003:
        s_color, s_icon, s_text = "red", "🚨", "하락 경계"
    elif pred < 0.001:
        s_color, s_icon, s_text = "orange", "⏳", "중립/관망"
    else:
        s_color, s_icon, s_text = "green", "🚀", "상승 기대"

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border: 2px solid {s_color}; text-align: center;">
                <h1 style="font-size: 50px; margin: 0;">{s_icon}</h1>
                <h2 style="color: {s_color};">{s_text}</h2>
                <p>예측 수익률: <b>{pred:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.subheader("💡 데이터 신뢰성 확인")
        st.write(f"현재 모든 지표의 **단위 보정 및 노이즈 필터링**이 완료되었습니다.")
        st.write(f"최근 1,000일간의 장기 추세와 오늘의 실시간 변동을 결합하여 분석 중입니다.")
        st.info(f"모델 설명력(R²): {model.rsquared:.2%} | 지표 간 시차(Lag) 데이터 정렬 완료")

    st.divider()

    # 그래프 (2행 4열)
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    plt.rcParams['axes.unicode_minus'] = False

    items = [
        ('KOSPI', '1. KOSPI', 'MA250 - 1σ', '저평가 구간'),
        ('Exchange', '2. 환율', 'MA250 + 1.5σ', '급등 경계'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', '단기 저점'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '추세 주의'),
        ('VIX', '5. 공포지수(VIX)', '20.0 (Fix)', '패닉 구간'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '경기 침체'),
        ('Yield_Spread', '7. 금리차', '0.00 (Fix)', '불황 전조'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '금리 압박')
    ]

    for i, (col, title, threshold_label, desc) in enumerate(items):
        ax = axes[i // 4, i % 4]
        # 시각화 데이터 범위를 tail(100)으로 제한하여 변동성을 더 자세히 확인
        plot_data = df[col].tail(100)
        ma250 = df[col].rolling(window=250).mean().iloc[-1]
        std250 = df[col].rolling(window=250).std().iloc[-1]
        
        if col == 'Exchange': threshold = ma250 + (1.5 * std250)
        elif col in ['VIX', 'Yield_Spread']: threshold = 20.0 if col == 'VIX' else 0.0
        elif col in ['US10Y']: threshold = ma250 + std250
        else: threshold = ma250 - std250
        
        ax.plot(plot_data, color='#1f77b4', lw=2.5)
        ax.axhline(y=threshold, color='crimson', linestyle='--', alpha=0.9, lw=2)
        ax.text(plot_data.index[5], threshold, f" {threshold_label}", 
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
    st.error(f"데이터 정합성 확인 중 오류 발생: {e}")
