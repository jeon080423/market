import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# [자동 업데이트] 15분 주기
st_autorefresh(interval=15 * 60 * 1000, key="datarefresh")

# [로컬 데이터 보존 설정]
HISTORY_FILE = 'prediction_history.csv'

def save_prediction_history(date_str, pred_val, actual_close, prev_close):
    """예측 데이터를 로컬 CSV 파일에 저장하여 메모리 유지"""
    pred_close = prev_close * (1 + pred_val)
    diff = actual_close - pred_close 
    
    new_data = pd.DataFrame([[
        date_str, f"{pred_val:.4%}", f"{pred_close:,.2f}", f"{actual_close:,.2f}",
        f"{diff:,.2f}", datetime.now().strftime('%H:%M:%S')
    ]], columns=["날짜", "전일대비 예측수익률", "예측 종가", "실제 종가", "예측 오차", "기록시각"])
    
    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            if date_str not in history_df["날짜"].values:
                current_time = datetime.now().time()
                market_close = datetime.strptime("15:30", "%H:%M").time()
                if current_time >= market_close:
                    history_df = pd.concat([history_df, new_data], ignore_index=True)
                    history_df.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
        except:
            new_data.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
    else:
        new_data.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')

def load_prediction_history():
    if os.path.exists(HISTORY_FILE):
        try: return pd.read_csv(HISTORY_FILE)
        except: return pd.DataFrame(columns=["날짜", "전일대비 예측수익률", "예측 종가", "실제 종가", "예측 오차", "기록시각"])
    return pd.DataFrame(columns=["날짜", "전일대비 예측수익률", "예측 종가", "실제 종가", "예측 오차", "기록시각"])

@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path): return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()
st.set_page_config(page_title="KOSPI 인텔리전스 진단 v3.0", layout="wide")

@st.cache_data(ttl=900)
def load_expert_data():
    tickers = {
        '^KS11': 'KOSPI', 'USDKRW=X': 'Exchange', '^SOX': 'SOX', '^GSPC': 'SP500', 
        '^VIX': 'VIX', '000001.SS': 'China', '^TNX': 'US10Y', '^IRX': 'US2Y',
        '005930.KS': 'Samsung', '000660.KS': 'Hynix', '005380.KS': 'Hyundai', '373220.KS': 'LG_Energy'
    }
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    combined_df = pd.DataFrame()
    for ticker, name in tickers.items():
        try:
            raw = yf.download(ticker, start=start_date, interval='1d', progress=False)
            if not raw.empty:
                rt = yf.download(ticker, period='1d', interval='1m', progress=False)
                val = rt['Close'].iloc[-1] if not rt.empty else raw['Close'].iloc[-1]
                series = raw['Close'].copy()
                series.iloc[-1] = val
                combined_df[name] = series
        except: continue
    if combined_df.empty: raise Exception("데이터 수집 실패")
    df = combined_df.ffill().interpolate()
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    return df.dropna().tail(300)

def get_analysis(df):
    features_list = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y']
    df_smooth = df.rolling(window=3).mean().dropna()
    y = df_smooth['KOSPI']
    X = df_smooth[features_list]
    X_scaled = (X - X.mean()) / X.std()
    X_scaled['SOX_SP500'] = X_scaled['SOX_lag1'] * X_scaled['SP500']
    X_final = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_final).fit()
    
    abs_coeffs = np.abs(model.params.drop(['const', 'SOX_SP500']))
    contribution = (abs_coeffs / abs_coeffs.sum()) * 100
    # 에러 방지를 위해 컬럼명 정보를 포함하여 반환
    return model, contribution, X.mean(), X.std(), X_final.columns.tolist()

def custom_date_formatter(x, pos):
    dt = mdates.num2date(x)
    return dt.strftime('%Y/%m') if dt.month == 1 else dt.strftime('%m')

try:
    df = load_expert_data()
    model, contribution_pct, train_mean, train_std, model_cols = get_analysis(df)
    
    # --- 데이터 계산 영역 (예측 로직 강화) ---
    def predict_return(target_mean_series):
        # 1. 정규화
        scaled = (target_mean_series[contribution_pct.index] - train_mean) / train_std
        # 2. 상호작용항 및 상수항 생성
        scaled_df = scaled.to_frame().T
        scaled_df['SOX_SP500'] = scaled_df['SOX_lag1'] * scaled_df['SP500']
        scaled_df['const'] = 1.0
        # 3. 모델 컬럼 순서와 완벽 일치시킴
        return model.predict(scaled_df[model_cols]).iloc[0]

    current_pred_level = predict_return(df.tail(3).mean())
    pred_val = (current_pred_level - df['KOSPI'].iloc[-2]) / df['KOSPI'].iloc[-2]
    
    mid_pred_level = predict_return(df.tail(20).mean())
    mid_pred_val = (mid_pred_level - df['KOSPI'].tail(20).iloc[0]) / df['KOSPI'].tail(20).iloc[0]

    r2 = model.rsquared
    reliability = "강함" if r2 > 0.85 else "보통" if r2 > 0.7 else "주의"

    # --- 레이아웃 구현 ---
    st.markdown(f"## 🏛️ KOSPI 인텔리전스 진단 시스템 <small>v3.0</small>", unsafe_allow_html=True)
    
    h1, h2 = st.columns([3, 1])
    with h1:
        mood = "상승 우세" if pred_val > 0 else "하락 압력"
        st.info(f"🤖 **AI 마켓 브리핑:** 현재 시장의 주동력은 **{contribution_pct.idxmax()}**이며, 분석 신뢰도는 **{reliability}**({r2:.1%})입니다. 단기적으로 **{mood}** 구간입니다.")
    with h2:
        cash = 10 if pred_val > 0.005 else 40 if pred_val > 0 else 70 if pred_val > -0.005 else 90
        st.metric("권장 현금 비중", f"{cash}%", f"{'방어적' if cash >= 70 else '공격적'} 전략")

    st.divider()

    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c1:
        today_str = datetime.now().strftime('%Y-%m-%d')
        save_prediction_history(today_str, pred_val, df['KOSPI'].iloc[-1], df['KOSPI'].iloc[-2])
        color = '#e74c3c' if pred_val < 0 else '#2ecc71'
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 15px; border-left: 10px solid {color}; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 260px;">
                <h3 style="margin: 0; color: #555;">📈 KOSPI 기대 수익률: <span style="color:{color}">{pred_val:+.2%}</span></h3>
                <p style="color: #444; font-size: 13px; margin-top: 10px; line-height: 1.5;">
                    <b>[단기 수치 해석]</b><br>
                    8대 지표의 실시간 변화를 다중 회귀 모델에 대입하여 산출한 기대 수익률입니다.<br>
                    - (+) 상승 압력 / (-) 하락 압력
                </p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        if pred_val < -0.005 and mid_pred_val < 0: signal, s_color = "🔴 즉시 매도", "#ff4b4b"
        elif pred_val < 0: signal, s_color = "🟠 매도 준비", "#ffa500"
        elif pred_val > 0.005 and mid_pred_val > 0: signal, s_color = "🔵 매수 유효", "#1f77b4"
        else: signal, s_color = "⚪ 보유 및 관망", "#888"
        
        reason = f"단기 기대치({pred_val:+.2%})와 중기 추세({mid_pred_val:+.2%}) 기반 결과입니다."
        if "매도" in signal: reason += " 하락 압력이 포착되므로 리스크 관리가 필요합니다."
        elif "매수" in signal: reason += " 상승 에너지가 강력하여 추가 상승이 기대됩니다."

        st.markdown(f"""
            <div style="display: flex; gap: 10px; height: 260px;">
                <div style="flex: 1; padding: 15px; border-radius: 10px; background-color: {s_color}; color: white; text-align: center; display: flex; flex-direction: column; justify-content: center;">
                    <h5 style="margin: 0;">⚡ 전략 신호</h5>
                    <h2 style="margin: 10px 0; font-weight: bold; font-size: 28px;">{signal}</h2>
                </div>
                <div style="flex: 1.2; padding: 15px; border-radius: 10px; border: 1px solid #ddd; background-color: #fff; overflow-y: auto;">
                    <h6 style="margin: 0 0 5px 0; color: #333;">🧐 판단 이유</h6>
                    <p style="margin: 0; font-size: 12px; line-height: 1.6; color: #555;">{reason}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        m_color = '#e74c3c' if mid_pred_val < 0 else '#2ecc71'
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 15px; border-left: 10px solid {m_color}; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 260px;">
                <h3 style="margin: 0; color: #555;">📅 중기 투자 전망: <span style="color:{m_color}">{mid_pred_val:+.2%}</span></h3>
                <p style="color: #444; font-size: 13px; margin-top: 10px; line-height: 1.5;">
                    <b>[중기 예측 설명]</b><br>
                    최근 <b>20거래일(약 1개월)</b>간의 글로벌 지표 누적 변화를 바탕으로 산출한 추세적 방향성입니다.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    col_hist, col_sector = st.columns([1.5, 1])
    with col_hist:
        history_df = load_prediction_history()
        if not history_df.empty:
            st.markdown("##### 📊 예측 히스토리 (최근 10회)")
            st.dataframe(history_df.tail(10), use_container_width=True)
    with col_sector:
        st.markdown("##### 🔄 주도 업종 수익률 모멘텀 (5일)")
        sector_rets = df[['Samsung', 'Hynix', 'Hyundai', 'LG_Energy']].pct_change(5).iloc[-1] * 100
        st.bar_chart(sector_rets)

    st.divider()
    st.markdown("##### 📊 글로벌 지표별 KOSPI 영향력 비중")
    def highlight_max(s):
        return ['color: red; font-weight: bold' if v == s.max() else '' for v in s]
    st.table(pd.DataFrame(contribution_pct).T.style.format("{:.1f}%").apply(highlight_max, axis=1))

    st.divider()
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    plt.subplots_adjust(hspace=0.4)
    config = [
        ('KOSPI', '1. KOSPI 본체', 'MA250 - 1σ', '선 아래로 하향 시 [추세 붕괴]'),
        ('Exchange', '2. 원/달러 환율', 'MA250 + 1.5σ', '선 위로 상향 시 [외인 자금 이탈]'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', '선 아래로 하향 시 [IT 공급망 위기]'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '선 아래로 하향 시 [글로벌 심리 위축]'),
        ('VIX', '5. 공포지수(VIX)', '20.0', '선 위로 상향 시 [시장 패닉 진입]'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '선 아래로 하향 시 [아시아권 경기 침체]'),
        ('Yield_Spread', '7. 장단기 금리차', '0.0', '선 아래로 하향 시 [경제 불황 전조]'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '선 위로 상향 시 [유동성 긴축 압박]')
    ]

    for i, (col, title, th_label, warn_text) in enumerate(config):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(100)
        ma, std = df[col].rolling(window=250).mean().iloc[-1], df[col].rolling(window=250).std().iloc[-1]
        if col == 'Exchange': threshold = ma + (1.5 * std)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(th_label)
        elif col in ['US10Y']: threshold = ma + std
        else: threshold = ma - std
        
        ax.plot(plot_data, color='#34495e', lw=2.5)
        ax.axhline(y=threshold, color='#e74c3c', ls='--', lw=2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold')
        ax.text(plot_data.index[0], threshold, f"근거: {th_label}", fontproperties=fprop, color='#e74c3c', va='bottom', fontsize=10, backgroundcolor='#ffffff')
        dist = abs(plot_data.iloc[-1] - threshold) / (abs(threshold) if threshold != 0 else 1)
        ax.set_xlabel(f"위험선 대비 거리: {dist:.1%} | {warn_text}", fontproperties=fprop, fontsize=11, color='#c0392b')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontproperties(fprop)

    st.pyplot(fig)

except Exception as e:
    st.error(f"분석 엔진 오류 발생: {e}")
