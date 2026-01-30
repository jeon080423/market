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
import gspread
from google.oauth2.service_account import Credentials

# [자동 업데이트] 5분 주기
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [구글 시트 설정]
# 주의: 서비스 계정 JSON 파일 경로를 올바르게 설정해야 합니다.
SERVICE_ACCOUNT_FILE = 'https://github.com/jeon080423/market/edit/main/key.json' 
SHEET_NAME = 'KOSPI_Prediction_History'

def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    return gspread.authorize(creds)

def update_gsheet_history(date, predicted_return, actual_close):
    try:
        client = get_gsheet_client()
        sh = client.open(SHEET_NAME).get_worksheet(0)
        
        # 데이터가 비어있을 경우 헤더 생성
        if not sh.get_all_values():
            sh.append_row(["Date", "Predicted_Return", "Actual_Close", "Accuracy_Diff"])
            
        # 기존 날짜 확인 (중복 기록 방지)
        dates = sh.col_values(1)
        if date not in dates:
            # 전일 예측치가 있다면 오늘 종가와 비교하여 기록 (간단한 구현 예시)
            sh.append_row([date, f"{predicted_return:.4f}", f"{actual_close:.2f}"])
    except Exception as e:
        st.warning(f"구글 시트 업데이트 실패: {e}")

def load_gsheet_history():
    try:
        client = get_gsheet_client()
        sh = client.open(SHEET_NAME).get_worksheet(0)
        data = sh.get_all_records()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame(columns=["Date", "Predicted_Return", "Actual_Close"])

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 정밀 진단 v2.7", layout="wide")

# [데이터 수집] 개별 수집으로 안정성 확보
@st.cache_data(ttl=300)
def load_expert_data():
    tickers = {
        '^KS11': 'KOSPI', 'USDKRW=X': 'Exchange', '^SOX': 'SOX', '^GSPC': 'SP500', 
        '^VIX': 'VIX', '000001.SS': 'China', '^TNX': 'US10Y', '^IRX': 'US2Y'
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
        except:
            continue

    df = combined_df.ffill().interpolate()
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    return df.dropna().tail(300)

# [분석] 영향도 및 설명력 극대화 모델
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
    return model, contribution

# [날짜 포맷터]
def custom_date_formatter(x, pos):
    dt = mdates.num2date(x)
    return dt.strftime('%Y/%m') if dt.month == 1 else dt.strftime('%m')

try:
    df = load_expert_data()
    model, contribution_pct = get_analysis(df)
    
    # 데이터 기록 (오늘 종가와 예측치 저장)
    today_str = datetime.now().strftime('%Y-%m-%d')
    current_close = df['KOSPI'].iloc[-1]
    
    # 상단 요약 가이드 섹션
    c1, c2, c3 = st.columns([1.1, 1.1, 1.3])
    
    with c1:
        current_data = df.tail(3).mean()
        current_scaled = (current_data[contribution_pct.index] - df[contribution_pct.index].mean()) / df[contribution_pct.index].std()
        current_scaled['SOX_SP500'] = current_scaled['SOX_lag1'] * current_scaled['SP500']
        
        pred_val_level = model.predict([1] + current_scaled.tolist())[0]
        prev_val_level = df['KOSPI'].iloc[-2]
        pred_val = (pred_val_level - prev_val_level) / prev_val_level
        
        # 구글 시트에 오늘의 예측치와 현재가 업데이트 시도
        update_gsheet_history(today_str, pred_val, current_close)
        
        color = "#e74c3c" if pred_val < 0 else "#2ecc71"
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 15px; border-left: 10px solid {color}; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 260px;">
                <h3 style="margin: 0; color: #555;">📈 KOSPI 기대 수익률: <span style="color:{color}">{pred_val:+.2%}</span></h3>
                <p style="color: #444; font-size: 13px; margin-top: 10px; line-height: 1.5;">
                    <b>[예측 히스토리]</b><br>
                    데이터가 구글 시트(<b>{SHEET_NAME}</b>)에 누적 기록됩니다.<br>
                    오늘의 실시간 종가: <b>{current_close:,.2f}</b>
                </p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        # 구글 시트에서 과거 히스토리 불러오기
        history_df = load_gsheet_history()
        if not history_df.empty:
            st.markdown(f"""
                <div style="padding: 20px; border-radius: 15px; border-left: 10px solid #3498db; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 260px; overflow-y: auto;">
                    <h3 style="margin: 0; color: #555;">📊 예측 정확도 히스토리</h3>
                    <p style="font-size: 12px; color: #666; margin-top: 5px;">최근 기록된 데이터입니다.</p>
                    {history_df.tail(5).to_html(index=False, classes='table')}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("구글 시트에 누적된 데이터가 없습니다.")

    with c3:
        st.subheader("📊 지표별 KOSPI 영향력 비중")
        def highlight_max(s):
            is_max = s == s.max()
            return ['color: red; font-weight: bold' if v else '' for v in is_max]
        cont_df = pd.DataFrame(contribution_pct).T
        st.table(cont_df.style.format("{:.1f}%").apply(highlight_max, axis=1))
        st.markdown(f"<div style='font-size: 12px; color: #666;'>설명력: {model.rsquared:.2%}</div>", unsafe_allow_html=True)

    st.divider()

    # 하단 그래프 (기존 유지)
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
        ma = df[col].rolling(window=250).mean().iloc[-1]
        std = df[col].rolling(window=250).std().iloc[-1]
        if col == 'Exchange': threshold = ma + (1.5 * std)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(th_label)
        elif col in ['US10Y']: threshold = ma + std
        else: threshold = ma - std
        ax.plot(plot_data, color='#34495e', lw=2.5)
        ax.axhline(y=threshold, color='#e74c3c', ls='--', lw=2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel(f"{warn_text}", fontproperties=fprop, fontsize=11, color='#c0392b')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"오류 발생: {e}")
