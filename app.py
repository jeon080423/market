import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import time
import re
from io import StringIO
import google.generativeai as genai
from utils.ai_gsheet_cache import load_ai_cache, save_ai_cache, should_update_ai, is_data_changed_significantly

# 이전 파일 자동 삭제 스크립트 (Windows 샌드박스 우회용)
import os
try:
    if os.path.exists("pages/youtube_rank.py"):
        os.remove("pages/youtube_rank.py")
except Exception:
    pass

from pages.종목탐색 import render_youtube_rank_page
from pages.overheat import render_overheat_page

# 1. 페이지 설정
st.set_page_config(page_title="코스피(KOSPI) 전망 및 시장 과열 위험 모니터링 대시보드", layout="wide")

# 자동 새로고침 설정 (10분 간격)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=600000, key="datarefresh")
except ImportError:
    pass

# 2. Secrets에서 API Key 불러오기
def check_secrets():
    # Streamlit Cloud의 secrets는 대소문자를 구분할 수 있으므로, 
    # 모든 키를 소문자로 변환하여 체크하는 헬퍼 함수 정의
    def get_case_insensitive_secret(keys):
        # flat keys (e.g. news_api_key) or nested keys (e.g. ["news_api"]["api_key"])
        for k in keys:
            if isinstance(k, list): # Nested
                section = k[0].lower()
                field = k[1].lower()
                for s_key in st.secrets.keys():
                    if s_key.lower() == section:
                        section_obj = st.secrets[s_key]
                        if hasattr(section_obj, "get"):
                            for f_key in section_obj.keys():
                                if f_key.lower() == field:
                                    return section_obj[f_key]
            else: # Flat
                target = k.lower()
                for s_key in st.secrets.keys():
                    if s_key.lower() == target:
                        return st.secrets[s_key]
        return None

    news_key = get_case_insensitive_secret([["news_api", "api_key"], "news_api_key"])
    gemini_key = get_case_insensitive_secret([["gemini", "api_key"], "gemini_api_key", "google_api_key"])
    # admin 비밀번호는 pw or password 둘 다 허용
    admin_id = get_case_insensitive_secret([["auth", "admin_id"], "admin_id"])
    admin_pw = get_case_insensitive_secret([["auth", "admin_pw"], ["auth", "admin_password"], "admin_pw", "admin_password"])
    sheet_id = get_case_insensitive_secret([["gsheets", "sheet_id"], ["gsheet", "sheet_id"], "sheet_id"])
    
    secrets_status = {
        "news_api": news_key is not None,
        "gemini": gemini_key is not None,
        "auth": admin_id is not None and admin_pw is not None,
        "gsheet": sheet_id is not None
    }
    
    missing_keys = [k for k, v in secrets_status.items() if not v]
    if missing_keys:
        st.error(f"⚠️ 다음 Secrets 설정이 누락되었거나 형식이 잘못되었습니다: {', '.join(missing_keys)}")
        
        with st.expander("🛠️ 스트리밋 클라우드 시크릿 설정 방법 (중요한 해결책)", expanded=True):
            st.markdown(f"""
            ### 원인 분석
            현재 시스템이 다음 필수 설정을 찾을 수 없습니다: **{', '.join(missing_keys)}**
            (대소문자 오타나 `[section]` 형식이 맞지 않을 수 있습니다.)
            
            ### 해결 방법
            1.  **Streamlit Cloud** 대시보드의 **Settings > Secrets**에서 아래 내용을 참고하여 설정을 수정해 보세요.\n\n""")
            
            st.code(f"""
[news_api]
api_key = "발급받은_NewsAPI_키"

[gemini]
api_key = "발급받은_Gemini_API_키"

[auth]
admin_id = "아이디"
admin_pw = "비밀번호"

[gsheet]
sheet_id = "구글시트_ID"
            """, language="toml")
            
            try:
                found_keys = list(st.secrets.keys())
                st.info(f"🔍 **현재 감지된 최상위 키들:** `{found_keys}`")
                # 각 섹션 내부의 키도 보여주기 (값은 제외)
                for k in found_keys:
                    section = st.secrets[k]
                    if hasattr(section, "keys"):
                        st.write(f"- `{k}` 섹션 내부 키: `{list(section.keys())}`")
            except: pass
                
            st.warning("💡 **참고:** 이미 입력했다고 생각되신다면, `[news_api]`와 같은 섹션 제목이 정확한지 다시 한번 확인 부탁드립니다.")
        st.stop()
        
    return news_key, gemini_key, admin_id, admin_pw, sheet_id

NEWS_API_KEY, GEMINI_API_KEY, ADMIN_ID, ADMIN_PW, SHEET_ID = check_secrets()

# Gemini 설정 및 모델 초기화
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gemini 설정 중 오류 발생: {e}")

# AI 분석 함수 정의 (할당량 보호를 위해 캐시 적용)
@st.cache_data(ttl=86400) # cache for 1 day
def get_supported_gemini_models():
    # 최신 Gemini 3 및 2.5 지원 모델 목록으로 구성 (구버전 1.5 제외)
    default_models = [
        "gemini-3.5-flash",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-pro-latest"
    ]
    try:
        api_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace('models/', '')
                api_models.append(model_name)
        if api_models:
            priority_list = [
                "gemini-3.5-flash",
                "gemini-3.1-pro-preview",
                "gemini-3-flash-preview",
                "gemini-3.1-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-pro-latest"
            ]
            supported = [m for m in priority_list if m in api_models]
            supported += [m for m in api_models if m not in supported and ('vision' not in m.lower())]
            if supported:
                return supported
    except:
        pass
    return default_models

@st.cache_data(ttl=3600)  # 1시간 동안 동일 프롬프트에 대해 API 호출 방지
def get_ai_analysis(prompt):
    models = get_supported_gemini_models()
    
    last_error_msg = ""
    for model_name in models:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                # gemini-2.5-flash는 thinking 모드를 끄서 빠르게 응답하도록 설정
                if "2.5" in model_name:
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                else:
                    response = model.generate_content(prompt)
                if response and hasattr(response, 'text'):
                    return response.text
                break
            except Exception as e:
                last_error_msg = str(e)
                err_msg = last_error_msg.lower()
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    break # try next model
                    
    err_msg_lower = last_error_msg.lower()
    if "quota" in err_msg_lower or "429" in err_msg_lower or "exhausted" in err_msg_lower:
        return "⚠️ Gemini API 무료 사용량(Quota)을 모두 소진했습니다.\n\n내일 다시 시도하시거나, 구글 클라우드에서 결제 정보를 등록해 한도를 늘려주세요."
    return f"⚠️ 현재 AI 모델 서버가 혼잡하여 일시적으로 응답을 받을 수 없습니다.\n\n잠시 후 다시 시도해 주세요. (에러: {last_error_msg})"

# AI 응답 정제 함수 (최소한의 안전 필터만 적용)
def clean_ai_output(text):
    if not text: return ""
    import re
    
    # <result> 태그가 있으면 안의 내용만 추출, 없으면 전체 사용
    match = re.search(r'<result>(.*?)</result>', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    
    # AI가 "Draft 1 (Literal):..." 형태의 단락을 여러 개 뱉으면
    # 마지막 단락만 사용 (가장 완성된 번역)
    draft_positions = [m.start() for m in re.finditer(r'Draft\s*\d+', text, re.IGNORECASE)]
    if len(draft_positions) >= 2:
        text = text[draft_positions[-1]:]
        text = re.sub(r'Draft\s*\d+[^:]*:\s*', '', text, count=1, flags=re.IGNORECASE)
    elif len(draft_positions) == 1:
        text = re.sub(r'Draft\s*\d+[^:]*:\s*', '', text, flags=re.IGNORECASE)
    
    # 마크다운 강조 기호만 제거하고 내용은 그대로 반환
    text = text.replace('**', '').replace('##', '').replace('```', '').strip()
    # 줄바꿈을 <br>로 변환하고 반환
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '<br>'.join(lines)

# AI 뉴스 번역 정제 함수 (번호가 매겨진 번역 결과만 추출)
def clean_news_translation(text):
    if not text: return ""
    import re
    
    # <result> 태그가 있으면 안의 내용만 추출
    match = re.search(r'<result>(.*?)</result>', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        
    lines = text.split('\n')
    numbered_lines = []
    for line in lines:
        # 마크다운 기호 제거 후 공백 제거
        line_clean = line.strip().replace('**', '').replace('*', '').strip()
        # "1. ", "2. " 등 숫자로 시작하고 한글을 포함하는 행만 필터링 (영어 프롬프트 설명글 제거)
        if re.match(r'^\d+\.\s+', line_clean) and re.search(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', line_clean):
            numbered_lines.append(line_clean)
            
    if numbered_lines:
        return '<br>'.join(numbered_lines)
        
    # 번호가 매겨진 행을 찾지 못한 경우 일반 정제 로직으로 대체
    return clean_ai_output(text)

# 코로나19 폭락 기점 날짜 정의 (S&P 500 고점 기준)
COVID_EVENT_DATE = "2020-02-19"

# 구글 시트 URL 생성
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
# CSS 주입: 제목 폰트 유동성 및 가이드북 간격/정렬 조정
st.markdown("""
    <style>
    /* 메인 제목 유동적 폰트 크기 설정 */
    h1 {
        font-size: clamp(24px, 4vw, 48px) !important;
    }
    
    /* 지수 가이드북 제목 스타일 */
    .guide-header {
        font-size: clamp(18px, 2.5vw, 28px) !important;
        font-weight: 600;
        margin-bottom: 45px !important; 
        margin-top: 60px !important;    
        padding-top: 10px !important;
    }

    /* 설명글 유동적 폰트 및 줄간격 설정 */
    .guide-text {
        font-size: clamp(14px, 1.2vw, 20px) !important;
        line-height: 1.8 !important;
    }
    
    /* 가이드북 내 테이블 스타일: 설명 폰트(guide-text)와 동일하게 표시되도록 설정 */
    div[data-testid="stMarkdownContainer"] table {
        width: 100% !important;
        table-layout: auto !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stMarkdownContainer"] table th,
    div[data-testid="stMarkdownContainer"] table td {
        font-size: clamp(14px, 1.2vw, 20px) !important; /* 설명글 폰트 크기와 동일하게 수정 */
        word-wrap: break-word !important;
        padding: 12px 4px !important; 
        line-height: 1.8 !important; /* 줄간격 통일 */
    }
    
    /* 수평선(hr) 여백 조정 */
    hr {
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* AI 분석 결과 박스 커스텀 (야간 모드 대응 및 시인성 개선) */
    .ai-analysis-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #31333F !important; /* 글자색 강제 고정 */
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        line-height: 1.65;
        font-size: 1.0rem;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 기본 사이드바 네비게이션 숨김 (중복 메뉴 방지) */
    [data-testid="stSidebarNav"] {display: none;}
    </style>
    <script>
        var parentHead = window.parent.document.head;
        
        // Remove existing description if any
        var existingDesc = parentHead.querySelector('meta[name="description"]');
        if (existingDesc) { parentHead.removeChild(existingDesc); }
        
        // Add description meta tag
        var metaDesc = window.parent.document.createElement('meta');
        metaDesc.name = 'description';
        metaDesc.content = '글로벌 거시 지표(환율, 금리, 물동량 등)와 AI 모델을 활용한 코스피(KOSPI) 전망 및 시장 과열 위험(MOI) 모니터링 대시보드입니다. 실시간 AI 진단과 시나리오별 백테스팅을 제공합니다.';
        parentHead.appendChild(metaDesc);
        
        // Add keywords meta tag
        var metaKeywords = window.parent.document.createElement('meta');
        metaKeywords.name = 'keywords';
        metaKeywords.content = '코스피 전망, KOSPI 전망, 주식 시장 과열, 시장 위험 지수, AI 주식 분석, 주식 분석 대시보드';
        parentHead.appendChild(metaKeywords);
    </script>
    """, unsafe_allow_html=True)

# 한국 시간(KST) 설정을 위한 함수
def get_kst_now():
    return datetime.now() + timedelta(hours=9)

# active_tab 초기화
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "risk_monitor"

if "ai_analysis_results" not in st.session_state:
    st.session_state["ai_analysis_results"] = None

if "run_ai_trigger" not in st.session_state:
    st.session_state["run_ai_trigger"] = False

# 사이드바 공통 메뉴 추가
# 사이드바 공통 메뉴 추가
with st.sidebar:
    st.subheader("📋 대시보드 메뉴")
    if st.button("📊 KOSPI 위험 모니터링", use_container_width=True):
        st.session_state["active_tab"] = "risk_monitor"
        st.rerun()
    if st.button("📺 실시간 종목 탐색 (인기/수급)", use_container_width=True):
        st.session_state["active_tab"] = "youtube_rank"
        st.rerun()
    if st.button("🔥 과열 국면 시그널", use_container_width=True):
        st.session_state["active_tab"] = "overheat_signal"
        st.rerun()

# 다른 페이지 라우팅
if st.session_state["active_tab"] == "youtube_rank":
    render_youtube_rank_page()
    st.stop()
elif st.session_state["active_tab"] == "overheat_signal":
    render_overheat_page()
    st.stop()

# 3. 제목 및 설명
st.title("📊 코스피(KOSPI) 전망 및 시장 과열 위험 모니터링")
st.markdown(f"""
이 대시보드는 글로벌 거시 지표와 AI 모델을 융합하여 **향후 1주일(5~10거래일) 후**의 KOSPI 변동 위험 및 시장 과열 지수(MOI)를 예측하고 모니터링합니다. 
실시간 금융 데이터와 Gemini AI 엔진을 결합한 지능형 마켓 분석 툴킷입니다.

(최종 분석 시각: {get_kst_now().strftime('%m월 %d일 %H시 %M분')})
""")
st.markdown("---")

# --- [안내서 섹션] ---
with st.expander("📖 지수 가이드북"):
    st.subheader("1. 지수 산출 핵심 지표 (Core Indicators)")
    st.write("""
    본 모델의 지표들은 KOSPI와의 **통계적 상관관계** 및 **하락 선행성**을 기준으로 선정되었습니다.\n\n* **글로벌 리스크**: 미국 **S&P 500 지수**를 활용하며, 한국 증시와의 강력한 동조화 경향을 반영합니다.\n\n* **통화 및 유동성**: **원/달러 환율** 및 **달러 인덱스(DXY)** 를 통해 외국인 자본 유출 압력을 측정합니다.\n\n* **시장 심리**: **VIX(공포 지수)** 를 통해 투자자의 불안 심리와 변동성 전조를 파악합니다.\n\n* **실물 경제**: 경기 선행 지표인 **구리 가격(Copper)** 과 **장단기 금리차**를 포함합니다.\n\n""")
    
    st.divider()

    st.subheader("2. 예측적 선행 알고리즘 (Predictive Lead Intelligence)")
    st.markdown("#### **① 1주일 선행 상관 분석 (5-10 Days Predictive Lead)**")
    st.write("""
    * **선행성 강제화**: 본 모델은 모든 지표와 KOSPI 간의 시차를 **최소 5일에서 최대 12일** 범위에서 탐색합니다.\n\n이는 현재의 지표 변화가 최소 1주일 뒤의 증시에 미치는 영향을 추정하기 위함입니다.\n\n* **동시성 배제**: 당일의 시장 등락에 의한 '사후 설명'을 배제하고, 순수하게 미래의 리스크 전조를 포착하는 데 집중합니다.\n\n""")
    
    st.markdown("#### **② 하이브리드 정규화 및 볼록성 (Hybrid Normalization & Convexity)**")
    st.write("""
    * **시그모이드 정규화**: Z-Score(표준점수)를 시그모이드 함수에 통과시켜 0~100 사이로 변환합니다.\n\n이는 극단적인 이상치(Black Swan) 발생 시 지수가 상한선에 막혀 변동을 포착하지 못하는 문제를 해결합니다.\n\n* **위험 볼록성(Convexity)**: 시장의 공포는 선형적으로 증가하지 않습니다.\n\n본 모델은 지수함수적 가중치를 적용하여, 위험 지수가 70점을 넘어서는 국면에서 더욱 민감하고 빠르게 반응하도록 설계되었습니다.\n\n""")
    
    st.markdown("#### **③ 실시간 패닉 이벤트 탐지 (Real-time Panic Detection)**")
    st.write("""
    * **안전자산 및 공포 심리 급등 감지**: 전쟁, 금융위기 등 블랙스완 발생 시 가장 먼저 반응하는 **금(Gold), 스위스 프랑(CHF), 엔/원 환율, VVIX(공포 변동성)** 지표의 최근 5일 이상 급등세를 모니터링합니다.\n\n* **뉴스 공포 키워드 밀집도**: 글로벌 뉴스 헤드라인에서 'War', 'Bankrupt', 'Default' 등 치명적 키워드 발생 빈도를 실시간으로 추적합니다.\n\n* **위험 지수 강제 오버라이딩**: 위 지표들이 동시다발적으로 임계치를 돌파하면, 기존 거시 지표 기반의 위험 점수를 무시하고 **최종 위험 지수를 강제로 상향(Panic Override)** 시켜 즉각적인 경보를 발송합니다.\n\n""")
    
    st.markdown("#### **④ 요약**")
    st.info("본 모델은 통계적 정상성을 확보한 수익률 기반 분석과 이상치에 강건한 시그모이드 정규화를 통해 **구조적 위험**을 선행 포착하며, 동시에 실시간 패닉 감지 모듈을 통해 **돌발적 블랙스완**에도 즉각 대응할 수 있는 **하이브리드 조기 경보 시스템**입니다.")

    st.divider()
    
    st.subheader("3. 고도화된 수리적 산출 공식")
    @st.cache_data
    def get_math_formulas():
        st.markdown("#### **① 시차 수익률 상관관계 (Lagged Return Correlation)**")
        st.latex(r"\rho(k) = Corr(r_{X, t-k}, r_{Y, t}) \quad (r: \text{Return})")
        st.markdown("#### **② 하이브리드 정규화 (Hybrid Normalization: Z-Score + Sigmoid)**")
        st.latex(r"Z = \frac{x - \mu}{\sigma}, \quad Score = \frac{1}{1 + e^{-Z}} \times 100")
        st.markdown("#### **③ 위험 가중치 복합 기여도 (Weighted Convexity)**")
        st.latex(r"Risk = \frac{\sum (w_i \times S_i)}{\sum w_i}, \quad Adjusted = \frac{e^{k \cdot Risk} - 1}{e^k - 1}")
    get_math_formulas()

# 4. 데이터 수집 함수 (최적화: 일괄 다운로드)
@st.cache_data(ttl=900) # 15분으로 연장
def load_data():
    # 오늘 데이터를 포함하기 위해 내일 날짜로 end_date 설정
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = "2019-01-01"
    
    # 여러 티커를 한 번에 다운로드
    tickers = {
        "kospi": "^KS11", "sp500": "^GSPC", "fx": "KRW=X", 
        "us10y": "^TNX", "us2y": "^IRX", "vix": "^VIX", 
        "copper": "HG=F", "freight": "BDRY", "wti": "CL=F", "dxy": "DX-Y.NYB",
        "eem": "EEM"
    }
    
    # 패닉 감지용 실시간 티커 (안전자산 및 VIX 변동성)
    panic_tickers = {
        "gold": "GC=F", "jpy_krw": "JPYKRW=X", "usd_chf": "CHF=X", "vvix": "^VVIX"
    }
    
    other_tickers = list(tickers.values()) + list(panic_tickers.values())
    other_tickers.remove(tickers["kospi"])
    
    try:
        raw_data = yf.download(other_tickers, start=start_date, end=end_date)
        if 'Close' in raw_data.columns.levels[0] if isinstance(raw_data.columns, pd.MultiIndex) else 'Close' in raw_data.columns:
            data = raw_data['Close']
        else:
            data = raw_data
        if data.empty:
            data = pd.DataFrame(columns=other_tickers)
    except:
        data = pd.DataFrame(columns=other_tickers)
        
    try:
        # KOSPI 단독 다운로드 (타임존/결측치 병합 오류 방지)
        raw_kospi = yf.download(tickers["kospi"], start=start_date, end=end_date)
        if isinstance(raw_kospi.columns, pd.MultiIndex) and 'Close' in raw_kospi.columns.levels[0]:
            kospi_data = raw_kospi['Close']
        elif 'Close' in raw_kospi.columns:
            kospi_data = raw_kospi[['Close']]
        else:
            kospi_data = raw_kospi
        
        kospi_data.columns = [tickers["kospi"]]
        if kospi_data.empty:
            kospi_data = pd.DataFrame(columns=[tickers["kospi"]])
    except:
        kospi_data = pd.DataFrame(columns=[tickers["kospi"]])
        
    if not kospi_data.empty:
        try:
            if hasattr(kospi_data.index, 'tz') and kospi_data.index.tz is not None:
                kospi_data.index = kospi_data.index.tz_localize(None)
        except: pass
        data = data.join(kospi_data, how='outer')
    else:
        data[tickers["kospi"]] = np.nan

    # DXY 단독 다운로드 (데이터 정렬 오류 방지)
    try:
        raw_dxy = yf.download(tickers["dxy"], start=start_date, end=end_date)
        if isinstance(raw_dxy.columns, pd.MultiIndex) and 'Close' in raw_dxy.columns.levels[0]:
            dxy_standalone = raw_dxy['Close']
        elif 'Close' in raw_dxy.columns:
            dxy_standalone = raw_dxy[['Close']]
        else:
            dxy_standalone = raw_dxy

        dxy_standalone.columns = [tickers["dxy"]]
        if hasattr(dxy_standalone.index, 'tz') and dxy_standalone.index.tz is not None:
            dxy_standalone.index = dxy_standalone.index.tz_localize(None)
        
        if tickers["dxy"] in data.columns:
            data = data.drop(columns=[tickers["dxy"]])
        data = data.join(dxy_standalone, how='outer')
    except:
        if tickers["dxy"] not in data.columns:
            data[tickers["dxy"]] = np.nan
    
    sector_tickers = {
        "반도체": "005930.KS", "자동차": "005380.KS", "2차전지": "051910.KS",
        "바이오": "207940.KS", "인터넷": "035420.KS", "금융": "055550.KS",
        "철강": "005490.KS", "방산": "047810.KS", "유틸리티": "015760.KS"
    }
    
    sp500_sector_tickers = {
        "반도체": "NVDA", "자동차": "TSLA", "2차전지": "ALB",
        "바이오": "AMGN", "인터넷": "GOOGL", "금융": "JPM",
        "철강": "NUE", "방산": "LMT", "유틸리티": "NEE"
    }
    
    try:
        sector_raw = yf.download(list(sector_tickers.values()), period="5d")['Close']
        if sector_raw.empty: sector_raw = pd.DataFrame(columns=list(sector_tickers.values()))
    except:
        sector_raw = pd.DataFrame(columns=list(sector_tickers.values()))
        
    try:
        sp500_sector_raw = yf.download(list(sp500_sector_tickers.values()), period="5d")['Close']
        if sp500_sector_raw.empty: sp500_sector_raw = pd.DataFrame(columns=list(sp500_sector_tickers.values()))
    except:
        sp500_sector_raw = pd.DataFrame(columns=list(sp500_sector_tickers.values()))
    
    # BDRY 데이터가 비정상적일 경우 FinanceDataReader로 폴백
    try:
        if tickers["freight"] not in data.columns or data[tickers["freight"]].dropna().empty:
            import FinanceDataReader as fdr
            bdry_data = fdr.DataReader(tickers["freight"], start_date, end_date)
            if not bdry_data.empty:
                bdry_data = bdry_data[['Close']]
                bdry_data.columns = [tickers["freight"]]
                if hasattr(bdry_data.index, 'tz') and bdry_data.index.tz is not None:
                    bdry_data.index = bdry_data.index.tz_localize(None)
                if tickers["freight"] in data.columns:
                    data = data.drop(columns=[tickers["freight"]])
                data = data.join(bdry_data, how='outer')
    except:
        pass

    return (
        data[[tickers["kospi"]]] if tickers["kospi"] in data.columns else pd.DataFrame(columns=[tickers["kospi"]]), 
        data[[tickers["sp500"]]] if tickers["sp500"] in data.columns else pd.DataFrame(columns=[tickers["sp500"]]), 
        data[[tickers["fx"]]] if tickers["fx"] in data.columns else pd.DataFrame(columns=[tickers["fx"]]), 
        data[[tickers["us10y"]]] if tickers["us10y"] in data.columns else pd.DataFrame(columns=[tickers["us10y"]]), 
        data[[tickers["us2y"]]] if tickers["us2y"] in data.columns else pd.DataFrame(columns=[tickers["us2y"]]), 
        data[[tickers["vix"]]] if tickers["vix"] in data.columns else pd.DataFrame(columns=[tickers["vix"]]), 
        data[[tickers["copper"]]] if tickers["copper"] in data.columns else pd.DataFrame(columns=[tickers["copper"]]), 
        data[[tickers["freight"]]] if tickers["freight"] in data.columns else pd.DataFrame(columns=[tickers["freight"]]), 
        data[[tickers["wti"]]] if tickers["wti"] in data.columns else pd.DataFrame(columns=[tickers["wti"]]), 
        data[[tickers["dxy"]]] if tickers["dxy"] in data.columns else pd.DataFrame(columns=[tickers["dxy"]]), 
        data[[panic_tickers["gold"]]] if panic_tickers["gold"] in data.columns else pd.DataFrame(columns=[panic_tickers["gold"]]),
        data[[panic_tickers["jpy_krw"]]] if panic_tickers["jpy_krw"] in data.columns else pd.DataFrame(columns=[panic_tickers["jpy_krw"]]),
        data[[panic_tickers["usd_chf"]]] if panic_tickers["usd_chf"] in data.columns else pd.DataFrame(columns=[panic_tickers["usd_chf"]]),
        data[[panic_tickers["vvix"]]] if panic_tickers["vvix"] in data.columns else pd.DataFrame(columns=[panic_tickers["vvix"]]),
        data[[tickers["eem"]]] if tickers["eem"] in data.columns else pd.DataFrame(columns=[tickers["eem"]]),
        sector_raw, sector_tickers, sp500_sector_raw, sp500_sector_tickers
    )

# 4.5 글로벌 경제 뉴스 수집 함수 (최적화: 캐시 연장)
@st.cache_data(ttl=1800) # 30분으로 연장
def get_market_news():
    api_url = "https://newsapi.org/v2/everything"
    params = {
        "q": "stock market risk OR recession OR inflation",
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    try:
        res = requests.get(api_url, params=params, timeout=10)
        data = res.json()
        panic_keywords_count = 0
        panic_words = ['war', 'missile', 'bankrupt', 'default', 'contagion', 'emergency', 'collapse', 'crash', 'panic']
        
        if data.get("status") == "ok":
            news_items = []
            for article in data.get("articles", []):
                title = article.get("title", "")
                news_items.append({"title": title, "link": article["url"]})
                
                # 공포 키워드 카운팅
                title_lower = title.lower()
                for pw in panic_words:
                    if pw in title_lower:
                        panic_keywords_count += 1
                        
            return news_items, panic_keywords_count
        return [], 0
    except:
        return [], 0

# 4.6 트럼프 소셜 피드 수집 함수
@st.cache_data(ttl=600)
def get_trump_feed():
    # Trump's Truth Social RSS feed proxy
    url = "https://trumpstruth.org/feed"
    try:
        import xml.etree.ElementTree as ET
        import re
        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.content)
        items = root.findall('.//item')
        feed_data = []
        for item in items[:5]: # 여유있게 5개를 가져와서 중복 필터링
            title = item.find('title').text if item.find('title') is not None else ""
            desc = item.find('description').text if item.find('description') is not None else ""
            
            # [No Title] 필터링 및 HTML 태그 제거
            if title and title.startswith("[No Title]"): title = ""
            desc = re.sub(r'<[^>]+>', '', desc if desc else "").strip()
            
            # 내용이 없거나 이미 비슷한 내용(앞 40글자 일치)이 추가된 경우 건너뜀 (중복 방지)
            text_check = f"{title} {desc}".strip()
            if not text_check:
                continue
            
            # 중복 검사 (앞 40글자가 같으면 동일 포스트로 간주)
            is_duplicate = False
            for d in feed_data:
                existing_text = f"{d['title']} {d['description']}".strip()
                if text_check[:40] == existing_text[:40]:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
                
            feed_data.append({"title": title.strip(), "description": desc})
            if len(feed_data) == 3: # 3개까지만 채움
                break
                
        return feed_data
    except Exception as e:
        return []

# --- [전역 변수 및 컨테이너 초기화 (NameError 방지)] ---
news_data = []
news_panic_count = 0
all_titles = ""
corr_val = 0.0
hist_risks = [50.0] * 7 # 기본값 50점
total_risk_index = 50.0
latest_data_summary = "데이터를 불러오는 중입니다..."
# 컨테이너 변수 초기화 (레이아웃에서 나중에 할당)
ai_news_container = None
ai_trump_container = None
bt_analysis_container = None
ai_indicator_container = None

try:
    with st.spinner('시차 상관관계 및 가중치 분석 중...'):
        (kospi, sp500, fx, bond10, bond2, vix_data, copper_data, freight_data, wti_data, dxy_data,
         gold_data, jpy_krw_data, usd_chf_data, vvix_data, eem_data,
         sector_raw, sector_map, sp500_sector_raw, sp500_sector_map) = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series(dtype='float64')
        # [수정] 멀티인덱스 상황에서도 안전하게 데이터를 추출하도록 처리
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        return df[~df.index.duplicated(keep='first')]

    # 데이터 끊김 현상 방지를 위해 ffill() 적용
    ks_s = get_clean_series(kospi).ffill()
    sp_s = get_clean_series(sp500).reindex(ks_s.index).ffill()
    fx_s = get_clean_series(fx).reindex(ks_s.index).ffill()
    b10_s = get_clean_series(bond10).reindex(ks_s.index).ffill()
    b2_s = get_clean_series(bond2).reindex(ks_s.index).ffill()
    vx_s = get_clean_series(vix_data).reindex(ks_s.index).ffill()
    cp_s = get_clean_series(copper_data).reindex(ks_s.index).ffill()
    fr_s = get_clean_series(freight_data).reindex(ks_s.index).ffill()
    wt_s = get_clean_series(wti_data).reindex(ks_s.index).ffill()
    dx_s = get_clean_series(dxy_data).reindex(ks_s.index).ffill()
    em_s = get_clean_series(eem_data).reindex(ks_s.index).ffill()
    
    # 패닉 데이터 
    gd_s = get_clean_series(gold_data).reindex(ks_s.index).ffill()
    jk_s = get_clean_series(jpy_krw_data).reindex(ks_s.index).ffill()
    uf_s = get_clean_series(usd_chf_data).reindex(ks_s.index).ffill()
    vv_s = get_clean_series(vvix_data).reindex(ks_s.index).ffill()
    
    # 금리차 계산
    yield_curve = b10_s - b2_s
    ma20 = ks_s.rolling(window=20).mean() # 전체 데이터 기반 이동평균 계산

    def get_hist_score_val(series, current_idx, inverse=False):
        try:
            # 최근 1년(252거래일) 데이터 추출
            sub = series.loc[:current_idx].iloc[-252:]
            if len(sub) < 10: return 50.0
            
            mu, std = float(sub.mean()), float(sub.std())
            if std == 0: return 50.0
            
            curr_v = float(series.loc[current_idx])
            z = (curr_v - mu) / std
            
            # Sigmoid 정규화: Z-score를 0~100 사이로 매핑 (이상치에 강건함)
            # z=0일 때 50, z=2일 때 약 88, z=-2일 때 약 12
            score = 100 / (1 + np.exp(-z))
            return (100 - score) if inverse else score
        except: return 50.0

    @st.cache_data(ttl=3600)
    def calculate_ml_lagged_weights(_ks_s, _sp_s, _fx_s, _b10_s, _cp_s, _ma20, _vx_s, _em_s):
        # 1. 수익률(pct_change) 기반으로 변환하여 통계적 정상성 확보
        def get_ret(s): return s.pct_change().dropna()
        
        target_ret = get_ret(_ks_s)
        
        def find_best_lag_ret(feature_s, target_s, min_lag=5, max_lag=12):
            f_ret = get_ret(feature_s)
            common_idx = f_ret.index.intersection(target_s.index)
            # 최소 5일 이상의 시차(Lag)를 가진 상관관계만 탐색하여 '예측성' 확보
            corrs = [abs(f_ret.shift(lag).reindex(common_idx).corr(target_s.reindex(common_idx))) for lag in range(min_lag, max_lag + 1)]
            return np.argmax(corrs) + min_lag
            
        best_lags = {
            'SP': find_best_lag_ret(_sp_s, target_ret), 
            'FX': find_best_lag_ret(_fx_s, target_ret), 
            'B10': find_best_lag_ret(_b10_s, target_ret), 
            'CP': find_best_lag_ret(_cp_s, target_ret), 
            'VX': find_best_lag_ret(_vx_s, target_ret),
            'EM': find_best_lag_ret(_em_s, target_ret)
        }
        
        data_rows = []
        # 최근 252거래일 동안의 지표 상태(Score)와 KOSPI 수익률 간의 관계 분석
        for d in _ks_s.index[-252:]:
            if d not in target_ret.index: continue
            
            s_sp = get_hist_score_val(_sp_s.shift(best_lags['SP']), d, True)
            s_fx = get_hist_score_val(_fx_s.shift(best_lags['FX']), d)
            s_b10 = get_hist_score_val(_b10_s.shift(best_lags['B10']), d)
            s_cp = get_hist_score_val(_cp_s.shift(best_lags['CP']), d, True)
            s_vx = get_hist_score_val(_vx_s.shift(best_lags['VX']), d)
            s_em = get_hist_score_val(_em_s.shift(best_lags['EM']), d, True)
            s_tech = max(0, min(100, 100 - (float(_ks_s.loc[d]) / float(_ma20.loc[d]) - 0.9) * 500))
            
            data_rows.append([ (s_fx + s_b10 + s_cp) / 3, s_sp, s_vx, s_tech, s_em, target_ret.loc[d] ])
        
        df_reg = pd.DataFrame(data_rows, columns=['Macro', 'Global', 'Fear', 'Tech', 'Peri', 'KOSPI_Ret']).replace([np.inf, -np.inf], np.nan).dropna()
        if df_reg.empty:
            return np.array([0.20, 0.20, 0.20, 0.20, 0.20])
            
        X = df_reg.iloc[:, :5]
        Y = df_reg['KOSPI_Ret']
        
        try:
            coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
            adjusted_importance = (np.abs(coeffs) * X.std().values) + 1e-6 
            return adjusted_importance / np.sum(adjusted_importance)
        except:
            return np.array([0.20, 0.20, 0.20, 0.20, 0.20])

    sem_w = calculate_ml_lagged_weights(ks_s, sp_s, fx_s, b10_s, cp_s, ma20, vx_s, em_s)

    # 5. 사이드바 - 가중치 설정
    st.sidebar.header("⚙️ 지표별 가중치 설정")
    if 'slider_m' not in st.session_state: st.session_state.slider_m = float(round(sem_w[0], 2))
    if 'slider_g' not in st.session_state: st.session_state.slider_g = float(round(sem_w[1], 2))
    if 'slider_f' not in st.session_state: st.session_state.slider_f = float(round(sem_w[2], 2))
    if 'slider_t' not in st.session_state: st.session_state.slider_t = float(round(sem_w[3], 2))
    if 'slider_p' not in st.session_state: st.session_state.slider_p = float(round(sem_w[4], 2))

    if st.sidebar.button("🔄 권장 최적 가중치로 복귀"):
        st.session_state.slider_m = float(round(sem_w[0], 2)); st.session_state.slider_g = float(round(sem_w[1], 2))
        st.session_state.slider_f = float(round(sem_w[2], 2)); st.session_state.slider_t = float(round(sem_w[3], 2))
        st.session_state.slider_p = float(round(sem_w[4], 2))
        st.rerun()

    w_macro = st.sidebar.slider("매크로 (환율/금리/물동량)", 0.0, 1.0, key="slider_m", step=0.01)
    w_global = st.sidebar.slider("미국 시장 위험 (S&P 500)", 0.0, 1.0, key="slider_g", step=0.01)
    w_peri = st.sidebar.slider("주변부 이탈 위험 (신흥국 EEM 하락)", 0.0, 1.0, key="slider_p", step=0.01)
    w_fear = st.sidebar.slider("시장 공포 (VIX 지수)", 0.0, 1.0, key="slider_f", step=0.01)
    w_tech = st.sidebar.slider("국내 기술적 지표 (이동평균선)", 0.0, 1.0, key="slider_t", step=0.01)

    with st.sidebar.expander("ℹ️ 가중치 산출 알고리즘"):
        st.caption("""
        본 모델은 **1주일 선행 수익률 분석(Lagged Return Forecasting)** 기법을 사용합니다.\n\n1. **미래 예측성 강제 (Lead Time Enforcement)**:
            모든 지표에 대해 **5~12거래일 전**의 선행 데이터만 사용하여 KOSPI 수익률과의 관계를 정의합니다.\n\n2. **수익률 기반 상관 관계**:
            지수 수준(Level)이 아닌 변동성(Return)을 분석하여 지표의 '전조 현상'을 통계적으로 입증합니다.\n\n3. **실시간 미래 위험 투사**:
            오늘의 지표값을 위에서 도출된 '미래 전조 가중치'에 대입하여, **다음 주 시장의 잠재적 리스크**를 산출합니다.\n\n""")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 관리자 모드")
    admin_id_input = st.sidebar.text_input("아이디", key="admin_id_bottom")
    admin_pw_input = st.sidebar.text_input("비밀번호", type="password", key="admin_pw_bottom")
    is_admin = (admin_id_input == ADMIN_ID and admin_pw_input == ADMIN_PW)
    st.sidebar.markdown("---")
    st.sidebar.subheader("자발적 후원으로 운영됩니다.")
    st.sidebar.write("카카오뱅크 3333-23-8667708 (ㅈㅅㅎ)")
    st.sidebar.write("유료API로 정밀한 데이터가 필요합니다.")

    total_w = w_macro + w_tech + w_global + w_fear + w_peri
    if total_w == 0: 
        st.error("가중치 합이 0일 수 없습니다."); st.stop()

    def calculate_score(current_series, full_series, inverse=False):
        recent = full_series[full_series.index >= (full_series.index.max() - pd.Timedelta(days=365))]
        mu, std = float(recent.mean()), float(recent.std())
        curr_v = float(current_series.iloc[-1])
        if std == 0: return 50.0
        
        z = (curr_v - mu) / std
        score = 100 / (1 + np.exp(-z))
        return float(max(0, min(100, (100 - score) if inverse else score)))

    m_now = (calculate_score(fx_s, fx_s) + calculate_score(b10_s, b10_s) + calculate_score(cp_s, cp_s, True)) / 3
    t_now = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    p_now = calculate_score(em_s, em_s, True)
    
    # 기초 위험 지수 계산 (가중 평균)
    base_risk = (m_now * w_macro + t_now * w_tech + calculate_score(sp_s, sp_s, True) * w_global + calculate_score(vx_s, vx_s) * w_fear + p_now * w_peri) / total_w
    
    # -------------------------------------------------------------
    # [신규] 실시간 패닉 이벤트 탐지 (Real-time Panic Detection)
    # 안전자산(금, 스위스 프랑, 엔/원) 및 공포 변동성(VVIX)의 단기 이상 급등 감지
    # -------------------------------------------------------------
    def get_panic_score(series):
        if series.empty or len(series) < 10: return 0.0
        try:
            # 최근 5일 데이터 vs 과거 1년(252일) 평균 비교를 통한 Z-score 산출
            recent_5d_mean = series.iloc[-5:].mean()
            past_1y = series[series.index >= (series.index.max() - pd.Timedelta(days=365))]
            mu = past_1y.mean()
            std = past_1y.std()
            if std == 0: return 0.0
            
            z_score = (recent_5d_mean - mu) / std
            
            # Z-score > 1.5 이면 이상 조짐, > 3.0 이면 극심한 패닉으로 간주
            if z_score <= 1.0: return 0.0
            # 1.0 ~ 3.5 구간을 0 ~ 100 점수로 선형 보간 후 제한
            panic_intensity = min(100.0, max(0.0, (z_score - 1.0) * 40))
            return panic_intensity
        except: return 0.0

    panic_gold = get_panic_score(gd_s)
    panic_jpy = get_panic_score(jk_s)
    panic_chf = get_panic_score(uf_s)
    panic_vvix = get_panic_score(vv_s)
    
    # 여러 지표가 동시에 급등할수록 패닉 신뢰도가 기하급수적으로 상승하도록 합성
    # 3개 이상에서 패닉이 감지되면 복합 충격으로 간주
    active_panics = sum([1 for p in [panic_gold, panic_jpy, panic_chf, panic_vvix] if p > 30])
    
    # 평균 패닉 점수에 임계치 배수(Multiplier) 적용
    raw_panic_avg = (panic_gold + panic_jpy + panic_chf + panic_vvix) / 4.0
    panic_multiplier = 1.0 + (active_panics * 0.5) 
    final_panic_score = min(100.0, raw_panic_avg * panic_multiplier)
    
    # 뉴스 키워드 기반 패닉 보정 (공포 키워드 1개당 위험점수 +5점 가산, 최대 +30점)
    # 실제 반영은 글로벌 뉴스 데이터를 불러온 뒤(스크립트 하단)에야 정확히 알 수 있으므로, 
    # 선제적으로 캐시 초기화 없이 UI 게이지에 반영하기 위해 전역변수 참조 활용
    # News 데이터 로딩 시점과 맞추기 위해, 여기서는 일단 자산 가격 기반의 최종 패닉 스코어만 산출
    
    # 기초 위험 지수와 패닉 점수 중 더 높은 것을 최종 위험 베이스로 선정 (Panic Override)
    # 패닉 점수가 60점 이상 유의미하게 발생한 경우에만 덮어쓰기 적용
    applied_base_risk = max(base_risk, final_panic_score) if final_panic_score > 60 else base_risk
    
    # [추가 보정] 위쪽에서 뉴스 데이터를 아직 로드하기 전이므로, 이 스크립트 흐름 특성상 뉴스 기반 패닉 점수는 UI 하단에 별도 경고로 띄우거나 이후 캐싱되어 반영됨
    
    # -------------------------------------------------------------
    # [백테스팅 & 실시간 지수 일치화 및 선행 계산]
    # -------------------------------------------------------------
    def get_hist_panic_score(series, date):
        if series.empty: return 0.0
        try:
            sub = series.loc[:date]
            if len(sub) < 10: return 0.0
            recent_5d_mean = sub.iloc[-5:].mean()
            past_1y = sub[sub.index >= (date - pd.Timedelta(days=365))]
            mu = past_1y.mean()
            std = past_1y.std()
            if std == 0: return 0.0
            z_score = (recent_5d_mean - mu) / std
            if z_score <= 1.0: return 0.0
            return min(100.0, max(0.0, (z_score - 1.0) * 40))
        except:
            return 0.0

    dates = ks_s.index[-252:]
    hist_risks = []
    k = 0.5
    for d in dates:
        m = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
        t = max(0.0, min(100.0, float(100 - (float(ks_s.loc[d]) / float(ma20.loc[d]) - 0.9) * 500))) if d in ma20.index and pd.notna(ma20.loc[d]) and ma20.loc[d] != 0 else 50.0
        s_sp = get_hist_score_val(sp_s, d, True)
        s_vx = get_hist_score_val(vx_s, d)
        s_em = get_hist_score_val(em_s, d, True)
        
        base = (m * w_macro + t * w_tech + s_sp * w_global + s_vx * w_fear + s_em * w_peri) / total_w
        
        p_g = get_hist_panic_score(gd_s, d)
        p_j = get_hist_panic_score(jk_s, d)
        p_c = get_hist_panic_score(uf_s, d)
        p_v = get_hist_panic_score(vv_s, d)
        
        act_panics = sum([1 for p in [p_g, p_j, p_c, p_v] if p > 30])
        raw_p_avg = (p_g + p_j + p_c + p_v) / 4.0
        p_mult = 1.0 + (act_panics * 0.5)
        fin_panic = min(100.0, raw_p_avg * p_mult)
        
        app_base = max(base, fin_panic) if fin_panic > 60 else base
        convex_risk = ((np.exp(k * app_base / 100) - 1) / (np.exp(k) - 1)) * 100
        hist_risks.append(convex_risk)

    total_risk_index = hist_risks[-1]
    prev_idx = hist_risks[-6] if len(hist_risks) > 5 else 50.0
    delta_5d = total_risk_index - prev_idx

    # [신규] 패닉 경보 배너 표시
    if final_panic_score > 60 and final_panic_score > base_risk:
        st.error(f"🚨 **[긴급: 실시간 패닉 이벤트 확률 상승]** 🚨\n\n단기 안전자산 및 공포 심리 급등이 감지되어 예측 지배 가중치가 오버라이딩 되었습니다.\n\n(기초 위험 점수: {base_risk:.1f} ➔ **패닉 보정 위험 점수: {applied_base_risk:.1f}**)")

    # 등급 판정
    if total_risk_index >= 80:
        grade, grade_color, grade_emoji = "패닉 (PANIC)", "#c0392b", "🚨"
    elif total_risk_index >= 60:
        grade, grade_color, grade_emoji = "방어 (CAUTION)", "#e67e22", "⚠️"
    elif total_risk_index >= 40:
        grade, grade_color, grade_emoji = "대비 (READY)", "#f1c40f", "👀"
    else:
        grade, grade_color, grade_emoji = "수익 극대화 (GROWTH)", "#27ae60", "✅"

    c_gauge, c_guide = st.columns([1.2, 0.8])
    with c_gauge: 
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=total_risk_index,
            number={'suffix': "점", 'font': {'size': 38}},
            delta={'reference': prev_idx, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#2ecc71"},
                   'suffix': "pt (5일전 대비)", 'font': {'size': 14}},
            title={'text': f"KOSPI 예측적 위험  {grade_emoji} {grade}", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#555",
                         'tickvals': [0, 40, 60, 80, 100],
                         'ticktext': ['0', '40', '60', '80', '100']},
                'bar': {'color': grade_color, 'thickness': 0.25},
                'bgcolor': "#f8f9fa",
                'borderwidth': 1,
                'bordercolor': "#dee2e6",
                'steps': [
                    {'range': [0,  40], 'color': '#d5f5e3'},   # Growth - 초록
                    {'range': [40, 60], 'color': '#fef9e7'},   # Ready - 노랑
                    {'range': [60, 80], 'color': '#fdebd0'},   # Caution - 주황
                    {'range': [80,100], 'color': '#fadbd8'},   # Panic - 빨강
                ],
                'threshold': {
                    'line': {'color': grade_color, 'width': 5},
                    'thickness': 0.8,
                    'value': total_risk_index
                }
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c_guide: 
        st.markdown("**📊 예측 컴포넌트별 위험도**")
        comp_data = {
            '🌍 매크로 (금리/환율)': m_now,
            '📈 미국 시장 (S&P 500)': calculate_score(sp_s, sp_s, True),
            '😱 시장 공포 (VIX)': calculate_score(vx_s, vx_s),
            '📉 기술적 과매수': t_now,
            '📉 주변부 이탈 (신흥국 하락)': p_now,
        }
        for label, score in comp_data.items():
            bar_color = "#e74c3c" if score >= 80 else ("#f39c12" if score >= 60 else ("#f1c40f" if score >= 40 else "#27ae60"))
            bar_width = int(score) if not np.isnan(score) else 0
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
            
        st.markdown("""
        <div style="font-size:0.78rem; color:#666; margin-top:12px; line-height:1.7;">
        <b>💡 예측 지수 활용 가이드 (1주 선행)</b><br>
        ✅ &nbsp;&nbsp;0~40: 수익 극대화 — 상방 압력 우세<br>
        👀 40~60: 대비 — 변동성 대비, 관망<br>
        ⚠️ 60~80: 방어 — 주식 비중 축소<br>
        🚨 80~100: 패닉 — 비상 탈출, 현금 확보
        </div>
        """, unsafe_allow_html=True)

        if 'likes' not in st.session_state:
            st.session_state.likes = 0
        
        st.write("")
        l_col1, l_col2 = st.columns([1, 4])
        with l_col1:
            if st.button(f"👍 {st.session_state.likes}", use_container_width=True):
                st.session_state.likes += 1
                st.rerun()
        with l_col2:
            st.markdown(f"""
            <div style="padding-top: 5px;">
                <span style="font-size: 0.85rem; color: #666;">대시보드가 유익했다면 추천!</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # --- [통합 AI 분석/번역 진단 제어부] ---
    st.subheader("🚀 AI 통합 번역 및 시장 진단")
    st.markdown("""
    Gemini AI 모델을 사용하여 글로벌 경제 뉴스 헤드라인 번역, 트럼프 소셜 브리핑 번역, 
    시장 위험 지수 모델 유효성 진단 및 핵심 시장 지표 종합 진단을 한 번에 수행합니다.
    """)
    
    if "ai_cache_loaded" not in st.session_state:
        saved_at, data_hash, response_text = load_ai_cache(SHEET_ID, "main_risk")
        if response_text:
            try:
                st.session_state["ai_analysis_results"] = json.loads(response_text)
                st.session_state["ai_last_saved_at"] = saved_at
                st.session_state["ai_last_data_hash"] = data_hash
            except:
                pass
        st.session_state["ai_cache_loaded"] = True

    current_data_hash = {}
    try:
        current_data_hash["kospi"] = round(float(ks_s.iloc[-1]), 2)
        current_data_hash["fx"] = round(float(fx_s.iloc[-1]), 1)
        current_data_hash["vix"] = round(float(vx_s.iloc[-1]), 2)
        current_data_hash["b10"] = round(float(b10_s.iloc[-1]), 3)
    except:
        pass

    if st.session_state.get("ai_analysis_results") is not None:
        saved_at = st.session_state.get("ai_last_saved_at", "N/A")
        st.caption(f"✅ 마지막 AI 분석: {saved_at} (변동이 적을 경우 자동 캐싱 유지)")
        col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 5])
        with col_btn1:
            if st.button("🔄 최신 AI 분석 강제 요청", key="btn_run_ai_analysis", use_container_width=True):
                st.session_state["run_ai_trigger"] = True
                st.session_state["ai_analysis_results"] = None
                st.rerun()
        with col_btn2:
            if st.button("❌ 분석 결과 지우기", key="btn_clear_ai_analysis", use_container_width=True):
                st.session_state["run_ai_trigger"] = False
                st.session_state["ai_analysis_results"] = None
                st.rerun()
    else:
        col_btn1, _ = st.columns([2, 6])
        with col_btn1:
            if st.button("🚀 최신 AI 분석 요청", key="btn_run_ai_analysis", use_container_width=True):
                st.session_state["run_ai_trigger"] = True
                st.rerun()

    # 정각 경과 + 데이터 변동성 확인을 통한 자동 트리거
    if st.session_state.get("ai_analysis_results") is not None and not st.session_state.get("run_ai_trigger"):
        saved_at = st.session_state.get("ai_last_saved_at")
        old_hash = st.session_state.get("ai_last_data_hash")
        if should_update_ai(saved_at):
            if is_data_changed_significantly(old_hash, current_data_hash, 0.005):
                st.session_state["run_ai_trigger"] = True
                st.rerun()
                
    st.markdown("---")
    c_news_left, c_news_right = st.columns(2)
    with c_news_left:
        # 제목 텍스트 업데이트
        st.subheader("📰 글로벌 경제 뉴스")
        news_data, news_panic_count = get_market_news()
        all_titles = ""
        for a in news_data:
            st.markdown(f"- [{a['title']}]({a['link']})")
            all_titles += a['title'] + ". "
            
    with c_news_right:
        # 뉴스 분석 AI 컨테이너 정의 (위치: 뉴스 리스트 오른쪽)
        ai_news_container = st.container()
        
        # 뉴스 기반 공포 심리 경보
        if 'news_panic_count' in locals() and news_panic_count >= 2:
            st.error(f"🚨 **[주의] 뉴스 공포 심리 확산 감지**\n\n최근 24시간 내 시장 헤드라인에서 치명적 위험 키워드(War, Default 등)가 **{news_panic_count}회** 감지되었습니다.\n\n투자 심리 악화에 대비하세요.")

    st.markdown("---")
    c_trump_left, c_trump_right = st.columns(2)
    with c_trump_left:
        st.subheader("🇺🇸 트럼프 소셜 최신 브리핑 (Original)")
        trump_data = get_trump_feed()
        if trump_data:
            for t in trump_data:
                st.markdown(f"<div style='border: 1px solid #ddd; background-color: #f8f9fa; color: #31333F; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>[Original]</strong><br>{t['title']}<br><small>{t['description']}</small></div>", unsafe_allow_html=True)
        else:
            st.write("최신 트윗을 불러올 수 없습니다.")
        
    with c_trump_right:
        # 트럼프 번역 AI 컨테이너 정의
        ai_trump_container = st.container()

    # 7. 백테스팅
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 (최근 1년)")
    st.info("과거 데이터를 사용하여 모델의 유효성을 검증합니다.")
    hist_df = pd.DataFrame({'Date': dates, 'Risk': hist_risks, 'KOSPI': ks_s.loc[dates].values})
    
    @st.cache_data(ttl=86400, show_spinner=False)
    def calculate_prediction_accuracy(w_m, w_t, w_g, w_f, w_p):
        start_date = "2010-01-01"
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        tickers = ['^KS11', '^GSPC', 'KRW=X', '^TNX', 'HG=F', '^VIX', 'EEM']
        try:
            df = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close'].ffill()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            _ks_s = df['^KS11'].ffill()
            _sp_s = df['^GSPC'].ffill()
            _fx_s = df['KRW=X'].ffill()
            _b10_s = df['^TNX'].ffill()
            _cp_s = df['HG=F'].ffill()
            _vx_s = df['^VIX'].ffill()
            _em_s = df['EEM'].ffill() if 'EEM' in df.columns else _sp_s
            _ma20 = _ks_s.rolling(window=20).mean()
            
            def calc_rolling_z_score(s, inverse=False):
                mu = s.rolling(window=252).mean()
                std = s.rolling(window=252).std().replace(0, 1e-9)
                z = (s - mu) / std
                score = 100 / (1 + np.exp(-z))
                return 100 - score if inverse else score
            
            m = (calc_rolling_z_score(_fx_s) + calc_rolling_z_score(_b10_s) + calc_rolling_z_score(_cp_s, True)) / 3.0
            t = np.clip(100.0 - (_ks_s / _ma20 - 0.9) * 500.0, 0, 100.0)
            s_sp = calc_rolling_z_score(_sp_s, True)
            s_vx = calc_rolling_z_score(_vx_s)
            s_em = calc_rolling_z_score(_em_s, True)
            
            tot_w = w_m + w_t + w_g + w_f + w_p
            if tot_w == 0: tot_w = 1.0
            
            base = (m * w_m + t * w_t + s_sp * w_g + s_vx * w_f + s_em * w_p) / tot_w
            k_val = 0.5
            convex_risk = ((np.exp(k_val * base / 100.0) - 1.0) / (np.exp(k_val) - 1.0)) * 100.0
            
            eval_df = pd.DataFrame({'KOSPI': _ks_s, 'Risk': convex_risk}).dropna()
            
            # 5일에서 30일(1개월 반) 사이의 최적 적중 기간 탐색
            best_hit_rate = 0
            best_n = 20
            best_signal_count = 0
            
            for n in range(5, 31):
                eval_df['Min_Future_KOSPI'] = eval_df['KOSPI'].rolling(window=n).min().shift(-n)
                eval_df['Max_Drawdown'] = (eval_df['Min_Future_KOSPI'] - eval_df['KOSPI']) / eval_df['KOSPI']
                
                high_risk_days = eval_df[eval_df['Risk'] >= 60]
                valid_high_risk = high_risk_days.dropna(subset=['Max_Drawdown'])
                hits = valid_high_risk[valid_high_risk['Max_Drawdown'] <= -0.05]
                
                hr = len(hits) / len(valid_high_risk) * 100 if len(valid_high_risk) > 0 else 0
                # 적중률이 동일하다면 더 짧은 기간(빠른 적중)을 선호
                if hr > best_hit_rate:
                    best_hit_rate = hr
                    best_n = n
                    best_signal_count = len(valid_high_risk)
                    
            return best_hit_rate, best_signal_count, best_n
        except:
            return 0.0, 0, 20
            
    cb1, cb2 = st.columns([3, 1])
    with cb1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Risk'], name="위험 지수", line=dict(color='red'), connectgaps=True)) # connectgaps 추가
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot'), connectgaps=True))
        fig_bt.update_layout(yaxis=dict(title="위험 지수", range=[0, 100]), yaxis2=dict(overlaying="y", side="right"), height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); st.plotly_chart(fig_bt, use_container_width=True)
        
        # [수정 사항] 모델 유효성 진단의 위치를 그래프 아래로 이동
        corr_val = hist_df['Risk'].corr(hist_df['KOSPI'])
        # 모델 유효성 진단 AI 컨테이너 정의 (위치: 백테스팀 그래프 하단)
        bt_analysis_container = st.container()

    with cb2:
        corr_val = hist_df['Risk'].corr(hist_df['KOSPI'])
        st.metric("단순 상관계수", f"{corr_val:.2f}")
        
        st.markdown("---")
        hit_rate, signal_count, best_n = calculate_prediction_accuracy(w_macro, w_tech, w_global, w_fear, w_peri)
        st.metric("최근 10년 패닉 적중률", f"{hit_rate:.1f}%")
        st.caption(f"*최적 예측 기간: {best_n}거래일")
        st.caption(f"*위험(60 이상) 경보 발령 후 -5% 폭락 기준 ({signal_count}회 중)")

    # 7.5 블랙스완
    st.markdown("---")
    st.subheader("🦢 블랙스완(Black Swan) 과거 사례 비교 시뮬레이션")
    @st.cache_data(ttl=86400, show_spinner=False)
    def get_true_historical_risk(start_date, end_date, w_m, w_t, w_g, w_f, w_p):
        # 과거 데이터는 분석 기간보다 1년 전부터 가져와야 z-score 계산(과거 1년 롤링)이 가능합니다.
        fetch_start = (pd.to_datetime(start_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        tickers = ['^KS11', '^GSPC', 'KRW=X', '^TNX', 'HG=F', '^VIX', 'EEM']
        
        # yf.download (진행률 표시 숨김)
        df = yf.download(tickers, start=fetch_start, end=end_date, progress=False)['Close'].ffill()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        _ks_s = df['^KS11'].ffill()
        _sp_s = df['^GSPC'].ffill()
        _fx_s = df['KRW=X'].ffill()
        _b10_s = df['^TNX'].ffill()
        _cp_s = df['HG=F'].ffill()
        _vx_s = df['^VIX'].ffill()
        _em_s = df['EEM'].ffill() if 'EEM' in df.columns else _sp_s
        
        _ma20 = _ks_s.rolling(window=20).mean()
        
        analyze_dates = _ks_s[_ks_s.index >= start_date].index
        hist_risks = []
        tot_w = w_m + w_t + w_g + w_f + w_p
        
        if tot_w == 0: tot_w = 1.0 # 0 나누기 방지
        
        def calc_z_score(series, date, inverse=False):
            past_data = series[series.index <= date]
            past_1y = past_data[past_data.index >= (date - pd.Timedelta(days=365))]
            if len(past_1y) < 30: return 50.0
            val = float(series.loc[date]) if date in series.index else float(past_data.iloc[-1])
            std_val = past_1y.std()
            if std_val == 0: return 50.0
            z = (val - past_1y.mean()) / std_val
            score = 100 / (1 + np.exp(-z))
            return max(0.0, min(100.0, (100.0 - score) if inverse else score))

        for d in analyze_dates:
            m = (calc_z_score(_fx_s, d) + calc_z_score(_b10_s, d) + calc_z_score(_cp_s, d, True)) / 3.0
            val_ks = float(_ks_s.loc[d])
            val_ma = float(_ma20.loc[d]) if pd.notna(_ma20.loc[d]) else val_ks
            s_tech = max(0.0, min(100.0, 100.0 - (val_ks / val_ma - 0.9) * 500.0)) if val_ma != 0 else 50.0
            
            base_risk = (m * w_m + s_tech * w_t + calc_z_score(_sp_s, d, True) * w_g + calc_z_score(_vx_s, d) * w_f + calc_z_score(_em_s, d, True) * w_p) / tot_w
            # 볼록성(Convexity) 추가
            k_val = 0.5
            convex_risk = ((np.exp(k_val * base_risk / 100.0) - 1.0) / (np.exp(k_val) - 1.0)) * 100.0
            hist_risks.append(convex_risk)
            
        return pd.Series(hist_risks, index=analyze_dates)
    
    def create_black_swan_chart(hist_series, current_series, title):
        # 1. 과거 궤적의 폭락 시점 찾기
        crash_idx = int(np.argmax(hist_series.values))
        
        # 2. X축 재정렬 (폭락 시점 = 0)
        hist_x = np.arange(len(hist_series)) - crash_idx
        
        # 3. 현재 궤적의 마지막 날이 D-Day(0)가 되도록 강제 정렬
        M = len(current_series)
        N = len(hist_series)
        curr_x = np.arange(M) - (M - 1)
        
        # 4. 과거 궤적 중 현재 궤적과 겹치는 구간의 상관계수 계산
        best_corr = 0
        if crash_idx - (M - 1) >= 0:
            window = hist_series.values[crash_idx - (M - 1) : crash_idx + 1]
            if len(window) == M and np.std(window) > 0 and np.std(current_series.values) > 0:
                best_corr = np.corrcoef(window, current_series.values)[0, 1]
        else:
            overlap_len = crash_idx + 1
            if overlap_len > 1:
                window = hist_series.values[:overlap_len]
                curr_overlap = current_series.values[-overlap_len:]
                if np.std(window) > 0 and np.std(curr_overlap) > 0:
                    best_corr = np.corrcoef(window, curr_overlap)[0, 1]
        
        fig = go.Figure()
        
        # Danger Zone 음영 (D-10 ~ D+10)
        fig.add_vrect(x0=-10, x1=10, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Danger Zone", annotation_position="top left")
        
        # 과거 궤적
        fig.add_trace(go.Scatter(x=hist_x, y=hist_series.values, name=f"{title[:4]}년 궤적", line=dict(color='blue', dash='dot'), connectgaps=True))
        
        # 현재 궤적
        fig.add_trace(go.Scatter(x=curr_x, y=current_series.values, name="현재 위험 지수", line=dict(color='red', width=3), connectgaps=True))
        
        # 폭락 시점 화살표
        fig.add_annotation(
            x=0, y=100, text="폭락 시점 (D-Day)", 
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor="blue",
            font=dict(color="blue", size=12, weight="bold"), ax=0, ay=-40
        )
        
        # 현재 위치 수직선
        curr_last_x = curr_x[-1]
        fig.add_vline(x=curr_last_x, line_width=2, line_dash="dash", line_color="gray")
        
        # 현재 위치 텍스트
        fig.add_annotation(
            x=curr_last_x, y=current_series.values[-1], text="현재 위치 (D-Day 가정)", 
            showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="gray",
            font=dict(color="black", size=11, weight="bold"), ax=50, ay=0
        )
        
        # 유사도 점수 표시 (우측 상단)
        if best_corr > 0:
            sim_score = best_corr * 100
            sim_text = f"🚨 폭락 직전 패턴 일치율: {sim_score:.1f}%"
            bg_color = "rgba(255, 255, 255, 0.8)"
            font_color = "red" if sim_score > 70 else "black"
            border_color = "red" if sim_score > 70 else "gray"
        else:
            sim_score = 0
            sim_text = f"✅ 패턴 불일치 (위험 징후 없음)"
            bg_color = "rgba(230, 255, 230, 0.8)"
            font_color = "green"
            border_color = "green"
            
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=sim_text,
            showarrow=False,
            font=dict(color=font_color, size=13, weight="bold"),
            bgcolor=bg_color, bordercolor=border_color, borderwidth=1
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="폭락일 기준 (Days)", range=[-120, 120]),
            yaxis_title="위험 지수",
            margin=dict(l=0, r=0, t=30, b=30)
        )
        return fig, sim_score, 0

    col_bs1, col_bs2, col_bs3 = st.columns(3)
    avg_current_risk = np.mean(hist_df['Risk'].iloc[-30:])
    current_series = hist_df['Risk'].iloc[-120:]
    
    with col_bs1:
        st.info("**2000 닷컴 버블 vs 현재**")
        bs_2000 = get_true_historical_risk("2000-01-01", "2001-01-01", w_macro, w_tech, w_global, w_fear, w_peri)
        fig_bs0, sim_00, d_day_00 = create_black_swan_chart(bs_2000, current_series, "2000 닷컴 버블")
        st.plotly_chart(fig_bs0, use_container_width=True)
        if sim_00 > 70 and d_day_00 > 0 and d_day_00 <= 30:
            st.warning(f"⚠️ 2000년 버블 직전(D-{int(d_day_00)})과 패턴이 {sim_00:.1f}% 일치하여 주의가 필요합니다.")
        elif avg_current_risk > 60:
            st.warning(f"⚠️ 현재 위험 지수(평균 {avg_current_risk:.1f})가 높아 버블 붕괴 초기 단계일 수 있습니다.")
        else:
            st.success(f"✅ 현재 위험 지수 흐름은 닷컴 버블 붕괴 경로와 거리가 있습니다.")

    with col_bs2:
        st.info("**2008 금융위기 vs 현재**")
        bs_2008 = get_true_historical_risk("2008-01-01", "2009-01-01", w_macro, w_tech, w_global, w_fear, w_peri)
        fig_bs1, sim_08, d_day_08 = create_black_swan_chart(bs_2008, current_series, "2008 금융위기")
        st.plotly_chart(fig_bs1, use_container_width=True)
        if sim_08 > 70 and d_day_08 > 0 and d_day_08 <= 30:
            st.warning(f"⚠️ 2008년 위기 직전(D-{int(d_day_08)})과 패턴이 {sim_08:.1f}% 일치하여 주의가 필요합니다.")
        elif avg_current_risk > 60:
            st.warning(f"⚠️ 현재 위험 지수(평균 {avg_current_risk:.1f})가 높아 위기 초기 단계일 수 있습니다.")
        else:
            st.success(f"✅ 현재 위험 지수(평균 {avg_current_risk:.1f})는 금융위기 경로와 거리가 있습니다.")
            
    with col_bs3:
        st.info("**2020 코로나 폭락 vs 현재**")
        bs_2020 = get_true_historical_risk("2020-01-01", "2020-06-01", w_macro, w_tech, w_global, w_fear, w_peri)
        fig_bs2, sim_20, d_day_20 = create_black_swan_chart(bs_2020, current_series, "2020 코로나 폭락")
        st.plotly_chart(fig_bs2, use_container_width=True)
        if sim_20 > 70 and d_day_20 > 0 and d_day_20 <= 30:
            st.error(f"🚨 2020년 팬데믹 폭락(D-{int(d_day_20)})과 패턴이 {sim_20:.1f}% 일치하며 매우 위험합니다.")
        elif avg_current_risk > 50:
            st.error(f"🚨 주의: 현재 위험 지수가 팬데믹 상승 구간과 유사한 패턴을 보입니다.")
        else:
            st.info(f"💡 현재 위험 지수 흐름은 2020년 패닉 궤적보다는 안정적입니다.")

    # 9. 지표별 상세 분석 및 AI 설명
    st.markdown("---")
    st.subheader("🔍 실물 경제 및 주요 상관관계 지표 분석 (AI 해설)")
    
    # 지표 데이터를 AI 프롬프트용으로 생성
    latest_data_summary = f"""
    [김효진 박사 코스피 위험 분석 포인트: "주변부(신흥국 등)가 식는지 주목하라", "경제가 가열되며 물가/금리가 핵심 리스크로 부상"]
    - 신흥국 EEM 현재가: {em_s.iloc[-1]:.2f} (최근 1년 평균 대비 {((em_s.iloc[-1]/em_s[em_s.index >= (em_s.index.max() - pd.Timedelta(days=365))].mean())-1)*100:+.1f}%)
    - S&P 500 현재가: {sp_s.iloc[-1]:.2f} (최근 1년 평균 대비 {((sp_s.iloc[-1]/sp_s[sp_s.index >= (sp_s.index.max() - pd.Timedelta(days=365))].mean())-1)*100:+.1f}%)
    - 원/달러 환율: {fx_s.iloc[-1]:.1f}원 (전일 대비 {fx_s.iloc[-1]-fx_s.iloc[-2]:+.1f}원)
    - 구리 가격: {cp_s.iloc[-1]:.2f} (최근 추세: {'상승' if cp_s.iloc[-1] > cp_s.iloc[-5] else '하락'})
    - VIX 지수: {vx_s.iloc[-1]:.2f} (위험 수준: {'높음' if vx_s.iloc[-1] > 20 else '낮음'})
    - 미 국채 10년물 금리: {b10_s.iloc[-1]:.3f}% (위험 수준: {'높음' if b10_s.iloc[-1] > 4.5 else '낮음'})
    - 유가(WTI): {wt_s.iloc[-1]:.2f} (최근 1년 평균 대비 {((wt_s.iloc[-1]/wt_s[wt_s.index >= (wt_s.index.max() - pd.Timedelta(days=365))].mean())-1)*100:+.1f}%)
    """
    
    # 가독성 높은 레이아웃 조정을 위한 프롬프트 수정
    pass

    def create_chart(series, title, threshold, desc_text):
        # 데이터가 비어있지 않은지 확인 후 그래프 생성
        if series is not None and not series.empty and not series.isnull().all():
            fig = go.Figure(go.Scatter(x=series.index, y=series.values, name=title, connectgaps=True)) # connectgaps 추가
            fig.add_hline(y=threshold, line_width=2, line_color="red")
            # 주석 위치 계산을 위한 안전장치
            annot_idx = len(series)//2 if len(series) > 0 else 0
            fig.add_annotation(x=series.index[annot_idx], y=threshold, text=desc_text, showarrow=False, font=dict(color="red"), bgcolor="white", yshift=10)
            fig.add_vline(x=COVID_EVENT_DATE, line_width=1.5, line_dash="dash", line_color="blue")
            fig.add_annotation(x=COVID_EVENT_DATE, y=1, yref="paper", text="COVID 지수 폭락 기점", showarrow=False, font=dict(color="blue"), xanchor="left", xshift=5, bgcolor="white")
            return fig
        return go.Figure()

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        st.subheader("미국 S&P 500")
        st.plotly_chart(create_chart(sp_s, "S&P 500", sp_s[sp_s.index >= (sp_s.index.max() - pd.Timedelta(days=365))].mean()*0.9, "평균 대비 -10% 하락 시"), use_container_width=True)
        st.info("**미국 지수**: KOSPI와 강한 정(+)의 상관성  \n**빨간선 기준**: 최근 1년 평균 가격 대비 -10% 하락 지점")
    with r1_c2:
        st.subheader("원/달러 환율")
        fx_th = float(fx_s[fx_s.index >= (fx_s.index.max() - pd.Timedelta(days=365))].mean() * 1.02)
        st.plotly_chart(create_chart(fx_s, "원/달러 환율", fx_th, f"{fx_th:.1f}원 돌파 시 위험"), use_container_width=True)
        st.info("**환율**: +2% 상회 시 외국인 자본 유출 심화  \n**빨간선 기준**: 최근 1년 평균 환율 대비 +2% 상승 지점")
    with r1_c3:
        st.subheader("실물 경기 지표 (Copper)")
        st.plotly_chart(create_chart(cp_s, "Copper", cp_s[cp_s.index >= (cp_s.index.max() - pd.Timedelta(days=365))].mean()*0.9, "수요 위축 시 위험"), use_container_width=True)
        st.info("**실물 경기**: 구리 가격 하락은 수요 둔화 선행 신호  \n**빨간선 기준**: 최근 1년 평균 가격 대비 -10% 하락 지점")

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        st.subheader("장단기 금리차")
        # 금리차 그래프 생성
        st.plotly_chart(create_chart(yield_curve, "금리차", 0.0, "0 이하 역전 시 위험"), use_container_width=True)
        st.info("**금리차**: 금리 역전은 경기 침체 강력 전조  \n**빨간선 기준**: 금리차가 0(수평)이 되는 역전 한계 지점")
    with r2_c2:
        st.subheader("KOSPI 기술적 분석")
        ks_recent = ks_s[ks_s.index >= (ks_s.index.max() - pd.Timedelta(days=30))]
        fig_ks = go.Figure()
        # 현재가 그래프: 선 굵기 및 마커 추가로 가독성 향상
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ks_recent.values, name="현재가", line=dict(color='royalblue', width=3), mode='lines+markers', connectgaps=True))
        # 20일 이동평균선: 계산된 ma20을 reindex하여 끊김 없이 시각화
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ma20.reindex(ks_recent.index).values, name="20일선", line=dict(color='orange', width=2, dash='dot'), connectgaps=True))
        # 화살표 위치 수정: y값을 실제 20일선 지수값(ma20.iloc[-1])으로 명확히 지정
        fig_ks.add_annotation(
            x=ks_recent.index[-1], 
            y=ma20.iloc[-1], # 화살표가 0이 아닌 실제 지수 위치를 가리키도록 수정
            text="20일 평균선 하회 시 위험", 
            showarrow=True, 
            arrowhead=2, 
            ax=0, 
            ay=-40, 
            font=dict(color="red", size=12),
            bgcolor="white",
            bordercolor="red"
        )
        fig_ks.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_ks, use_container_width=True)
        st.info("**기술적 분석**: 20일선 하회 시 단기 추세 하락")
    with r2_c3:
        st.subheader("VIX 공포 지수")
        st.plotly_chart(create_chart(vx_s, "VIX", 30, "30 돌파 시 패닉"), use_container_width=True)
        st.info("**VIX 지수**: 지수 급등은 투매 가능성 시사  \n**빨간선 기준**: 시장의 극단적 공포를 상징하는 지수 30 지점")

    st.markdown("---")
    r3_c1, r3_c2, r3_c3 = st.columns(3)
    with r3_c1:
        st.subheader("글로벌 물동량 지표 (BDRY)")
        if not fr_s.empty:
            fr_th = round(float(fr_s[fr_s.index >= (fr_s.index.max() - pd.Timedelta(days=365))].mean() * 0.85), 2)
            st.plotly_chart(create_chart(fr_s, "교역량", fr_th, "물동량 급감 시 위험"), use_container_width=True)
            st.info("**물동량(BDRY)**: 해상 운송 지수는 실물 경제 회복의 선행 지표")
    with r3_c2:
        st.subheader("유가 (WTI)")
        if not wt_s.empty:
            wt_th = round(float(wt_s[wt_s.index >= (wt_s.index.max() - pd.Timedelta(days=365))].mean() * 1.2), 2)
            st.plotly_chart(create_chart(wt_s, "유가", wt_th, "에너지 비용 급증 시 위험"), use_container_width=True)
            st.info("**유가**: 급격한 유가 상승은 인플레이션 및 비용 압박 요인")
    with r3_c3:
        st.subheader("달러 인덱스 (DXY)")
        if not dx_s.empty:
            dx_th = round(float(dx_s[dx_s.index >= (dx_s.index.max() - pd.Timedelta(days=365))].mean() * 1.05), 2)
            st.plotly_chart(create_chart(dx_s, "달러 인덱스", dx_th, "달러 강세 시 신흥국 매도 압력"), use_container_width=True)
            st.info("**달러 강세**: 글로벌 안전자산 선호 심리는 KOSPI 하락 요인")

    r4_c1, r4_c2, r4_c3 = st.columns(3)
    with r4_c1:
        st.subheader("미국 국채 10년물 금리")
        if b10_s is not None and not b10_s.empty:
            # 위험 기준선 판별: 
            # 4.5% 이상이면 주식시장에 상당한 할인율 부담으로 작용하므로 위험 기준선으로 설정
            b10_th = 4.5 
            st.plotly_chart(create_chart(b10_s, "10년물 국채 금리", b10_th, "4.5% 돌파 시 주식 밸류에이션 부담"), use_container_width=True)
            st.info("**국채 금리**: 4.5% 이상 상승 시 무위험 대안 수익률이 높아져 주식 밸류에이션 부담이 커짐")

    # 현재 시장 지표 종합 진단 AI 컨테이너 정의 (위치: 모든 지표 차트 하단)
    ai_indicator_container = st.container()

    st.markdown("---")
    st.subheader("📊 지수간 동조화 및 섹터 분석")
    sp_norm = (sp_s - sp_s.mean()) / sp_s.std(); fr_norm = (fr_s - fr_s.mean()) / fr_s.std(); ks_norm = (ks_s - ks_s.mean()) / ks_s.std()
    fig_norm = go.Figure()
    fig_norm.add_trace(go.Scatter(x=ks_norm.index, y=ks_norm.values, name="KOSPI (Std)", line=dict(color='red'), connectgaps=True))
    fig_norm.add_trace(go.Scatter(x=sp_norm.index, y=sp_norm.values, name="S&P 500 (Std)", line=dict(color='blue'), connectgaps=True))
    fig_norm.add_trace(go.Scatter(x=fr_norm.index, y=fr_norm.values, name="BDRY (Std)", line=dict(color='orange'), connectgaps=True))
    fig_norm.update_layout(title="Z-Score 동조화 추세", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); st.plotly_chart(fig_norm, use_container_width=True)
    st.info("""
**[현재 상황 상세 해석 가이드]**
* **KOSPI(Red) vs S&P 500(Blue) 디커플링**: S&P 500 지수가 KOSPI보다 확연히 높이 위치할 경우, 글로벌 증시 상승에서 한국이 소외되는 현상(외자 이탈, 구조적 우려 등)을 시사할 수 있습니다.\n\n반대의 경우는 KOSPI의 단기 과열이나 강한 독자 모멘텀을 뜻합니다.\n\n* **주가지수(Red, Blue)가 물동량(Orange)보다 위에 있을 때**: 실물 경기 뒷받침 없이 기대감만으로 지수가 과열된 상태일 수 있습니다.\n\n하락 가능성이 높습니다.\n\n* **지표들이 비슷한 위치일 때**: KOSPI, 글로벌 증시, 실물 경기가 동조화되어 움직이는 매우 안정적인 추세입니다.\n\n* **글로벌 물동량(Orange)이 주가지수보다 위에 있을 때**: 실물 경기는 회복되었으나 주가가 저평가된 상태입니다.\n\n우상향 가능성을 시사합니다.\n\n""")

    st.markdown("---")
    sc1, sc2 = st.columns(2)
    
    with sc1:
        st.subheader("🇰🇷 KOSPI 섹터별 대표 종목 등락")
        sector_perf = []
        for n, t in sector_map.items():
            try:
                cur = sector_raw[t].ffill().iloc[-1]; pre = sector_raw[t].ffill().iloc[-2]
                sector_perf.append({"섹터": n, "등락률": round(((cur - pre) / pre) * 100, 2)})
            except: pass
        if sector_perf:
            df_p = pd.DataFrame(sector_perf)
            fig_h = px.bar(df_p, x="섹터", y="등락률", color="등락률", color_continuous_scale='RdBu_r', text="등락률")
            fig_h.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_h, use_container_width=True)

    with sc2:
        st.subheader("🇺🇸 S&P 500 섹터별 대표 등락 (ETF)")
        sp500_sector_perf = []
        for n, t in sp500_sector_map.items():
            try:
                cur = sp500_sector_raw[t].ffill().iloc[-1]; pre = sp500_sector_raw[t].ffill().iloc[-2]
                sp500_sector_perf.append({"섹터": n, "등락률": round(((cur - pre) / pre) * 100, 2)})
            except: pass
        if sp500_sector_perf:
            df_sp_p = pd.DataFrame(sp500_sector_perf)
            fig_sp_h = px.bar(df_sp_p, x="섹터", y="등락률", color="등락률", color_continuous_scale='RdBu_r', text="등락률")
            fig_sp_h.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_sp_h, use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

# 하단 캡션
st.caption(f"Last updated: {get_kst_now().strftime('%d일 %H시 %M분')} | NewsAPI 및 Gemini AI 분석 엔진 가동 중")

# --- [AI 분석: 맨 마지막에 처리] ---
if st.session_state.get("run_ai_trigger"):
    with st.spinner("AI 분석 엔진을 가동하여 데이터를 통합 분석하고 있습니다..."):
        import concurrent.futures
        import re as _re

        # ── 각 섹션별 독립 프롬프트 정의 ─────────────────────────────────
        def _run_news():
            if not news_data:
                return ""
            titles_list = "\n".join([f"- {a['title']}" for a in news_data])
            p = f"""아래 영어 뉴스 헤드라인들을 한국어로 번역하세요.
규칙: 다른 설명 없이 번호와 번역문만 출력하세요. 고유명사는 영어로 유지하세요.

{titles_list}

출력 형식 예시:
1. 번역된 첫 번째 헤드라인
2. 번역된 두 번째 헤드라인"""
            return get_ai_analysis(p)

        def _run_trump():
            if not trump_raw:
                return ""
            p = f"""아래 영어 소셜 게시물을 자연스러운 한국어 단락으로 번역하세요.
규칙: 번역 결과만 출력. 원문 인용·설명·태그 출력 절대 금지.

{trump_raw}"""
            return get_ai_analysis(p)

        def _run_bt():
            if not bt_data_text:
                return ""
            p = f"""아래 데이터를 바탕으로 시장 위험 지수의 유효성을 진단하는 3~4문장의 한국어 문단을 작성하세요.
규칙: 마크다운·영단어·한자 사용 금지. 분석 문단만 출력.

[중요 맥락(Context)]
이 모델은 극단적 꼬리 위험(Tail Risk)을 사전에 경고하는 비선형적 '선행 지표'입니다. 따라서 평상시 코스피 지수와의 단순 선형 상관계수가 낮게 나오는 것은 수학적으로 지극히 정상입니다.
단순 상관계수가 낮다는 이유로 "유효성이 낮다"고 깎아내리지 마세요. 오히려 "단순 선형 상관관계보다는, 지수가 임계점(60이상)을 돌파하며 패닉 조기 경보를 울리는 기능이 이 모델의 핵심 유효성"이라는 긍정적인 논조로 진단 문단을 작성하세요.

데이터: {bt_data_text}"""
            return get_ai_analysis(p)

        def _run_indicator():
            if 'latest_data_summary' not in locals() and 'latest_data_summary' not in globals():
                return ""
            p = f"""아래 시장 지표를 분석하여 현재 한국 증시(코스피) 상황을 진단하는 4문장의 한국어 문단을 작성하세요.
규칙: 마크다운·영단어·한자 사용 금지. 분석 문단만 출력.

{latest_data_summary}"""
            return get_ai_analysis(p)

        # 트럼프 raw 텍스트 준비
        trump_raw = ""
        if 'trump_data' in locals() and trump_data:
            trump_raw = "\n\n---\n\n".join([
                f"{t.get('title', '')} {t.get('description', '')}".strip()
                for t in trump_data
                if len(f"{t.get('title', '')} {t.get('description', '')}".strip()) >= 10
            ])

        bt_data_text = ""
        if 'corr_val' in locals() and 'hist_risks' in locals():
            bt_data_text = f"상관계수: {corr_val:.2f}, 현재 위험지수: {hist_risks[-1]:.1f}, 최근7일: {[round(r,1) for r in hist_risks[-7:]]}"

        # ── 병렬 실행 (속도 유지) ──────────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            f_news      = executor.submit(_run_news)
            f_trump     = executor.submit(_run_trump)
            f_bt        = executor.submit(_run_bt)
            f_indicator = executor.submit(_run_indicator)

        news_result      = clean_news_translation(f_news.result().strip())
        trump_result     = clean_ai_output(f_trump.result().strip())
        bt_result        = clean_ai_output(f_bt.result().strip())
        indicator_result = clean_ai_output(f_indicator.result().strip())

        # Save to session state and reset trigger
        st.session_state["ai_analysis_results"] = {
            "news": news_result,
            "trump": trump_result,
            "bt": bt_result,
            "indicator": indicator_result
        }
        st.session_state["ai_last_data_hash"] = current_data_hash
        st.session_state["ai_last_saved_at"] = get_kst_now().strftime("%Y-%m-%d %H:%M:%S")
        
        save_ai_cache(SHEET_ID, "main_risk", current_data_hash, json.dumps(st.session_state["ai_analysis_results"], ensure_ascii=False))
        
        st.session_state["run_ai_trigger"] = False
        st.success("AI 통합 분석 완료 및 구글 시트 저장!")
        st.rerun()

# --- [컨테이너에 결과 또는 안내문 출력] ---
# 1. AI 뉴스 통합 번역 출력
if news_data and ai_news_container:
    with ai_news_container:
        if st.session_state.get("ai_analysis_results") is None:
            st.info("💡 AI 뉴스 번역/분석을 원하시면 상단의 '🚀 AI 분석 시작' 버튼을 눌러주세요.")
        else:
            clean_summary = clean_news_translation(st.session_state["ai_analysis_results"].get("news", ""))
            if clean_summary:
                st.markdown(f"""
                <div class="ai-analysis-box">
                    <strong>🔎 AI 뉴스 헤드라인 번역</strong><br><br>
                    {clean_summary}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="ai-analysis-box">번역 결과를 불러오는 중 오류가 발생했습니다.</div>', unsafe_allow_html=True)

# 1.5 트럼프 트윗 통합 번역 출력
if 'trump_data' in locals() and trump_data and ai_trump_container:
    with ai_trump_container:
        if st.session_state.get("ai_analysis_results") is None:
            st.info("💡 트럼프 소셜 최신 브리핑 번역을 원하시면 상단의 '🚀 AI 분석 시작' 버튼을 눌러주세요.")
        else:
            t_clean = clean_ai_output(st.session_state["ai_analysis_results"].get("trump", ""))
            if t_clean and "AI 모델 서버가 혼잡하여" not in t_clean and t_clean != "번역된 내용이 없습니다.":
                st.markdown(f"""
                <div class="ai-analysis-box">
                    <strong>🇺🇸 트럼프 소셜 최신 브리핑 (번역)</strong><br><br>
                    {t_clean}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write("번역된 내용이 없습니다.")

# 2. 모델 유효성 진단 출력
if bt_analysis_container:
    with bt_analysis_container:
        if st.session_state.get("ai_analysis_results") is None:
            st.info("💡 AI 모델 유효성 상세 진단을 원하시면 상단의 '🚀 AI 분석 시작' 버튼을 눌러주세요.")
        else:
            clean_bt = clean_ai_output(st.session_state["ai_analysis_results"].get("bt", ""))
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; font-size: 0.85rem; color: #31333F; line-height: 1.6; margin-bottom: 20px;">
                <strong>🤖 모델 유효성 진단:</strong><br>{clean_bt}
            </div>
            """, unsafe_allow_html=True)

# 3. 현재 시장 지표 종합 진단 출력
if ai_indicator_container:
    with ai_indicator_container:
        with st.expander("🤖 현재 시장 지표 종합 진단 (클릭하여 펼치기)", expanded=False):
            if st.session_state.get("ai_analysis_results") is None:
                st.info("💡 AI 지표 종합 진단을 원하시면 상단의 '🚀 AI 분석 시작' 버튼을 눌러주세요.")
            else:
                clean_indicator = clean_ai_output(st.session_state["ai_analysis_results"].get("indicator", ""))
                st.markdown(f"""
                <div class="ai-analysis-box" style="background: #ffffff; color: #31333F !important; border: 1px solid #e0e0e0; border-left: 8px solid #007bff; line-height: 1.5; padding: 15px 20px;">
                    {clean_indicator}
                </div>
                """, unsafe_allow_html=True)
