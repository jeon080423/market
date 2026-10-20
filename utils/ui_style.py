"""
메뉴 공통 디자인 기준 (2026-10-20 통일)
- 모든 메뉴(메인 위험 모니터링 / 종목 탐색 / 과열 시그널 / 종목 가치)에 동일한
  폰트 크기·색상을 적용하기 위한 단일 기준 모듈
"""
import streamlit as st

INK = "#2c3e50"     # 제목·본문 기본 색상
SUB = "#6c757d"     # 보조 텍스트 색상
ACCENT = "#1f4e79"  # 강조 색상

COMMON_CSS = """
<style>
/* ===== 메뉴 공통 타이포그래피·컬러 기준 ===== */
h1 { font-size: 1.75rem !important; font-weight: 700 !important; color: #2c3e50; line-height: 1.3 !important; }
h2 { font-size: 1.35rem !important; font-weight: 700 !important; color: #2c3e50; }
h3 { font-size: 1.12rem !important; font-weight: 600 !important; color: #2c3e50; }
h5 { font-size: 1.0rem !important; font-weight: 700 !important; color: #2c3e50; }
.stMarkdown { color: #2c3e50; }
.stMarkdown p, .stMarkdown li { font-size: 0.95rem; line-height: 1.65; }
div[data-testid="stCaptionContainer"] p { font-size: 0.8rem !important; color: #6c757d !important; }
/* 공통 지표 카드 */
.metric-card { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
.metric-title { font-size: 0.85rem; color: #6c757d; margin-bottom: 8px; font-weight: 600; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #2c3e50; }
.metric-sub { font-size: 0.8rem; color: #495057; margin-top: 5px; }
</style>
"""


def apply_common_style():
    """각 메뉴 렌더링 시작 시 호출해 공통 스타일을 주입"""
    st.markdown(COMMON_CSS, unsafe_allow_html=True)


def style_fig(fig, title_size=15):
    """Plotly 차트 공통 폰트(종류·크기·색상) 적용"""
    fig.update_layout(
        font=dict(family="NanumGothic, 'Malgun Gothic', sans-serif", size=12, color=INK),
        title_font=dict(size=title_size, color=INK),
    )
    return fig


import re as _re

_EMOJI_RE = _re.compile(
    r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
    r"\uFE0F\u20E3\u203C\u2049\u3297\u3299]"
)


def strip_emojis(text):
    """동적 텍스트(외부 API·캐시 등)에서 이모지 제거"""
    if not isinstance(text, str):
        return text
    return _EMOJI_RE.sub("", text)


def strip_df_emojis(df):
    """DataFrame의 모든 문자열 셀에서 이모지 제거 (표시 직전 적용)"""
    if df is None or getattr(df, "empty", True):
        return df
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: strip_emojis(v) if isinstance(v, str) else v)
    return df
