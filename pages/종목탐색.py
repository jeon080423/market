import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import re
import html
import concurrent.futures

# Header for crawler requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ─────────────────────────────────────────────────────────────
# 네이버 금융 개편(2026) 대응: 구형 sise HTML 크롤링 → 신형 API로 전면 교체
#  - 종목 랭킹(인기검색/거래량/급등/외국인·기관 순매수): stock.naver.com 공식 API
#  - 애널리스트 리포트: m.stock.naver.com 리서치 API
# ─────────────────────────────────────────────────────────────
STOCK_RANK_API = "https://stock.naver.com/api/domestic/market/stock/default"
RESEARCH_API = "https://m.stock.naver.com/api/research/company"
STOCK_API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Referer': 'https://stock.naver.com/',
}
MOBILE_API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Referer': 'https://m.stock.naver.com/',
}


def _fmt_num(v) -> str:
    """'1234567' / '-9,000' 같은 값을 천단위 콤마 문자열로 변환"""
    try:
        return f"{int(float(str(v).replace(',', ''))):,}"
    except Exception:
        return str(v) if v is not None else ""


def _change_str(item: dict) -> str:
    """신형 API 종목 아이템 → '▲ 1,000 (+2.35%)' 형식 문자열"""
    try:
        chg = float(str(item.get('prevChangePrice', '0')).replace(',', ''))
        rate = float(str(item.get('prevChangeRate', '0')).replace(',', ''))
    except Exception:
        return ""
    if chg > 0:
        direction = "▲"
    elif chg < 0:
        direction = "▼"
    else:
        direction = ""
    return f"{direction} {abs(int(chg)):,} ({rate:+.2f}%)".strip()


def _change_dir(item: dict) -> str:
    try:
        chg = float(str(item.get('prevChangePrice', '0')).replace(',', ''))
    except Exception:
        return ""
    if chg > 0:
        return "▲"
    if chg < 0:
        return "▼"
    return ""


def _fetch_stock_rank(order_type: str, market_type: str = None, count: int = 30) -> list:
    """stock.naver.com 랭킹 API 호출 (orderType: searchTop/quantTop/up/foreignPureBuy/organizationPureBuy 등)"""
    params = {'orderType': order_type, 'startIdx': 0, 'pageSize': count}
    if market_type:
        params['marketType'] = market_type
    res = requests.get(STOCK_RANK_API, params=params, headers=STOCK_API_HEADERS, timeout=10)
    res.raise_for_status()
    data = res.json()
    return data if isinstance(data, list) else []


def _rank_to_df(items: list, with_volume: bool = False) -> pd.DataFrame:
    stocks = []
    for i, item in enumerate(items, 1):
        row = {
            '순위': str(i),
            '종목명': item.get('itemname', ''),
            '티커': item.get('itemcode', ''),
            '현재가': _fmt_num(item.get('nowPrice')),
            '등락률': _change_str(item),
        }
        if with_volume:
            row['거래량'] = _fmt_num(item.get('tradeVolume'))
        stocks.append(row)
    return pd.DataFrame(stocks)


@st.cache_data(ttl=60)  # 60초 캐싱으로 실시간성을 극대화하되 단기간 반복 로딩 부담 제어
def crawl_naver_popular_stocks() -> pd.DataFrame:
    """
    네이버 증권 실시간 인기 검색 종목 (상위 30) - 신형 API(searchTop)
    - 반환: DataFrame [순위, 종목명, 티커, 현재가, 등락률]
    """
    try:
        items = _fetch_stock_rank('searchTop', count=30)
        if not items:
            return pd.DataFrame()
        return _rank_to_df(items)
    except Exception as e:
        st.warning(f"인기 검색 종목 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def crawl_naver_volume_top(sosok: int) -> pd.DataFrame:
    """
    네이버 증권 실시간 거래량 상위 종목 - 신형 API(quantTop)
    - sosok: 0 = KOSPI, 1 = KOSDAQ
    """
    market = 'KOSPI' if sosok == 0 else 'KOSDAQ'
    try:
        items = _fetch_stock_rank('quantTop', market_type=market, count=30)
        if not items:
            return pd.DataFrame()
        return _rank_to_df(items, with_volume=True)
    except Exception as e:
        st.warning(f"거래량 상위 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def crawl_naver_price_surge(sosok: int) -> pd.DataFrame:
    """
    네이버 증권 실시간 주가 급등(상승률 상위) 종목 - 신형 API(up)
    - sosok: 0 = KOSPI, 1 = KOSDAQ
    """
    market = 'KOSPI' if sosok == 0 else 'KOSDAQ'
    try:
        items = _fetch_stock_rank('up', market_type=market, count=30)
        if not items:
            return pd.DataFrame()
        return _rank_to_df(items, with_volume=True)
    except Exception as e:
        st.warning(f"주가 급등 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


def _investor_top_df(order_type: str) -> pd.DataFrame:
    items = _fetch_stock_rank(order_type, market_type='KOSPI', count=30)
    stocks = []
    for i, item in enumerate(items, 1):
        stocks.append({
            '순위': str(i),
            '종목명': item.get('itemname', ''),
            '티커': item.get('itemcode', ''),
            '현재가': _fmt_num(item.get('nowPrice')),
            '등락': _change_dir(item),
        })
    return pd.DataFrame(stocks)


@st.cache_data(ttl=60)
def crawl_naver_foreigner_top() -> pd.DataFrame:
    """
    네이버 증권 외국인 순매수 상위 종목 - 신형 API(foreignPureBuy)
    - 반환: DataFrame [순위, 종목명, 티커, 현재가, 등락]
    """
    try:
        return _investor_top_df('foreignPureBuy')
    except Exception as e:
        st.warning(f"외국인 순매수 상위 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def crawl_naver_institution_top() -> pd.DataFrame:
    """
    네이버 증권 기관 순매수 상위 종목 - 신형 API(organizationPureBuy)
    - 반환: DataFrame [순위, 종목명, 티커, 현재가, 등락]
    """
    try:
        return _investor_top_df('organizationPureBuy')
    except Exception as e:
        st.warning(f"기관 순매수 상위 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # 리포트는 업데이트 주기가 길어 5분 캐싱
def crawl_naver_analyst_reports() -> pd.DataFrame:
    """
    네이버 증권 리서치 종목분석 리포트 (언급 빈도 순 정렬) - 신형 리서치 API
    - 반환: DataFrame [순위, 종목명, 티커, 언급 빈도, 최근 리포트 제목, 핵심 키워드, 발행 증권사, 최근 발행일, 리포트 링크]
    """
    reports = []
    # 2개 페이지(40개 리포트) 수집하여 빈도 집계의 정확성 향상
    for page in [1, 2]:
        try:
            res = requests.get(RESEARCH_API, params={'page': page}, headers=MOBILE_API_HEADERS, timeout=10)
            res.raise_for_status()
            for item in res.json():
                reports.append({
                    'stock_name': item.get('itemName', ''),
                    'ticker': item.get('itemCode', ''),
                    'title': item.get('title', ''),
                    'report_url': item.get('endUrl', ''),
                    'pdf_url': '',
                    'brokerage': item.get('brokerName', ''),
                    'date': item.get('writeDate', ''),
                    'research_id': item.get('researchId'),
                })
        except Exception as e:
            st.warning(f"애널리스트 리포트 크롤링(페이지 {page}) 중 오류가 발생했습니다: {e}")

    if not reports:
        return pd.DataFrame()

    # 종목별 그룹화 및 빈도 집계
    from collections import defaultdict
    stock_groups = defaultdict(list)
    for rep in reports:
        stock_groups[rep['stock_name']].append(rep)

    freq_list = []
    for stock_name, reps in stock_groups.items():
        # reps[0]은 수집 순서상 가장 최신 리포트
        latest_rep = reps[0]
        freq_list.append({
            'stock_name': stock_name,
            'ticker': latest_rep['ticker'],
            'count': len(reps),
            'latest_title': latest_rep['title'],
            'latest_report_url': latest_rep['report_url'],
            'latest_pdf_url': latest_rep['pdf_url'],
            'latest_brokerage': latest_rep['brokerage'],
            'latest_date': latest_rep['date'],
            'research_id': latest_rep.get('research_id'),
        })

    # 빈도(count) 내림차순, 동일 빈도 시 최신 발행일(latest_date) 내림차순 정렬
    freq_list.sort(key=lambda x: (x['count'], x['latest_date']), reverse=True)
    top_items = freq_list[:10]

    # 상위 10개 종목의 리포트 상세를 병렬 조회하여 투자의견/목표가/요약 추출
    def _fetch_report_detail(item: dict) -> dict:
        """Thread worker: 리서치 상세 API에서 투자의견, 목표가, 핵심 요약 추출"""
        result = {'opinion': '', 'target_price': '', 'summary': ''}
        rid = item.get('research_id')
        if not rid:
            return result
        try:
            r = requests.get(f"{RESEARCH_API}/{rid}", headers=MOBILE_API_HEADERS, timeout=8)
            r.raise_for_status()
            rc = r.json().get('researchContent', {})
            if rc.get('opinion'):
                result['opinion'] = str(rc['opinion']).strip()
            if rc.get('goalPrice'):
                result['target_price'] = f"목표가 {_fmt_num(rc['goalPrice'])}"
            content = rc.get('content', '')
            if content:
                text_only = re.sub(r'<[^>]+>', ' ', content)
                text_only = html.unescape(text_only)
                text_only = re.sub(r'\s+', ' ', text_only).strip()
                if len(text_only) > 20:
                    result['summary'] = text_only[:160]
        except Exception:
            pass
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        detail_futures = {executor.submit(_fetch_report_detail, item): i for i, item in enumerate(top_items)}
        detail_results = [None] * len(top_items)
        for future in concurrent.futures.as_completed(detail_futures):
            idx = detail_futures[future]
            try:
                detail_results[idx] = future.result()
            except Exception:
                detail_results[idx] = {'opinion': '', 'target_price': '', 'summary': ''}

    # 상위 10개 종목 구성
    sorted_reports = []
    rank_counter = 1
    for i, item in enumerate(top_items):
        detail = detail_results[i] or {'opinion': '', 'target_price': '', 'summary': ''}
        opinion_str = detail.get('opinion', '')
        tp_str = detail.get('target_price', '')
        summary_str = detail.get('summary', '')
        keyword_parts = []
        if opinion_str:
            keyword_parts.append(f"투자의견: {opinion_str}")
        if tp_str:
            keyword_parts.append(tp_str)
        if summary_str:
            keyword_parts.append(summary_str)
        keywords = " | ".join(keyword_parts) if keyword_parts else "-"

        link_url = item['latest_report_url'] or item['latest_pdf_url'] or ""
        sorted_reports.append({
            '순위': str(rank_counter),
            '종목명': f"{item['stock_name']} ({item['ticker']})" if item['ticker'] else item['stock_name'],
            '티커': item['ticker'],
            '언급 빈도': f"{item['count']}회",
            '최근 리포트 제목': item['latest_title'],
            '핵심 키워드': keywords,
            '발행 증권사': item['latest_brokerage'],
            '최근 발행일': item['latest_date'],
            '리포트 링크': link_url
        })
        rank_counter += 1

    return pd.DataFrame(sorted_reports)

def render_naver_sise_table(df: pd.DataFrame):
    """
    KOSPI / KOSDAQ 공통 데이터 프레임 렌더링 테이블
    """
    if df.empty:
        st.info("📊 실시간 데이터를 불러올 수 없습니다.")
        return

    df_display = df.head(10).copy()
    
    # 순위 텍스트 이모지화
    def make_rank_label(rank):
        r = str(rank).strip()
        if r == "1": return "🥇 1"
        elif r == "2": return "🥈 2"
        elif r == "3": return "🥉 3"
        return r
        
    df_display["순위"] = df_display["순위"].apply(make_rank_label)
    df_display = df_display.reset_index(drop=True)
    
    # 1, 2, 3위 행 강조 스타일링
    def row_style(row):
        rank_val = row["순위"]
        if "🥇" in str(rank_val):
            return ["background-color: rgba(255, 215, 0, 0.15); font-weight: bold;"] * len(row)
        elif "🥈" in str(rank_val):
            return ["background-color: rgba(192, 192, 192, 0.15); font-weight: bold;"] * len(row)
        elif "🥉" in str(rank_val):
            return ["background-color: rgba(205, 127, 50, 0.15); font-weight: bold;"] * len(row)
        return [""] * len(row)
    
    styled_df = df_display.style.apply(row_style, axis=1)
    
    # 컬럼 설정 구성
    cols_config = {
        "순위": st.column_config.TextColumn("순위", width="small"),
        "종목명": st.column_config.TextColumn("종목명", width="medium"),
        "티커": None,  # 가로 공간 확보를 위해 감춤
    }
    
    if "검색 비율" in df_display.columns:
        cols_config["검색 비율"] = st.column_config.TextColumn("검색 비율", width="small")
    if "현재가" in df_display.columns:
        cols_config["현재가"] = st.column_config.TextColumn("현재가", width="small")
    if "등락률" in df_display.columns:
        cols_config["등락률"] = st.column_config.TextColumn("등락률", width="medium")
    if "등락" in df_display.columns:
        cols_config["등락"] = st.column_config.TextColumn("등락", width="small")
    if "거래량" in df_display.columns:
        cols_config["거래량"] = st.column_config.TextColumn("거래량", width="small")
    if "리포트 제목" in df_display.columns:
        cols_config["리포트 제목"] = st.column_config.TextColumn("리포트 제목", width="large")
    if "증권사" in df_display.columns:
        cols_config["증권사"] = st.column_config.TextColumn("증권사", width="medium")
    if "작성일" in df_display.columns:
        cols_config["작성일"] = st.column_config.TextColumn("작성일", width="small")
    if "최근 리포트 제목" in df_display.columns:
        cols_config["최근 리포트 제목"] = st.column_config.TextColumn("최근 리포트 제목", width="large")
    if "발행 증권사" in df_display.columns:
        cols_config["발행 증권사"] = st.column_config.TextColumn("발행 증권사", width="medium")
    if "최근 발행일" in df_display.columns:
        cols_config["최근 발행일"] = st.column_config.TextColumn("최근 발행일", width="small")
    if "언급 빈도" in df_display.columns:
        cols_config["언급 빈도"] = st.column_config.TextColumn("언급 빈도", width="small")
        
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=388,
        hide_index=True,
        column_config=cols_config
    )


def render_analyst_reports_table(df: pd.DataFrame):
    """
    애널리스트 리포트 전용 렌더링 테이블
    - '리포트 링크' 커럼을 LinkColumn으로 만들어 켜릭 시 최신 리포트를 새 브라우저 탭에서 열도록 구현
    """
    if df.empty:
        st.info("📊 리포트 데이터를 불러올 수 없습니다.")
        return

    df_display = df.head(10).copy()

    # 순위 이모지화
    def make_rank_label(rank):
        r = str(rank).strip()
        if r == "1": return "🥇 1"
        elif r == "2": return "🥈 2"
        elif r == "3": return "🥉 3"
        return r
    df_display["순위"] = df_display["순위"].apply(make_rank_label)
    df_display = df_display.reset_index(drop=True)

    # 1~3위 행 강조 스타일
    def row_style(row):
        rank_val = row["순위"]
        if "🥇" in str(rank_val):
            return ["background-color: rgba(255, 215, 0, 0.15); font-weight: bold;"] * len(row)
        elif "🥈" in str(rank_val):
            return ["background-color: rgba(192, 192, 192, 0.15); font-weight: bold;"] * len(row)
        elif "🥉" in str(rank_val):
            return ["background-color: rgba(205, 127, 50, 0.15); font-weight: bold;"] * len(row)
        return [""] * len(row)
    styled_df = df_display.style.apply(row_style, axis=1)

    # 커럼 설정: '리포트 링크'를 LinkColumn으로 설정하여 켜릭 시 네이버 리포트 페이지가 새 탭에서 열림
    cols_config = {
        "순위": st.column_config.TextColumn("순위", width="small"),
        "티커": None,  # 숨김
        "리포트 링크": st.column_config.LinkColumn(
            "리포트 열기 ↗️",
            display_text="최신 리포트 보기",
            width="medium"
        ),
        "종목명": st.column_config.TextColumn("종목명", width="medium"),
        "언급 빈도": st.column_config.TextColumn("언급 빈도", width="small"),
        "최근 리포트 제목": st.column_config.TextColumn("최근 리포트 제목", width="large"),
        "핵심 키워드": st.column_config.TextColumn("핵심 키워드", width="large"),
        "발행 증권사": st.column_config.TextColumn("발행 증권사", width="medium"),
        "최근 발행일": st.column_config.TextColumn("최근 발행일", width="small"),
    }

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=450,
        hide_index=True,
        column_config=cols_config,
        column_order=["순위", "종목명", "언급 빈도", "최근 리포트 제목", "핵심 키워드", "발행 증권사", "최근 발행일", "리포트 링크"]
    )

def render_youtube_rank_page():
    """
    실시간 수급 및 가격 지표 기반 선행 종목 탐색 엔진 (app.py 호환 진입점)
    """
    st.title("📊 실시간 코스피(KOSPI) 거래량 급증 및 주가 급등 종목 탐색")
    
    st.markdown("""
    이 대시보드는 주식 시장의 실시간 가격, 수급, 거래량 정보를 모니터링합니다.
    작성 속도가 느리고 과거 사실만 기록하는 일반 뉴스보다 훨씬 선행하여 자금의 이동을 알려주는 **'거래량 및 주가 돌파 데이터(Market Trinity)'**를 네이버 금융에서 60초 간격으로 초고속 추적합니다.
    """)
    
    # 아코디언 가이드 최적화
    with st.expander("💡 실시간 거래량 및 가격 선행 지표 탐색기 작동 원리 자세히 보기", expanded=False):
        st.markdown("""
        이 탐색기는 뉴스 미디어의 정보 지연 문제를 완벽히 해결하기 위해 주식 트레이딩의 핵심 요소인 **'개인 관심도(Search)', '외국인 관심도(Foreigner Net Buy)', '유동성(Volume)', '모멘텀(Price)'**을 결합하여 시장의 자금 흐름을 선제적으로 포착합니다.
        
        #### 1. 개인 관심도: 실시간 인기 검색 종목
        * 포털 사이트 내 투자자들의 실시간 개별 종목 검색 볼륨 트렌드를 포착합니다. 개인 투자자들의 투자 심리 쏠림 및 시장의 단기 이슈를 가장 즉각적으로 반영합니다.
        
        #### 2. 외국인 관심도: 외국인 실시간 순매수 상위 종목
        * 당일 외국인 투자자들의 자금이 집중 유입되는 순매수 상위 종목입니다. 외국인 메이저 자금의 매수 트렌드와 장기 관심도를 직접적으로 나타냅니다.
        
        #### 3. 기관 관심도: 기관 실시간 순매수 상위 종목
        * 당일 기관 투자자들의 순매수 상위 종목입니다. **외국인 + 기관이 동시에 순매수하는 '더블 수급'** 종목은 특히 높은 단기 상승력을 보여줍니다.
        
        #### 4. 유동성 (선행 지표): 실시간 거래량 상위 종목
        * 기술적 분석에서 **"거래량은 주가에 선행한다"**는 절대 원칙에 기반합니다.
        * KOSPI 및 KOSDAQ 각 시장의 실시간 총 거래량 상위 종목을 추출하여 기관/외국인 자금(Smart Money)이 유입되는 시장 유동성 병목 지점을 포착합니다.
        
        #### 5. 모멘텀: 실시간 주가 급등 종목
        * 당일 강한 상승 돌파 거래량과 함께 가격 상승 변동폭이 극대화되는 상방 모멘텀 종목을 추출합니다. 당일 주도 테마의 대장주 및 변동성 돌파 거래 타겟을 신속하게 식별할 수 있습니다.
        """)
        
    st.markdown("---")
    
    # 실시간 데이터 갱신 제어 및 마지막 업데이트 시간
    col_refresh, col_time = st.columns([1.5, 5])
    with col_refresh:
        if st.button("🔄 실시간 수급 데이터 갱신", use_container_width=True):
            st.cache_data.clear()
            if "market_data" in st.session_state:
                del st.session_state["market_data"]
            st.rerun()
            
    with col_time:
        last_updated = st.session_state.get("market_last_updated", "미조회")
        st.markdown(f"<div style='padding-top: 5px; color: gray;'>마지막 업데이트 시각: <b>{last_updated}</b> (60초 주기로 최신 정보 갱신)</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 60초 TTL 캐싱 데이터 로딩
    if "market_data" not in st.session_state or "foreigner" not in st.session_state["market_data"] or "reports" not in st.session_state["market_data"] or "institution" not in st.session_state["market_data"]:
        with st.spinner("실시간 시장 데이터(인기검색/외인순매수/기관순매수/리포트/거래상위/급등)를 크롤링 중입니다..."):
            # 1. 인기검색
            df_popular = crawl_naver_popular_stocks()
            # 2. 외국인 순매수 상위
            df_foreigner = crawl_naver_foreigner_top()
            # 3. 기관 순매수 상위
            df_institution = crawl_naver_institution_top()
            # 4. 애널리스트 리포트 언급 종목
            df_reports = crawl_naver_analyst_reports()
            # 5. 거래상위 (KOSPI & KOSDAQ)
            df_vol_kospi = crawl_naver_volume_top(0)
            df_vol_kosdaq = crawl_naver_volume_top(1)
            # 6. 주가급등 (KOSPI & KOSDAQ)
            df_surge_kospi = crawl_naver_price_surge(0)
            df_surge_kosdaq = crawl_naver_price_surge(1)
            
            st.session_state["market_data"] = {
                "popular": df_popular,
                "foreigner": df_foreigner,
                "institution": df_institution,
                "reports": df_reports,
                "vol_kospi": df_vol_kospi,
                "vol_kosdaq": df_vol_kosdaq,
                "surge_kospi": df_surge_kospi,
                "surge_kosdaq": df_surge_kosdaq
            }
            st.session_state["market_last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
    # Load from session state
    m_data = st.session_state["market_data"]
    df_popular = m_data["popular"]
    df_foreigner = m_data.get("foreigner", pd.DataFrame())
    df_institution = m_data.get("institution", pd.DataFrame())
    df_reports = m_data.get("reports", pd.DataFrame())
    df_vol_kospi = m_data["vol_kospi"]
    df_vol_kosdaq = m_data["vol_kosdaq"]
    df_surge_kospi = m_data["surge_kospi"]
    df_surge_kosdaq = m_data["surge_kosdaq"]


    # --- 1행: 실시간 인기 검색 종목 (개인 관심도) & 외국인 순매수 상위 종목 (외국인 관심도) ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔥 1. 실시간 포털 인기 검색 종목 (개인 관심도)")
        st.caption("네이버 금융에서 현재 개인 투자자들이 가장 집중적으로 검색하고 토론하는 실시간 관심 종목 리스트입니다.")
        render_naver_sise_table(df_popular)
        
    with col_b:
        st.subheader("🌍 2. 외국인 실시간 순매수 상위 종목 (외국인 관심도)")
        st.caption("당일 외국인 투자자들의 자금이 유입되며 순매수 강세를 보이는 시장 주도 종목 리스트입니다.")
        render_naver_sise_table(df_foreigner)
        
    st.markdown("---")

    # --- 1.5행: 기관 순매수 상위 종목 ---
    st.subheader("🏦 3. 기관 실시간 순매수 상위 종목 (기관 관심도)")
    st.caption("당일 기관 투자자들이 순매수하는 상위 종목입니다. 외국인과 함께 시장을 주도하는 스마트머니의 흐름을 파악할 수 있습니다.")
    col_inst, col_inst_fill = st.columns([1, 1])
    with col_inst:
        render_naver_sise_table(df_institution)
    with col_inst_fill:
        st.info("💡 **기관 순매수 해석 가이드**\n\n기관과 외국인이 동시에 순매수하는 종목은 **'더블 수급 호재'** 신호로, 단기 상승 모멘텀이 발생할 가능성이 높습니다. 두 리스트의 공통 종목을 우선 주목하세요.")
    
    st.markdown("---")


    # --- 2행: 실시간 거래량 상위 종목 (KOSPI vs KOSDAQ) ---
    st.subheader("📊 4. 실시간 거래량 상위 종목 (기관/외인 수급 및 거래 유동성)")
    st.caption("KOSPI 및 KOSDAQ 시장에서 대량 거래가 유입되어 유동성 쏠림이 극대화된 종목 리스트입니다. 거래량은 수급의 선행 시그널입니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🇰🇷 KOSPI 거래량 상위")
        render_naver_sise_table(df_vol_kospi)
    with col2:
        st.markdown("#### 🇰🇷 KOSDAQ 거래량 상위")
        render_naver_sise_table(df_vol_kosdaq)
    st.markdown("---")

    # --- 3행: 실시간 주가 급등 종목 (KOSPI vs KOSDAQ) ---
    st.subheader("⚡ 5. 실시간 주가 급등 종목 (상방 모멘텀 및 호재 돌파)")
    st.caption("당일 거래량 증가를 수반하며 가격 등락 변동률이 가장 높은 상방 돌파 주도 테마 종목 리스트입니다.")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🇰🇷 KOSPI 주가 급등")
        render_naver_sise_table(df_surge_kospi)
    with col4:
        st.markdown("#### 🇰🇷 KOSDAQ 주가 급등")
        render_naver_sise_table(df_surge_kosdaq)

    st.markdown("---")

    # --- 4행: 최근 증권사 애널리스트 리포트 언급 종목 ---
    st.subheader("📑 6. 최근 증권사 애널리스트 리포트 언급 종목 (최신 분석 동향)")
    st.caption("국내 주요 증권사 리서치 센터에서 발행한 최신 종목 분석 리포트 현황입니다. 애널리스트의 분석 대상이 된 최신 관심 종목 흐름을 나타냅니다.")
    render_analyst_reports_table(df_reports)

if __name__ == "__main__":
    render_youtube_rank_page()
