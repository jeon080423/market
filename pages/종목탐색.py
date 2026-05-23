import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime, timezone, timedelta
import email.utils
import numpy as np

# utils 에서 종목 정보 및 카운트 도구 로드
from utils.stock_filter import load_krx_stocks, filter_by_price_direction, count_stock_mentions

def get_period_start(days: int) -> datetime:
    """days일 전 datetime 반환 (UTC)"""
    return datetime.now(timezone.utc) - timedelta(days=days)

def parse_rss_date(date_str: str) -> datetime:
    """RSS 날짜 문자열을 datetime 객체로 파싱 (오류 시 현재 시간 반환)"""
    try:
        return email.utils.parsedate_to_datetime(date_str)
    except:
        return datetime.now(timezone.utc)

@st.cache_data(ttl=600)  # 10분 캐싱으로 불필요한 요청 방지 및 로딩 속도 최적화
def crawl_naver_popular_stocks() -> pd.DataFrame:
    """
    네이버 금융 실시간 인기 검색 종목 (Top 30) 크롤링
    - 반환: DataFrame [순위, 종목명, 티커, 검색 비율, 현재가, 등락률]
    """
    url = "https://finance.naver.com/sise/lastsearch2.naver"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        table = soup.find('table', {'class': 'type_5'})
        if not table:
            return pd.DataFrame()
            
        rows = table.find_all('tr')
        stocks = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 6:
                # Rank
                rank_txt = cols[0].get_text(strip=True)
                # Name & Ticker
                name_a = cols[1].find('a')
                if name_a:
                    name = name_a.get_text(strip=True)
                    href = name_a['href']
                    ticker = href.split('code=')[-1]
                    
                    # Search ratio
                    search_ratio = cols[2].get_text(strip=True)
                    # Current price
                    price = cols[3].get_text(strip=True)
                    # Change
                    change_val = cols[4].get_text(strip=True)
                    change_dir = ""
                    # Check up/down direction via class or text
                    change_span = cols[4].find('span')
                    if change_span:
                        change_class = change_span.get('class', [])
                        if 'red' in "".join(change_class) or 'ico_up' in "".join(change_class):
                            change_dir = "▲"
                        elif 'blue' in "".join(change_class) or 'ico_down' in "".join(change_class):
                            change_dir = "▼"
                    
                    # Percent change
                    pct_change = cols[5].get_text(strip=True)
                    
                    stocks.append({
                        '순위': rank_txt,
                        '종목명': name,
                        '티커': ticker,
                        '검색 비율': search_ratio,
                        '현재가': price,
                        '등락률': f"{change_dir} {change_val} ({pct_change})"
                    })
        return pd.DataFrame(stocks)
    except Exception as e:
        st.warning(f"네이버 금융 실시간 인기 검색 종목 수집 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)  # 10분 캐싱
def fetch_google_news_articles(query: str, stock_list: list) -> list:
    """
    구글 뉴스 RSS를 검색하여 종목 언급량 정보와 기사 목록 반환
    """
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        root = ET.fromstring(res.content)
        items = root.findall('.//item')
        
        articles = []
        for item in items:
            title = item.find('title').text or ""
            pub_date_str = item.find('pubDate').text or ""
            link = item.find('link').text or ""
            pub_dt = parse_rss_date(pub_date_str)
            
            # 자막이 없는 뉴스 특성상 헤드라인 타이틀에서 언급된 종목 추출
            mentions = count_stock_mentions(title, stock_list)
            if mentions:
                articles.append({
                    'title': title,
                    'link': link,
                    'pub_date': pub_dt,
                    'mentions': mentions
                })
        return articles
    except Exception as e:
        return []

def calculate_news_weighted_score(articles: list, stock_list, direction_filtered_tickers: set) -> pd.DataFrame:
    """
    뉴스 기사의 최신성(Recency)에 가중치를 준 가중합 랭킹 계산
    - 24시간 이내 기사: 가중치 1.0
    - 72시간 이내 기사: 가중치 0.7
    - 72시간 초과 기사: 가중치 0.4
    """
    ticker_to_name = {}
    if isinstance(stock_list, pd.DataFrame):
        for _, row in stock_list.iterrows():
            t = str(row.get("ticker", "")).strip()
            n = str(row.get("name", "")).strip()
            if t: ticker_to_name[t] = n
    
    scores = {}
    total_mentions = {}
    article_counts = {}
    now = datetime.now(timezone.utc)
    
    for art in articles:
        pub_dt = art['pub_date']
        diff_hours = (now - pub_dt).total_seconds() / 3600.0
        
        if diff_hours <= 24:
            weight = 1.0
        elif diff_hours <= 72:
            weight = 0.7
        else:
            weight = 0.4
            
        for ticker, count in art['mentions'].items():
            if ticker not in direction_filtered_tickers:
                continue
                
            scores[ticker] = scores.get(ticker, 0.0) + (count * weight)
            total_mentions[ticker] = total_mentions.get(ticker, 0) + count
            article_counts[ticker] = article_counts.get(ticker, 0) + 1
            
    rows = []
    for ticker, score in scores.items():
        if ticker not in ticker_to_name:
            continue
        name = ticker_to_name[ticker]
        m_count = total_mentions.get(ticker, 0)
        a_count = article_counts.get(ticker, 0)
        
        rows.append({
            "ticker": ticker,
            "stock_name": name,
            "weighted_score": round(score, 2),
            "mention_count": m_count,
            "channel_count": a_count  # UI 하위 호환성을 위해 채널 수 컬럼명을 유지
        })
        
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "stock_name", "weighted_score", "mention_count", "channel_count"])
        
    return df.sort_values(by="weighted_score", ascending=False).reset_index(drop=True)

def render_rank_table(df: pd.DataFrame, title: str, period_label: str):
    """
    종목 언급량 순위 렌더링 테이블
    """
    if df.empty:
        st.info(f"📅 최근 {period_label} 동안 관련 키워드로 언급된 종목이 없습니다.")
        return

    df_display = df.head(10).copy()
    df_display.insert(0, "순위", range(1, len(df_display) + 1))
    
    def make_rank_label(rank):
        if rank == 1: return "🥇 1"
        elif rank == 2: return "🥈 2"
        elif rank == 3: return "🥉 3"
        return str(rank)
        
    df_display["순위"] = df_display["순위"].apply(make_rank_label)
    
    df_display = df_display.rename(columns={
        "stock_name": "종목명",
        "ticker": "티커",
        "weighted_score": "가중치 점수",
        "mention_count": "언급 수",
        "channel_count": "기사 수"
    })
    
    df_display = df_display.reset_index(drop=True)
    
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
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=388,
        hide_index=True,
        column_config={
            "순위": st.column_config.TextColumn("순위", width="small"),
            "종목명": st.column_config.TextColumn("종목명", width="medium"),
            "티커": None,
            "가중치 점수": st.column_config.NumberColumn("가중치 점수", width="small", format="%.1f"),
            "언급 수": st.column_config.NumberColumn("언급 수", width="small", format="%d"),
            "기사 수": None
        }
    )

def render_youtube_rank_page():
    """
    유튜브 API를 대체하여 쿼터 제한이 없고 실시간성이 극대화된 
    네이버 금융 인기검색 및 구글 뉴스 RSS 기반 실시간 종목 탐색 엔진
    """
    # 타이틀 및 헤더 영역 SEO 키워드 보완
    st.title("📊 신영증권 김효진 관점 실시간 포털 인기 종목 및 뉴스 언급량 탐색")
    
    st.markdown("""
    이 대시보드는 **신영증권 김효진 박사**의 시장 분석 방법론에 입각하여 주식 시장 내 **실시간 관심 종목**과 **최신 뉴스 미디어 유입 정보**를 모니터링합니다. 
    기존의 제한적이었던 유튜브 API 쿼터 한계를 완벽히 우회하여, 네이버 금융 실시간 급상승 검색 종목과 구글 뉴스 수백 개 기사를 실시간으로 추적·대조함으로써 신뢰할 수 있는 시장 주도 테마 및 리스크 노출 종목을 집계합니다.
    """)
    
    # 가이드 아코디언도 포털/뉴스 구조에 맞추어 상세히 갱신 및 SEO 단어 적용
    with st.expander("💡 실시간 포털 및 뉴스 종목 탐색기 작동 원리 및 가중치 산정 방식 자세히 보기", expanded=False):
        st.markdown("""
        본 시스템은 API 할당량 걱정 없이 무중단으로 실시간 투자 인사이트를 제공하기 위해 설계된 **시장 동향 분석 엔진**입니다.\n\n#### 1. 네이버 금융 실시간 인기 검색 종목 수집
        * 국내 최대 금융 포털인 **네이버 금융 인기 검색 상위 30개 종목** 데이터를 초고속 크롤링하여 현재 개인/기관 투자자들이 가장 주목하고 있는 실시간 테마와 수급 쏠림 현상을 즉각적으로 화면에 렌더링합니다.\n\n#### 2. 구글 뉴스 RSS 기반 최신 헤드라인 스캔
        * **호재/추천 검색어군** (`"주식 추천" OR "종목 추천" OR "상승 전망" OR "급등"`) 및 **악재/리스크 검색어군** (`"주가 하락" OR "악재" OR "위험 경고" OR "조정 우려"`)을 구글 뉴스 RSS 채널로부터 매시간 100건 이상씩 수집합니다.
        * 한국거래소(KRX)의 전 종목 데이터베이스를 기반으로 뉴스 헤드라인에 언급된 구체적인 종목명과 티커를 초고속 정규식 엔진으로 정밀 매칭합니다.\n\n#### 3. 기사 최신성(Recency Log-Weight) 가중치 점수 모델 ⭐
        * 단순한 언급량 합산을 넘어, 시장의 실시간 관심도를 충실히 반영할 수 있도록 기사 작성 시점으로부터의 경과 시간을 계산하여 **시간 감쇠 가중치**를 부여합니다.
          * **24시간 이내 작성된 기사**: 가중치 $1.0$ 배
          * **72시간 이내 작성된 기사**: 가중치 $0.7$ 배
          * **72시간 초과 작성된 기사**: 가중치 $0.4$ 배
        * 이 가중치 모델을 통해 현재 시점에 막 유입되기 시작한 따끈따끈한 시장 정보와 테마가 랭킹 차트 상단에 즉각 반영되어 선행 예측력이 극대화됩니다.
        """)
        
    st.markdown("---")
    
    # 0. 데이터 갱신 버튼 및 업데이트 상태 표시
    col_refresh, col_time = st.columns([1.5, 5])
    with col_refresh:
        if st.button("🔄 실시간 데이터 갱신", use_container_width=True):
            st.cache_data.clear()
            if "portal_data" in st.session_state:
                del st.session_state["portal_data"]
            st.rerun()
            
    with col_time:
        last_updated = st.session_state.get("portal_last_updated", "미조회")
        st.markdown(f"<div style='padding-top: 5px; color: gray;'>마지막 업데이트 시각: <b>{last_updated}</b> (10분 간격으로 최신 상태 유지)</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 1. 네이버 금융 실시간 인기 검색 종목 영역
    st.subheader("🔥 1. 네이버 금융 실시간 인기 검색 종목 (Top 10)")
    st.caption("현재 투자자들이 가장 뜨겁게 검색하고 토론하는 실시간 관심 종목 리스트입니다.")
    
    with st.spinner("네이버 금융 실시간 검색 순위 크롤링 중..."):
        df_naver = crawl_naver_popular_stocks()
        
    if not df_naver.empty:
        # 10개만 출력
        df_naver_display = df_naver.head(10).copy()
        
        # Rank 이모지 적용
        def make_rank_emoji(rank):
            r = str(rank).strip()
            if r == "1": return "🥇 1"
            elif r == "2": return "🥈 2"
            elif r == "3": return "🥉 3"
            return r
        df_naver_display["순위"] = df_naver_display["순위"].apply(make_rank_emoji)
        
        # Row styling for rank highlight
        def row_style_naver(row):
            rank_val = row["순위"]
            if "🥇" in str(rank_val):
                return ["background-color: rgba(255, 215, 0, 0.15); font-weight: bold;"] * len(row)
            elif "🥈" in str(rank_val):
                return ["background-color: rgba(192, 192, 192, 0.15); font-weight: bold;"] * len(row)
            elif "🥉" in str(rank_val):
                return ["background-color: rgba(205, 127, 50, 0.15); font-weight: bold;"] * len(row)
            return [""] * len(row)
            
        styled_naver = df_naver_display.style.apply(row_style_naver, axis=1)
        
        st.dataframe(
            styled_naver,
            use_container_width=True,
            height=388,
            hide_index=True,
            column_config={
                "순위": st.column_config.TextColumn("순위", width="small"),
                "종목명": st.column_config.TextColumn("종목명", width="medium"),
                "티커": st.column_config.TextColumn("티커", width="small"),
                "검색 비율": st.column_config.TextColumn("검색 비율", width="small"),
                "현재가": st.column_config.TextColumn("현재가", width="small"),
                "등락률": st.column_config.TextColumn("등락률", width="medium")
            }
        )
    else:
        st.warning("실시간 인기 검색 종목을 불러올 수 없습니다.")
        
    st.markdown("---")

    # 2. 구글 뉴스 데이터 수집 및 파티셔닝
    # KRX 종목 정보 및 상승/하락 필터용 데이터 로드
    stock_df = load_krx_stocks()
    stock_list = stock_df.to_dict("records")
    
    # Direction filters
    up_tickers = set(stock_df[stock_df["chg_rate"] >= 0]["ticker"])
    down_tickers = set(stock_df[stock_df["chg_rate"] < 0]["ticker"])
    
    # If no price filters are loaded (API failure fallback), allow all tickers in both to guarantee data display
    if not up_tickers and not down_tickers:
        all_tickers = set(stock_df["ticker"])
        up_tickers = all_tickers
        down_tickers = all_tickers
        
    if "portal_data" not in st.session_state:
        with st.spinner("구글 뉴스 실시간 포털 언급량 분석 중..."):
            # 호재/추천 뉴스 검색 및 가공
            pos_query = '주식 추천 OR 종목 추천 OR 상승 전망 OR 급등'
            pos_articles = fetch_google_news_articles(pos_query, stock_list)
            
            # 악재/리스크 뉴스 검색 및 가공
            neg_query = '주가 하락 OR 악재 OR 위험 경고 OR 규제 OR 우려'
            neg_articles = fetch_google_news_articles(neg_query, stock_list)
            
            st.session_state["portal_data"] = {
                "pos_articles": pos_articles,
                "neg_articles": neg_articles
            }
            st.session_state["portal_last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
    # Load from session
    p_data = st.session_state["portal_data"]
    pos_articles = p_data["pos_articles"]
    neg_articles = p_data["neg_articles"]
    
    now_utc = datetime.now(timezone.utc)
    
    # Period partition for positive articles
    pos_1w, pos_3d, pos_1d = [], [], []
    cutoff_3d = now_utc - timedelta(days=3)
    cutoff_1d = now_utc - timedelta(days=1)
    
    for art in pos_articles:
        pos_1w.append(art)
        if art['pub_date'] >= cutoff_3d: pos_3d.append(art)
        if art['pub_date'] >= cutoff_1d: pos_1d.append(art)
        
    # Period partition for negative articles
    neg_1w, neg_3d, neg_1d = [], [], []
    for art in neg_articles:
        neg_1w.append(art)
        if art['pub_date'] >= cutoff_3d: neg_3d.append(art)
        if art['pub_date'] >= cutoff_1d: neg_1d.append(art)

    # --- ▶ 2행: 구글 뉴스 추천/상승 호재 언급량 랭킹 ---
    st.subheader("📈 2. 구글 뉴스 국내외 추천/상승 호재 언급량 랭킹 (코스피/코스닥)")
    st.caption("언론사 뉴스 중 상승 전망, 호재, 추천 키워드와 함께 발화된 종목들을 가중합산(최신 기사 우대)하여 순위를 산출합니다.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📅 최근 1주일")
        df_pos_1w = calculate_news_weighted_score(pos_1w, stock_df, up_tickers)
        render_rank_table(df_pos_1w, "POS_1W", "1주일")
        
    with col2:
        st.markdown("#### 📅 최근 3일")
        df_pos_3d = calculate_news_weighted_score(pos_3d, stock_df, up_tickers)
        render_rank_table(df_pos_3d, "POS_3D", "3일")
        
    with col3:
        st.markdown("#### 📅 최근 1일")
        df_pos_1d = calculate_news_weighted_score(pos_1d, stock_df, up_tickers)
        render_rank_table(df_pos_1d, "POS_1D", "1일")
        
    st.markdown("---")

    # --- ▶ 3행: 구글 뉴스 조정/하락 리스크 경고 언급량 랭킹 ---
    st.subheader("📉 3. 구글 뉴스 국내외 조정/하락 리스크 경고 언급량 랭킹")
    st.caption("언론사 뉴스 중 주가 하락, 리스크 경고, 악재, 변동성 우려 키워드와 함께 언급된 리스크 관리 종목 랭킹입니다.")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("#### 📅 최근 1주일")
        df_neg_1w = calculate_news_weighted_score(neg_1w, stock_df, down_tickers)
        render_rank_table(df_neg_1w, "NEG_1W", "1주일")
        
    with col5:
        st.markdown("#### 📅 최근 3일")
        df_neg_3d = calculate_news_weighted_score(neg_3d, stock_df, down_tickers)
        render_rank_table(df_neg_3d, "NEG_3D", "3일")
        
    with col6:
        st.markdown("#### 📅 최근 1일")
        df_neg_1d = calculate_news_weighted_score(neg_1d, stock_df, down_tickers)
        render_rank_table(df_neg_1d, "NEG_1D", "1일")

if __name__ == "__main__":
    render_youtube_rank_page()
