import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

# Header for crawler requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

@st.cache_data(ttl=60)  # 60초 캐싱으로 실시간성을 극대화하되 단기간 반복 로딩 부담 제어
def crawl_naver_popular_stocks() -> pd.DataFrame:
    """
    네이버 금융 실시간 인기 검색 종목 (Top 30) 크롤링
    - 반환: DataFrame [순위, 종목명, 티커, 검색 비율, 현재가, 등락률]
    """
    url = "https://finance.naver.com/sise/lastsearch2.naver"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        table = soup.find('table', {'class': 'type_5'})
        if not table: return pd.DataFrame()
            
        rows = table.find_all('tr')
        stocks = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 6:
                rank_txt = cols[0].get_text(strip=True)
                name_a = cols[1].find('a')
                if name_a:
                    name = name_a.get_text(strip=True)
                    ticker = name_a['href'].split('code=')[-1]
                    search_ratio = cols[2].get_text(strip=True)
                    price = cols[3].get_text(strip=True)
                    change_val = cols[4].get_text(strip=True)
                    change_dir = ""
                    change_span = cols[4].find('span')
                    if change_span:
                        change_class = change_span.get('class', [])
                        if 'red' in "".join(change_class) or 'ico_up' in "".join(change_class):
                            change_dir = "▲"
                        elif 'blue' in "".join(change_class) or 'ico_down' in "".join(change_class):
                            change_dir = "▼"
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
        st.warning(f"인기 검색 종목 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def crawl_naver_volume_top(sosok: int) -> pd.DataFrame:
    """
    네이버 금융 실시간 거래량 상위 종목 크롤링
    - sosok: 0 = KOSPI, 1 = KOSDAQ
    """
    url = f"https://finance.naver.com/sise/sise_quant.naver?sosok={sosok}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        table = soup.find('table', {'class': 'type_2'})
        if not table: return pd.DataFrame()
        
        rows = table.find_all('tr')
        stocks = []
        rank_counter = 1
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 10:
                name_a = cols[1].find('a')
                if name_a:
                    name = name_a.get_text(strip=True)
                    ticker = name_a['href'].split('code=')[-1]
                    price = cols[2].get_text(strip=True)
                    chg_span = cols[3].find('span')
                    change_dir = ""
                    if chg_span:
                        change_class = chg_span.get('class', [])
                        if 'red' in "".join(change_class) or 'ico_up' in "".join(change_class):
                            change_dir = "▲"
                        elif 'blue' in "".join(change_class) or 'ico_down' in "".join(change_class):
                            change_dir = "▼"
                    change_val = cols[3].get_text(strip=True)
                    pct = cols[4].get_text(strip=True)
                    vol = cols[5].get_text(strip=True)
                    
                    stocks.append({
                        '순위': str(rank_counter),
                        '종목명': name,
                        '티커': ticker,
                        '현재가': price,
                        '등락률': f"{change_dir} {change_val} ({pct})",
                        '거래량': vol
                    })
                    rank_counter += 1
        return pd.DataFrame(stocks)
    except Exception as e:
        st.warning(f"거래량 상위 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def crawl_naver_price_surge(sosok: int) -> pd.DataFrame:
    """
    네이버 금융 실시간 주가 급등 종목 크롤링
    - sosok: 0 = KOSPI, 1 = KOSDAQ
    """
    url = f"https://finance.naver.com/sise/sise_low_up.naver?sosok={sosok}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        table = soup.find('table', {'class': 'type_2'})
        if not table: return pd.DataFrame()
        
        rows = table.find_all('tr')
        stocks = []
        rank_counter = 1
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 10:
                name_a = cols[2].find('a')  # 급등 비율 컬럼이 1번에 오므로, 2번에서 이름 추출
                if name_a:
                    name = name_a.get_text(strip=True)
                    ticker = name_a['href'].split('code=')[-1]
                    price = cols[3].get_text(strip=True)
                    chg_span = cols[4].find('span')
                    change_dir = ""
                    if chg_span:
                        change_class = chg_span.get('class', [])
                        if 'red' in "".join(change_class) or 'ico_up' in "".join(change_class):
                            change_dir = "▲"
                        elif 'blue' in "".join(change_class) or 'ico_down' in "".join(change_class):
                            change_dir = "▼"
                    change_val = cols[4].get_text(strip=True)
                    pct = cols[5].get_text(strip=True)
                    vol = cols[6].get_text(strip=True)
                    
                    stocks.append({
                        '순위': str(rank_counter),
                        '종목명': name,
                        '티커': ticker,
                        '현재가': price,
                        '등락률': f"{change_dir} {change_val} ({pct})",
                        '거래량': vol
                    })
                    rank_counter += 1
        return pd.DataFrame(stocks)
    except Exception as e:
        st.warning(f"주가 급등 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def crawl_naver_foreigner_top() -> pd.DataFrame:
    """
    네이버 금융 실시간 외국인 순매수 상위 종목 크롤링
    - 반환: DataFrame [순위, 종목명, 티커, 현재가, 등락]
    """
    url = "https://finance.naver.com/sise/sise_deal_rank.naver?investor_gubun=1000"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        table = soup.find('table', {'class': 'type_r1'})
        if not table: return pd.DataFrame()
            
        rows = table.find_all('tr')
        stocks = []
        rank_counter = 1
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                name_a = cols[1].find('a')
                if name_a:
                    name = name_a.get_text(strip=True)
                    ticker = name_a['href'].split('code=')[-1]
                    price = cols[2].get_text(strip=True)
                    
                    change_dir = ""
                    if len(cols) >= 4:
                        change_img = cols[3].find('img')
                        if change_img:
                            alt_val = change_img.get('alt', '')
                            if 'up' in alt_val:
                                change_dir = "▲"
                            elif 'down' in alt_val:
                                change_dir = "▼"
                                
                    stocks.append({
                        '순위': str(rank_counter),
                        '종목명': name,
                        '티커': ticker,
                        '현재가': price,
                        '등락': change_dir
                    })
                    rank_counter += 1
        return pd.DataFrame(stocks)
    except Exception as e:
        st.warning(f"외국인 순매수 상위 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # 리포트는 업데이트 주기가 길어 5분 캐싱
def crawl_naver_analyst_reports() -> pd.DataFrame:
    """
    네이버 금융 리서치 종목분석 리포트 크롤링
    - 반환: DataFrame [순위, 종목명, 티커, 리포트 제목, 증권사, 작성일]
    """
    url = "https://finance.naver.com/research/company_list.naver"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        table = soup.find('table', {'class': 'type_1'})
        if not table: return pd.DataFrame()
            
        rows = table.find_all('tr')
        reports = []
        rank_counter = 1
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                stock_td = cols[0]
                stock_a = stock_td.find('a')
                stock_name = stock_a.get_text(strip=True) if stock_a else stock_td.get_text(strip=True)
                
                ticker = ""
                if stock_a and 'code=' in stock_a.get('href', ''):
                    ticker = stock_a['href'].split('code=')[-1]
                
                title_td = cols[1]
                title_a = title_td.find('a')
                title = title_a.get_text(strip=True) if title_a else title_td.get_text(strip=True)
                
                brokerage = cols[2].get_text(strip=True)
                date = cols[4].get_text(strip=True)
                
                if stock_name and title:
                    reports.append({
                        '순위': str(rank_counter),
                        '종목명': f"{stock_name} ({ticker})" if ticker else stock_name,
                        '티커': ticker,
                        '리포트 제목': title,
                        '증권사': brokerage,
                        '작성일': date
                    })
                    rank_counter += 1
                    if rank_counter > 10:  # 최대 10개
                        break
        return pd.DataFrame(reports)
    except Exception as e:
        st.warning(f"애널리스트 리포트 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

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
        
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=388,
        hide_index=True,
        column_config=cols_config
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
        
        #### 3. 유동성 (선행 지표): 실시간 거래량 상위 종목
        * 기술적 분석에서 **"거래량은 주가에 선행한다"**는 절대 원칙에 기반합니다.
        * KOSPI 및 KOSDAQ 각 시장의 실시간 총 거래량 상위 종목을 추출하여 기관/외국인 자금(Smart Money)이 유입되는 시장 유동성 병목 지점을 포착합니다.
        
        #### 4. 모멘텀: 실시간 주가 급등 종목
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
    if "market_data" not in st.session_state or "foreigner" not in st.session_state["market_data"] or "reports" not in st.session_state["market_data"]:
        with st.spinner("실시간 시장 데이터(인기검색/외인순매수/리포트/거래상위/급등)를 크롤링 중입니다..."):
            # 1. 인기검색
            df_popular = crawl_naver_popular_stocks()
            # 2. 외국인 순매수 상위
            df_foreigner = crawl_naver_foreigner_top()
            # 3. 애널리스트 리포트 언급 종목
            df_reports = crawl_naver_analyst_reports()
            # 4. 거래상위 (KOSPI & KOSDAQ)
            df_vol_kospi = crawl_naver_volume_top(0)
            df_vol_kosdaq = crawl_naver_volume_top(1)
            # 5. 주가급등 (KOSPI & KOSDAQ)
            df_surge_kospi = crawl_naver_price_surge(0)
            df_surge_kosdaq = crawl_naver_price_surge(1)
            
            st.session_state["market_data"] = {
                "popular": df_popular,
                "foreigner": df_foreigner,
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

    # --- 2행: 실시간 거래량 상위 종목 (KOSPI vs KOSDAQ) ---
    st.subheader("📊 3. 실시간 거래량 상위 종목 (기관/외인 수급 및 거래 유동성)")
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
    st.subheader("⚡ 4. 실시간 주가 급등 종목 (상방 모멘텀 및 호재 돌파)")
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
    st.subheader("📑 5. 최근 증권사 애널리스트 리포트 언급 종목 (최신 분석 동향)")
    st.caption("국내 주요 증권사 리서치 센터에서 발행한 최신 종목 분석 리포트 현황입니다. 애널리스트의 분석 대상이 된 최신 관심 종목 흐름을 나타냅니다.")
    render_naver_sise_table(df_reports)

if __name__ == "__main__":
    render_youtube_rank_page()
