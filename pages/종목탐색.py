import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import re
import concurrent.futures

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
    URL: sise_trans_style.naver?sosok=0 (KOSPI 투자자별 매매동향)
    - Table[0]: 외국인 순매수 상위 / Table[1]: 기관 순매수 상위
    - 반환: DataFrame [순위, 종목명, 티커, 현재가]
    """
    url = "https://finance.naver.com/sise/sise_trans_style.naver?sosok=0"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        # Table[0] = 외국인 순매수 상위
        tables = soup.find_all('table', {'class': 'type_r1'})
        if not tables: return pd.DataFrame()
        table = tables[0]  # 외국인 순매수
            
        rows = table.find_all('tr')
        stocks = []
        rank_counter = 1
        for row in rows:
            cols = row.find_all('td')
            # cols[0]=빈셀, cols[1]=종목명(링크포함), cols[2]=현재가, cols[3]=등락 이미지
            if len(cols) >= 3:
                name_a = cols[1].find('a')
                if name_a:
                    name = name_a.get_text(strip=True)
                    href = name_a.get('href', '')
                    ticker = href.split('code=')[-1] if 'code=' in href else ''
                    price = cols[2].get_text(strip=True)
                    
                    change_dir = ""
                    if len(cols) >= 4:
                        change_img = cols[3].find('img')
                        if change_img:
                            alt_val = change_img.get('alt', '').lower()
                            if 'up' in alt_val or '상승' in alt_val:
                                change_dir = "▲"
                            elif 'down' in alt_val or '하락' in alt_val:
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

@st.cache_data(ttl=60)
def crawl_naver_institution_top() -> pd.DataFrame:
    """
    네이버 금융 실시간 기관 순매수 상위 종목 크롤링
    URL: sise_trans_style.naver?sosok=0 (KOSPI 투자자별 매매동향)
    - Table[1]: 기관 순매수 상위
    - 반환: DataFrame [순위, 종목명, 티커, 현재가]
    """
    url = "https://finance.naver.com/sise/sise_trans_style.naver?sosok=0"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        # Table[1] = 기관 순매수 상위
        tables = soup.find_all('table', {'class': 'type_r1'})
        if len(tables) < 2: return pd.DataFrame()
        table = tables[1]  # 기관 순매수
            
        rows = table.find_all('tr')
        stocks = []
        rank_counter = 1
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                name_a = cols[1].find('a')
                if name_a:
                    name = name_a.get_text(strip=True)
                    href = name_a.get('href', '')
                    ticker = href.split('code=')[-1] if 'code=' in href else ''
                    price = cols[2].get_text(strip=True)
                    
                    change_dir = ""
                    if len(cols) >= 4:
                        change_img = cols[3].find('img')
                        if change_img:
                            alt_val = change_img.get('alt', '').lower()
                            if 'up' in alt_val or '상승' in alt_val:
                                change_dir = "▲"
                            elif 'down' in alt_val or '하락' in alt_val:
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
        st.warning(f"기관 순매수 상위 크롤링 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # 리포트는 업데이트 주기가 길어 5분 캐싱
def crawl_naver_analyst_reports() -> pd.DataFrame:
    """
    네이버 금융 리서치 종목분석 리포트 크롤링 (언급 빈도 순 정렬)
    - 반환: DataFrame [순위, 종목명, 티커, 언급 빈도, 최근 리포트 제목, 발행 증권사, 최근 발행일]
    """
    reports = []
    # 2개 페이지(60개 리포트) 수집하여 빈도 집계의 정확성 향상
    for page in [1, 2]:
        url = f"https://finance.naver.com/research/company_list.naver?page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
            
            table = soup.find('table', {'class': 'type_1'})
            if not table: continue
                
            rows = table.find_all('tr')
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

                    # 리포트 페이지 URL (company_read.naver?nid=XXXXX)
                    report_href = title_a.get('href', '') if title_a else ''
                    report_url = f"https://finance.naver.com/research/{report_href}" if report_href and not report_href.startswith('http') else report_href

                    # PDF 직접 링크 (cols[3]에 있음)
                    pdf_url = ""
                    if len(cols) > 3:
                        pdf_a = cols[3].find('a')
                        if pdf_a:
                            pdf_url = pdf_a.get('href', '')

                    brokerage = cols[2].get_text(strip=True)
                    date = cols[4].get_text(strip=True)
                    
                    if stock_name and title:
                        reports.append({
                            'stock_name': stock_name,
                            'ticker': ticker,
                            'title': title,
                            'report_url': report_url,
                            'pdf_url': pdf_url,
                            'brokerage': brokerage,
                            'date': date
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
        # reps[0]은 스크래핑 순서상 가장 최신 리포트
        latest_rep = reps[0]
        freq_list.append({
            'stock_name': stock_name,
            'ticker': latest_rep['ticker'],
            'count': len(reps),
            'latest_title': latest_rep['title'],
            'latest_report_url': latest_rep['report_url'],
            'latest_pdf_url': latest_rep['pdf_url'],
            'latest_brokerage': latest_rep['brokerage'],
            'latest_date': latest_rep['date']
        })
        
    # 빈도(count) 내림차순, 동일 빈도 시 최신 발행일(latest_date) 내림차순 정렬
    freq_list.sort(key=lambda x: (x['count'], x['latest_date']), reverse=True)
    top_items = freq_list[:10]

    # 상위 10개 종목의 리포트 상세 페이지를 병렬 크롤링하여 투자의견/목표가/요약 키워드 추출
    def _fetch_report_detail(item: dict) -> dict:
        """Thread worker: 리포트 상세 페이지에서 투자의견, 목표가, 핵심 요약 키워드 크롤링"""
        result = {'opinion': '', 'target_price': '', 'summary': ''}
        url = item.get('latest_report_url', '')
        if not url:
            return result
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.content.decode('euc-kr', 'replace'), 'html.parser')
            content_area = soup.find(id='contentarea')
            if not content_area:
                return result

            # 소스 태그(목표가|투자의견) 찾기
            source_p = content_area.find('p', {'class': 'source'})
            if source_p:
                src_txt = source_p.get_text(strip=True)
                # 목표가 판두 e.g. "목표가 1,600,000|투자의견 Buy"
                tp_m = re.search(r'목표가\s*([\d,]+)', src_txt)
                op_m = re.search(r'투자의견\s*([\w ]+)', src_txt)
                if tp_m:
                    result['target_price'] = f"목표가 {tp_m.group(1)}"
                if op_m:
                    result['opinion'] = op_m.group(1).strip()[:20]

            # 학습 view_cnt td 내 핸실 요약 문단
            view_td = content_area.find('td', {'class': 'view_cnt'})
            if view_td:
                paras = [p.get_text(strip=True) for p in view_td.find_all('p') if len(p.get_text(strip=True)) > 20]
                if paras:
                    result['summary'] = paras[0][:160]  # 첫 번째 단락의 첫 160자
        except Exception:
            pass
        return result

    # ThreadPoolExecutor로 병렬 요청 (max 5 스레드, TTL 5분 캐싱으로 부담 최소화)
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
        # 투자의견 + 목표가 조합
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

        # 리포트 페이지 URL 우선, 없으면 PDF URL 사용
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
    KOSPI / KOSDAQ 공통 데이터 프레임 렌더링 테이블 -> Airbnb 리스팅 카드 뷰포트로 전환
    """
    if df.empty:
        st.info("📊 실시간 데이터를 불러올 수 없습니다.")
        return

    df_display = df.head(10).copy()
    
    # Airbnb 리스팅 카드 그리드 스타일로 종목 렌더링
    cards_html = ""
    for idx, row in df_display.iterrows():
        rank = str(row['순위']).strip()
        name = row['종목명']
        ticker = row.get('티커', '')
        
        # 순위 배지 디자인
        badge_emoji = "🥇" if rank == "1" else ("🥈" if rank == "2" else ("🥉" if rank == "3" else ""))
        badge_style = "background-color: #ffd700;" if rank == "1" else ("background-color: #e6e6e6;" if rank == "2" else ("background-color: #f5c293;" if rank == "3" else "background-color: #f2f2f2;"))
        rank_html = f'<span style="display:inline-block; border-radius:12px; padding:3px 8px; font-size:11px; font-weight:700; color:#222222; {badge_style} margin-right:8px;">{badge_emoji} {rank}위</span>'
        
        # 현재가 & 등락률 파싱
        price = row.get('현재가', '')
        change_rate = row.get('등락률', '')
        
        # 등락 부호에 따른 색상 정의 (Rausch vs Muted Gray)
        text_color = "#ff385c" if "▲" in change_rate or "+" in change_rate else ("#222222" if "▼" in change_rate or "-" in change_rate else "#6a6a6a")
        
        # 검색 비율 또는 거래량 추가 정보
        meta_html = ""
        if "검색 비율" in row:
            meta_html = f'<div style="font-size:13px; color:#6a6a6a; margin-top:4px;">검색 비율: {row["검색 비율"]}</div>'
        elif "거래량" in row:
            meta_html = f'<div style="font-size:13px; color:#6a6a6a; margin-top:4px;">거래량: {row["거래량"]} 주</div>'
            
        cards_html += f"""
        <div style="background-color:#ffffff; border:1px solid #dddddd; border-radius:14px; padding:16px; margin-bottom:12px; 
                    box-shadow: rgba(0, 0, 0, 0.02) 0 0 0 1px, rgba(0, 0, 0, 0.04) 0 2px 4px; transition: all 0.2s ease; cursor:pointer;" 
             class="airbnb-card-sise">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center;">
                    {rank_html}
                    <span style="font-weight:600; font-size:15px; color:#222222;">{name}</span>
                    <span style="font-size:11px; color:#929292; margin-left:6px;">{ticker}</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-weight:600; font-size:15px; color:#222222;">{price} 원</div>
                    <div style="font-weight:500; font-size:13px; color:{text_color};">{change_rate}</div>
                </div>
            </div>
            {meta_html}
        </div>
        """
        
    st.markdown(f"""
    <style>
    .airbnb-card-sise:hover {{
        transform: translateY(-2px);
        box-shadow: rgba(0, 0, 0, 0.02) 0 0 0 1px, rgba(0, 0, 0, 0.04) 0 4px 10px, rgba(0, 0, 0, 0.1) 0 6px 12px !important;
        border-color: #ff385c !important;
    }}
    </style>
    <div>
        {cards_html}
    </div>
    """, unsafe_allow_html=True)


def render_analyst_reports_table(df: pd.DataFrame):
    """
    애널리스트 리포트 전용 렌더링 테이블 -> Airbnb 리스팅 카드 뷰포트로 전환
    """
    if df.empty:
        st.info("📊 리포트 데이터를 불러올 수 없습니다.")
        return

    df_display = df.head(10).copy()
    
    cards_html = ""
    for idx, row in df_display.iterrows():
        rank = str(row['순위']).strip()
        name = row['종목명']
        ticker = row.get('티커', '')
        
        # 순위 배지
        badge_emoji = "🥇" if rank == "1" else ("🥈" if rank == "2" else ("🥉" if rank == "3" else ""))
        badge_style = "background-color: #ffd700;" if rank == "1" else ("background-color: #e6e6e6;" if rank == "2" else ("background-color: #f5c293;" if rank == "3" else "background-color: #f2f2f2;"))
        rank_html = f'<span style="display:inline-block; border-radius:12px; padding:3px 8px; font-size:11px; font-weight:700; color:#222222; {badge_style} margin-right:8px;">{badge_emoji} {rank}위</span>'
        
        title = row.get('최근 리포트 제목', row.get('리포트 제목', ''))
        keywords = row.get('핵심 키워드', '')
        broker = row.get('발행 증권사', row.get('증권사', ''))
        date = row.get('최근 발행일', row.get('작성일', ''))
        link = row.get('리포트 링크', '')
        freq = row.get('언급 빈도', '1')
        
        link_html = f'<a href="{link}" target="_blank" style="display:inline-block; background-color:#ff385c; color:#ffffff; font-size:12px; font-weight:600; padding:6px 12px; border-radius:8px; text-decoration:none; transition: background-color 0.2s;">리포트 열기 ↗️</a>' if link else ''
        
        cards_html += f"""
        <div style="background-color:#ffffff; border:1px solid #dddddd; border-radius:14px; padding:18px; margin-bottom:14px; 
                    box-shadow: rgba(0, 0, 0, 0.02) 0 0 0 1px, rgba(0, 0, 0, 0.04) 0 2px 4px; transition: all 0.2s ease;" 
             class="airbnb-card-report">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
                <div style="display:flex; align-items:center;">
                    {rank_html}
                    <span style="font-weight:700; font-size:16px; color:#222222;">{name}</span>
                    <span style="font-size:12px; color:#929292; margin-left:6px;">{ticker}</span>
                </div>
                <div style="background-color:#f7f7f7; border-radius:12px; padding:3px 10px; font-size:12px; font-weight:600; color:#ff385c;">
                    언급 빈도: {freq}회
                </div>
            </div>
            <div style="font-size:14px; font-weight:500; color:#222222; margin-top:8px; line-height:1.4;">
                {title}
            </div>
            <div style="font-size:13px; color:#ff385c; margin-top:6px; font-weight:500;">
                🔑 핵심 키워드: {keywords}
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; border-top:1px solid #ebebeb; padding-top:10px;">
                <span style="font-size:12px; color:#6a6a6a;">{broker} | {date}</span>
                {link_html}
            </div>
        </div>
        """
        
    st.markdown(f"""
    <style>
    .airbnb-card-report:hover {{
        transform: translateY(-2px);
        box-shadow: rgba(0, 0, 0, 0.02) 0 0 0 1px, rgba(0, 0, 0, 0.04) 0 4px 12px, rgba(0, 0, 0, 0.12) 0 8px 16px !important;
        border-color: #ff385c !important;
    }}
    .airbnb-card-report a:hover {{
        background-color: #e00b41 !important;
    }}
    </style>
    <div>
        {cards_html}
    </div>
    """, unsafe_allow_html=True)

def render_youtube_rank_page():
    """
    실시간 수급 및 가격 지표 기반 선행 종목 탐색 엔진 (app.py 호환 진입점)
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* 글로벌 폰트 및 라이트 모드 강제 */
    html, body, [data-testid="stAppViewContainer"], .st-emotion-cache-1102t3n, .st-emotion-cache-q8sbsg {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        color: #222222 !important;
        background-color: #ffffff !important;
    }
    
    /* 제목 (Airbnb 디스플레이 폰트 스케일 적용) */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-size: clamp(24px, 3.5vw, 28px) !important;
        font-weight: 700 !important;
        color: #222222 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 16px !important;
    }
    
    h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: #222222 !important;
        font-weight: 600 !important;
    }

    h2 {
        font-size: 21px !important;
        margin-top: 32px !important;
        margin-bottom: 16px !important;
    }

    h3 {
        font-size: 18px !important;
        margin-top: 24px !important;
        margin-bottom: 12px !important;
    }
    
    /* Airbnb 카드 형태 효과 */
    .airbnb-card, .ai-analysis-box, div[data-testid="stExpander"], div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #dddddd !important;
        border-radius: 14px !important;
        box-shadow: rgba(0, 0, 0, 0.02) 0 0 0 1px, rgba(0, 0, 0, 0.04) 0 2px 6px, rgba(0, 0, 0, 0.1) 0 4px 8px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }

    .airbnb-card:hover, .ai-analysis-box:hover {
        transform: translateY(-2px) !important;
        box-shadow: rgba(0, 0, 0, 0.02) 0 0 0 1px, rgba(0, 0, 0, 0.04) 0 4px 12px, rgba(0, 0, 0, 0.15) 0 8px 16px !important;
    }
    
    /* Expander 스타일 */
    div[data-testid="stExpander"] {
        border: 1px solid #dddddd !important;
        border-radius: 14px !important;
        box-shadow: rgba(0, 0, 0, 0.02) 0 2px 6px !important;
        margin-bottom: 20px !important;
    }

    div[data-testid="stExpander"] [data-testid="stExpanderHeader"] {
        font-weight: 600 !important;
        color: #222222 !important;
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
