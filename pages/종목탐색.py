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

@st.cache_data(ttl=120)
def crawl_pure_foreigner_rankings() -> dict:
    """
    KOSPI 및 KOSDAQ의 순수 외국인 (프로그램 제외) 순매수/순매도 데이터를 크롤링하고 정제하여 반환.
    - 반환: {
        '01': {'data': pd.DataFrame, 'date': str},  # KOSPI
        '02': {'data': pd.DataFrame, 'date': str}   # KOSDAQ
      }
    """
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://finance.naver.com/sise/sise_deal_rank.naver'
    }

    def clean_num(val):
        if not val:
            return 0.0
        val = val.replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return 0.0

    # Parallel worker function
    def fetch_iframe_page(sosok, gubun, type_, page):
        url = f"https://finance.naver.com/sise/sise_deal_rank_iframe.naver?sosok={sosok}&investor_gubun={gubun}&type={type_}&page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=8)
            res.raise_for_status()
            soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
            
            blocks = soup.find_all('div', {'class': 'box_type_ms'})
            page_data = {}
            for block in blocks:
                date_div = block.find('div', {'class': 'sise_guide_date'})
                if not date_div:
                    continue
                date_str = date_div.get_text(strip=True)
                
                if date_str not in page_data:
                    page_data[date_str] = {}
                    
                table = block.find('table')
                if not table:
                    continue
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    link = row.find('a')
                    if link and len(cols) >= 4:
                        name = link.get_text(strip=True)
                        ticker = link.get('href', '').split('code=')[-1]
                        qty = clean_num(cols[1].get_text(strip=True))
                        amt = clean_num(cols[2].get_text(strip=True))
                        
                        page_data[date_str][ticker] = {
                            'name': name,
                            'qty': qty,
                            'amt': amt
                        }
            return page_data
        except Exception:
            return {}

    # We will query 2 pages per combination to get top 80 stocks.
    # Total tasks: 2 (markets) * 2 (gubun: 1000, 9000) * 2 (types: buy, sell) * 2 (pages: 1, 2) = 16 tasks.
    tasks = []
    for sosok in ['01', '02']:
        for gubun in ['1000', '9000']:
            for type_ in ['buy', 'sell']:
                for page in [1, 2]:
                    tasks.append((sosok, gubun, type_, page))

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(fetch_iframe_page, *task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            sosok, gubun, type_, page = task
            try:
                page_data = future.result()
                key = (sosok, gubun, type_)
                if key not in results:
                    results[key] = {}
                for date_str, ticker_dict in page_data.items():
                    if date_str not in results[key]:
                        results[key][date_str] = {}
                    results[key][date_str].update(ticker_dict)
            except Exception:
                pass

    final_output = {}
    for sosok in ['01', '02']:
        sosok_dates = set()
        for gubun in ['1000', '9000']:
            for type_ in ['buy', 'sell']:
                key = (sosok, gubun, type_)
                if key in results:
                    sosok_dates.update(results[key].keys())
        
        dates_sorted = sorted(list(sosok_dates))
        if not dates_sorted:
            final_output[sosok] = {'data': pd.DataFrame(), 'date': 'N/A'}
            continue
        
        latest_date = dates_sorted[-1]
        
        f_buy = results.get((sosok, '1000', 'buy'), {}).get(latest_date, {})
        f_sell = results.get((sosok, '1000', 'sell'), {}).get(latest_date, {})
        p_buy = results.get((sosok, '9000', 'buy'), {}).get(latest_date, {})
        p_sell = results.get((sosok, '9000', 'sell'), {}).get(latest_date, {})
        
        tickers = set()
        names = {}
        for d in [f_buy, f_sell, p_buy, p_sell]:
            for ticker, info in d.items():
                tickers.add(ticker)
                names[ticker] = info['name']
                
        rows = []
        for ticker in tickers:
            name = names[ticker]
            
            if ticker in f_buy:
                f_qty = f_buy[ticker]['qty']
                f_amt = f_buy[ticker]['amt']
            elif ticker in f_sell:
                f_qty = -f_sell[ticker]['qty']
                f_amt = -f_sell[ticker]['amt']
            else:
                f_qty = 0.0
                f_amt = 0.0
                
            if ticker in p_buy:
                p_qty = p_buy[ticker]['qty']
                p_amt = p_buy[ticker]['amt']
            elif ticker in p_sell:
                p_qty = -p_sell[ticker]['qty']
                p_amt = -p_sell[ticker]['amt']
            else:
                p_qty = 0.0
                p_amt = 0.0
                
            pure_qty = f_qty - p_qty
            pure_amt = f_amt - p_amt
            
            rows.append({
                '티커': ticker,
                '종목명': name,
                '외인순매수_수량': f_qty,
                '외인순매수_금액': f_amt,
                '프로그램순매수_수량': p_qty,
                '프로그램순매수_금액': p_amt,
                '순수외인순매수_수량': pure_qty,
                '순수외인순매수_금액': pure_amt
            })
            
        df = pd.DataFrame(rows)
        final_output[sosok] = {
            'data': df,
            'date': latest_date
        }
        
    return final_output

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

def render_pure_foreigner_table(df: pd.DataFrame, sort_by="금액", is_buy=True):
    """
    순수 외국인 수급 테이블 렌더링
    - sort_by: "금액" 또는 "수량"
    - is_buy: True(순매수 상위), False(순매도 상위)
    """
    if df.empty:
        st.info("📊 데이터를 불러올 수 없습니다.")
        return

    df_copy = df.copy()
    
    if sort_by == "금액":
        sort_col = "순수외인순매수_금액"
    else:
        sort_col = "순수외인순매수_수량"
        
    df_sorted = df_copy.sort_values(by=sort_col, ascending=not is_buy)
    df_display = df_sorted.head(10).copy()
    
    sign = 1 if is_buy else -1
    
    df_display["순수 외인 수급"] = df_display[sort_col].apply(lambda x: f"{x * sign:,.1f}")
    
    if sort_by == "금액":
        df_display["전체 외인 수급"] = df_display["외인순매수_금액"].apply(lambda x: f"{x:,.1f}")
        df_display["프로그램 수급"] = df_display["프로그램순매수_금액"].apply(lambda x: f"{x:,.1f}")
        unit = "백만원"
    else:
        df_display["전체 외인 수급"] = df_display["외인순매수_수량"].apply(lambda x: f"{x:,.1f}")
        df_display["프로그램 수급"] = df_display["프로그램순매수_수량"].apply(lambda x: f"{x:,.1f}")
        unit = "천주"
        
    df_display = df_display.reset_index(drop=True)
    df_display["순위"] = [f"🥇 1" if i == 0 else f"🥈 2" if i == 1 else f"🥉 3" if i == 2 else str(i+1) for i in range(len(df_display))]
    
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
    
    cols_config = {
        "순위": st.column_config.TextColumn("순위", width="small"),
        "종목명": st.column_config.TextColumn("종목명", width="medium"),
        "순수 외인 수급": st.column_config.TextColumn(f"순수 외인 ({unit})", width="medium"),
        "전체 외인 수급": st.column_config.TextColumn(f"전체 외인 ({unit})", width="small"),
        "프로그램 수급": st.column_config.TextColumn(f"프로그램 ({unit})", width="small"),
        "티커": None,
        "외인순매수_수량": None,
        "외인순매수_금액": None,
        "프로그램순매수_수량": None,
        "프로그램순매수_금액": None,
        "순수외인순매수_수량": None,
        "순수외인순매수_금액": None
    }
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=388,
        hide_index=True,
        column_config=cols_config,
        column_order=["순위", "종목명", "순수 외인 수급", "전체 외인 수급", "프로그램 수급"]
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
    if "market_data" not in st.session_state or "foreigner" not in st.session_state["market_data"] or "reports" not in st.session_state["market_data"] or "institution" not in st.session_state["market_data"] or "pure_foreigner" not in st.session_state["market_data"]:
        with st.spinner("실시간 시장 데이터(인기검색/외인순매수/기관순매수/순수외인/리포트/거래상위/급등)를 크롤링 중입니다..."):
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
            # 7. 순수 외국인 수급 (프로그램 제외)
            dict_pure_foreigner = crawl_pure_foreigner_rankings()
            
            st.session_state["market_data"] = {
                "popular": df_popular,
                "foreigner": df_foreigner,
                "institution": df_institution,
                "reports": df_reports,
                "vol_kospi": df_vol_kospi,
                "vol_kosdaq": df_vol_kosdaq,
                "surge_kospi": df_surge_kospi,
                "surge_kosdaq": df_surge_kosdaq,
                "pure_foreigner": dict_pure_foreigner
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
    dict_pure_foreigner = m_data.get("pure_foreigner", {})

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

    # --- 1.8행: 순수 외국인 수급 상위 종목 (프로그램 제외) ---
    st.subheader("🌍 4. 순수 외국인 실시간 수급 상위 종목 (프로그램 매매 제외)")
    st.caption("당일 외국인 총 매매량에서 프로그램 매매량을 차감한 순수한 외국인의 비프로그램 수급 상위/하위 종목 리스트입니다.")
    
    # KOSPI / KOSDAQ dates
    date_kospi = dict_pure_foreigner.get('01', {}).get('date', 'N/A')
    date_kosdaq = dict_pure_foreigner.get('02', {}).get('date', 'N/A')
    st.markdown(f"<div style='font-size:0.85em; color:gray; margin-top:-10px; margin-bottom:10px;'>기준일자: KOSPI {date_kospi} / KOSDAQ {date_kosdaq} (최근 거래일 기준)</div>", unsafe_allow_html=True)
    
    # Filter selection: 금액 vs 수량
    col_filter_select, col_filter_fill = st.columns([1, 2])
    with col_filter_select:
        sort_by = st.radio("순위 기준 선택", ["금액 기준 (백만원)", "수량 기준 (천주)"], horizontal=True, label_visibility="collapsed")
        sort_unit = "금액" if "금액" in sort_by else "수량"
        
    tab_kospi, tab_kosdaq = st.tabs(["🇰🇷 KOSPI 순수 외국인 수급", "💎 KOSDAQ 순수 외국인 수급"])
    
    with tab_kospi:
        df_k = dict_pure_foreigner.get('01', {}).get('data', pd.DataFrame())
        col_kb, col_ks = st.columns(2)
        with col_kb:
            st.markdown("##### 📈 순수 외국인 순매수 상위 (Top 10)")
            render_pure_foreigner_table(df_k, sort_by=sort_unit, is_buy=True)
        with col_ks:
            st.markdown("##### 📉 순수 외국인 순매도 상위 (Top 10)")
            render_pure_foreigner_table(df_k, sort_by=sort_unit, is_buy=False)
            
    with tab_kosdaq:
        df_q = dict_pure_foreigner.get('02', {}).get('data', pd.DataFrame())
        col_qb, col_qs = st.columns(2)
        with col_qb:
            st.markdown("##### 📈 순수 외국인 순매수 상위 (Top 10)")
            render_pure_foreigner_table(df_q, sort_by=sort_unit, is_buy=True)
        with col_qs:
            st.markdown("##### 📉 순수 외국인 순매도 상위 (Top 10)")
            render_pure_foreigner_table(df_q, sort_by=sort_unit, is_buy=False)
            
    st.markdown("---")

    # --- 2행: 실시간 거래량 상위 종목 (KOSPI vs KOSDAQ) ---
    st.subheader("📊 5. 실시간 거래량 상위 종목 (기관/외인 수급 및 거래 유동성)")
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
    st.subheader("⚡ 6. 실시간 주가 급등 종목 (상방 모멘텀 및 호재 돌파)")
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
    st.subheader("📑 7. 최근 증권사 애널리스트 리포트 언급 종목 (최신 분석 동향)")
    st.caption("국내 주요 증권사 리서치 센터에서 발행한 최신 종목 분석 리포트 현황입니다. 애널리스트의 분석 대상이 된 최신 관심 종목 흐름을 나타냅니다.")
    render_analyst_reports_table(df_reports)

if __name__ == "__main__":
    render_youtube_rank_page()
