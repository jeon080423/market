import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BASE_URL = "https://finance.naver.com"

def fetch_report_summary(report_url: str) -> dict:
    """리포트 상세 페이지에서 요약 정보 크롤링"""
    result = {'opinion': '', 'target_price': '', 'summary': ''}
    if not report_url:
        return result
    try:
        r = requests.get(report_url, headers=HEADERS, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.content.decode('euc-kr', 'replace'), 'html.parser')

        # 투자의견 + 목표가
        content_area = soup.find(id='contentarea')
        if content_area:
            # 투자의견
            opinion_td = content_area.find('td', string=re.compile('투자의견'))
            if not opinion_td:
                # 대체 탐색: 텍스트에 Buy/Hold/Sell 포함
                for td in content_area.find_all('td'):
                    t = td.get_text(strip=True)
                    if any(w in t for w in ['Buy', 'Hold', 'Sell', 'Outperform', 'Neutral', 'Strong Buy']):
                        result['opinion'] = t[:30]
                        break

            # 목표가
            for td in (content_area.find_all('td') if content_area else []):
                t = td.get_text(strip=True)
                if '목표가' in t:
                    result['target_price'] = t[:40]
                    break

            # 핵심 요약 (view_cnt td 내부 p 태그들)
            view_td = content_area.find('td', {'class': 'view_cnt'})
            if view_td:
                paras = view_td.find_all('p')
                summary_parts = []
                for p in paras:
                    txt = p.get_text(strip=True)
                    if txt and len(txt) > 15:
                        summary_parts.append(txt)
                    if len(summary_parts) >= 3:
                        break
                result['summary'] = ' | '.join(summary_parts)

            if not result['summary']:
                # fallback: div 직접 텍스트
                for div in content_area.find_all('div'):
                    t = div.get_text(strip=True)
                    if len(t) > 60 and '©' not in t and '네이버' not in t:
                        result['summary'] = t[:300]
                        break

    except Exception as e:
        print(f"Error: {e}")
    return result

# 테스트
test_reports = [
    ("삼성전기", "https://finance.naver.com/research/company_read.naver?nid=93234&page=1"),
    ("한올바이오파마", "https://finance.naver.com/research/company_read.naver?nid=93229&page=1"),
]

for name, url in test_reports:
    print(f"\n=== {name} ===")
    info = fetch_report_summary(url)
    print(f"  투자의견: {info['opinion']}")
    print(f"  목표가:   {info['target_price']}")
    print(f"  요약:     {info['summary'][:200]}")
