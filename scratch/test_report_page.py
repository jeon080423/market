import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 실제 리포트 페이지 열기
url = "https://finance.naver.com/research/company_read.naver?nid=93234&page=1"
r = requests.get(url, headers=HEADERS, timeout=10)
soup = BeautifulSoup(r.content.decode('euc-kr', 'replace'), 'html.parser')

print("=== Page Title ===")
title = soup.find('title')
print(title.get_text(strip=True) if title else 'No title')

print("\n=== All text blocks (class/id hints) ===")
# 주요 div들 출력
for tag in soup.find_all(['div', 'p', 'td'], limit=60):
    cls = tag.get('class', [])
    id_ = tag.get('id', '')
    text = tag.get_text(strip=True)
    if text and len(text) > 20:
        print(f"  <{tag.name} class={cls} id={id_}>: {text[:120]}")

# 저장
with open('scratch/test_report_page.txt', 'w', encoding='utf-8') as f:
    f.write(soup.get_text())
print("\n>> Full text saved to scratch/test_report_page.txt")
