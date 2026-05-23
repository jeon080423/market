import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

url = "https://finance.naver.com/research/company_list.naver?page=1"
r = requests.get(url, headers=HEADERS, timeout=10)
soup = BeautifulSoup(r.content.decode('euc-kr', 'replace'), 'html.parser')

table = soup.find('table', {'class': 'type_1'})
rows = table.find_all('tr')

for i, row in enumerate(rows[:5]):
    cols = row.find_all('td')
    if len(cols) >= 5:
        stock_a = cols[0].find('a')
        title_a = cols[1].find('a')
        if stock_a and title_a:
            stock_name = stock_a.get_text(strip=True)
            title = title_a.get_text(strip=True)
            title_href = title_a.get('href', '')
            print(f"Stock: {stock_name}")
            print(f"Title: {title}")
            print(f"Title href: {title_href}")
            # PDF link (if any)
            pdf_a = cols[3].find('a') if len(cols) > 3 else None
            if pdf_a:
                print(f"PDF href: {pdf_a.get('href','')}")
            print("---")
