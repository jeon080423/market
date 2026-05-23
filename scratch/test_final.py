import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_crawl():
    url = "https://finance.naver.com/sise/sise_trans_style.naver?sosok=0"
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.content.decode('euc-kr', 'replace'), 'html.parser')
    
    tables = soup.find_all('table', {'class': 'type_r1'})
    print(f"Tables found: {len(tables)}")
    
    for i, table in enumerate(tables):
        label = "외국인 순매수" if i == 0 else "기관 순매수"
        rows = table.find_all('tr')
        print(f"\n=== Table {i}: {label} ({len(rows)} rows) ===")
        for j, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= 3:
                name_a = cols[1].find('a')
                if name_a:
                    name = name_a.get_text(strip=True)
                    href = name_a.get('href', '')
                    ticker = href.split('code=')[-1] if 'code=' in href else ''
                    price = cols[2].get_text(strip=True)
                    print(f"  {j+1}. {name} ({ticker}) - {price}")

if __name__ == "__main__":
    test_crawl()
