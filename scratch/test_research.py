import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_research():
    url = "https://finance.naver.com/research/company_list.naver"
    print(f"Fetching: {url}")
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        print("Page Title:", soup.title.string if soup.title else "No Title")
        
        tables = soup.find_all('table')
        print(f"Total tables: {len(tables)}")
        for i, t in enumerate(tables):
            cls = t.get('class')
            rows = t.find_all('tr')
            print(f"Table {i} (class={cls}, rows={len(rows)}):")
            # print first 5 rows
            for r_idx, r in enumerate(rows[:5]):
                cols = [td.get_text(strip=True) for td in r.find_all(['td', 'th'])]
                print(f"  Row {r_idx}: {cols}")
            print("-" * 50)
            
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_research()
