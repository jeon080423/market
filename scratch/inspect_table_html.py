import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def inspect_table():
    url = "https://finance.naver.com/sise/sise_deal_rank.naver?investor_gubun=1000"
    try:
        res = requests.get(url, headers=HEADERS, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        tables = soup.find_all('table')
        if tables:
            print("Table 0 HTML:")
            print(tables[0].prettify()[:3000])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    inspect_table()
