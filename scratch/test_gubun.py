import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_gubun():
    codes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9100, 9200]
    results = []
    for code in codes:
        url = f"https://finance.naver.com/sise/sise_deal_rank.naver?sosok=0&investor_gubun={code}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.raise_for_status()
            soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
            
            tables = soup.find_all('table')
            t0_rows = []
            if tables:
                rows = tables[0].find_all('tr')
                for r in rows[:3]:
                    cols = [td.get_text(strip=True) for td in r.find_all(['td', 'th'])]
                    if cols:
                        t0_rows.append(cols)
            results.append(f"investor_gubun={code} -> Table 0: {t0_rows}")
        except Exception as e:
            results.append(f"investor_gubun={code} -> Error: {e}")
            
    with open("scratch/test_gubun.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print("Done. Saved to scratch/test_gubun.txt")

if __name__ == "__main__":
    test_gubun()
