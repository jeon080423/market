import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_sosok_gubun():
    # sosok=0 (KOSPI), sosok=1 (KOSDAQ)
    # investor_gubun=1000 (Foreigner)
    results = []
    for sosok in [0, 1]:
        url = f"https://finance.naver.com/sise/sise_deal_rank.naver?sosok={sosok}&investor_gubun=1000"
        try:
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.raise_for_status()
            soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
            
            tables = soup.find_all('table')
            t0_rows = []
            if tables:
                rows = tables[0].find_all('tr')
                for r in rows:
                    cols = [td.get_text(strip=True) for td in r.find_all(['td', 'th'])]
                    links = [a['href'] for a in r.find_all('a')]
                    if cols:
                        t0_rows.append((cols, links))
            results.append(f"sosok={sosok} -> Table 0 rows:\n" + "\n".join([f"  {r}" for r in t0_rows]))
        except Exception as e:
            results.append(f"sosok={sosok} -> Error: {e}")
            
    with open("scratch/test_sosok_gubun.txt", "w", encoding="utf-8") as f:
        f.write("\n\n" + "="*50 + "\n\n".join(results))
    print("Done. Saved to scratch/test_sosok_gubun.txt")

if __name__ == "__main__":
    test_sosok_gubun()
