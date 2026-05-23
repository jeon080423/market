import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_codes():
    codes = [1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    results = []
    for code in codes:
        url = f"https://finance.naver.com/sise/sise_deal_rank.naver?sosok=0&investor_code={code}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.raise_for_status()
            soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
            
            tables = soup.find_all('table')
            t_names = []
            for t in tables:
                sibling = t.find_previous()
                while sibling:
                    if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']:
                        txt = sibling.get_text(strip=True)
                        if txt:
                            t_names.append(txt[:100])
                            break
                    sibling = sibling.find_previous()
            results.append(f"Code {code}: URL: {url} -> Tables headers: {t_names}")
        except Exception as e:
            results.append(f"Code {code}: Error: {e}")
            
    with open("scratch/test_codes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print("Done. Saved to scratch/test_codes.txt")

if __name__ == "__main__":
    test_codes()
