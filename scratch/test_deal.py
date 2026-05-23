import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_sosok(sosok):
    url = f"https://finance.naver.com/sise/sise_trans_style.naver?sosok={sosok}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        print(f"--- sosok={sosok} ---")
        
        tables = soup.find_all('table')
        for i, t in enumerate(tables):
            cls = t.get('class')
            rows = t.find_all('tr')
            print(f"Table {i} (class={cls}, rows={len(rows)}):")
            
            # Print sibling headings to know what the table is
            sibling = t.find_previous()
            h_text = ""
            while sibling:
                if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']:
                    h_text = sibling.get_text(strip=True)
                    if h_text and '순매수' in h_text:
                        break
                sibling = sibling.find_previous()
            print(f"  Header/Context: {h_text[:60]}")
            
            for r_idx, r in enumerate(rows[:5]):
                cols = [td.get_text(strip=True) for td in r.find_all(['td', 'th'])]
                print(f"    Row {r_idx}: {cols}")
    except Exception as e:
        print(f"Error for sosok={sosok}: {e}")

if __name__ == "__main__":
    test_sosok(0)
    test_sosok(1)
