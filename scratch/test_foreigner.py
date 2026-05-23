import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_foreigner_crawl(sosok=0):
    url = f"https://finance.naver.com/sise/sise_deal_rank.naver?sosok={sosok}&investor_code=9000"
    print(f"Fetching URL: {url}")
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        # Print tables to see class names
        tables = soup.find_all('table')
        for i, table in enumerate(tables):
            cls = table.get('class')
            print(f"Table {i}: class={cls}")
            
        # The main ranking table usually has class 'type_2' or similar.
        # Let's inspect rows from table of class 'type_2' or matching table
        target_table = soup.find('table', {'class': 'type_2'})
        if not target_table:
            print("Table with class 'type_2' not found, searching other tables...")
            # Fallback to look at all tables
            for i, table in enumerate(tables):
                rows = table.find_all('tr')
                if len(rows) > 5:
                    print(f"Table {i} has {len(rows)} rows. Sample headers:")
                    headers = [th.text.strip() for th in table.find_all('th')]
                    print(headers)
                    # Print first data row
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 5:
                            print([col.get_text(strip=True) for col in cols])
                            break
            return
            
        rows = target_table.find_all('tr')
        print(f"Found {len(rows)} rows in type_2 table.")
        count = 0
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 8:
                # Let's see what columns are in sise_deal_rank
                txts = [c.get_text(strip=True) for c in cols]
                print(txts)
                count += 1
                if count >= 10:
                    break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("--- KOSPI FOREIGNER NET BUY (sosok=0) ---")
    test_foreigner_crawl(0)
    print("--- KOSDAQ FOREIGNER NET BUY (sosok=1) ---")
    test_foreigner_crawl(1)
