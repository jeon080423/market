import requests
from bs4 import BeautifulSoup
from collections import defaultdict

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_frequency():
    # We will fetch page 1 and page 2 to get 60 reports for a more reliable frequency check
    reports = []
    for page in [1, 2]:
        url = f"https://finance.naver.com/research/company_list.naver?page={page}"
        print(f"Fetching page {page}: {url}")
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
            
            table = soup.find('table', {'class': 'type_1'})
            if not table:
                print(f"Table of class 'type_1' not found on page {page}.")
                continue
                
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    stock_td = cols[0]
                    stock_a = stock_td.find('a')
                    stock_name = stock_a.get_text(strip=True) if stock_a else stock_td.get_text(strip=True)
                    
                    ticker = ""
                    if stock_a and 'code=' in stock_a.get('href', ''):
                        ticker = stock_a['href'].split('code=')[-1]
                    
                    title_td = cols[1]
                    title_a = title_td.find('a')
                    title = title_a.get_text(strip=True) if title_a else title_td.get_text(strip=True)
                    
                    brokerage = cols[2].get_text(strip=True)
                    date = cols[4].get_text(strip=True)
                    
                    if stock_name and title:
                        reports.append({
                            'stock_name': stock_name,
                            'ticker': ticker,
                            'title': title,
                            'brokerage': brokerage,
                            'date': date
                        })
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            
    # Now group by stock
    # We want: Stock Name -> count, latest report title, latest brokerage, latest date
    stock_groups = defaultdict(list)
    for rep in reports:
        stock_groups[rep['stock_name']].append(rep)
        
    freq_list = []
    for stock_name, reps in stock_groups.items():
        # Sort reports within the same stock by date (reps are already in descending order from naver, but just in case, reps[0] is the latest)
        latest_rep = reps[0]
        freq_list.append({
            'stock_name': stock_name,
            'ticker': latest_rep['ticker'],
            'count': len(reps),
            'latest_title': latest_rep['title'],
            'latest_brokerage': latest_rep['brokerage'],
            'latest_date': latest_rep['date']
        })
        
    # Sort by: 1. count (descending), 2. latest_date (descending)
    freq_list.sort(key=lambda x: (x['count'], x['latest_date']), reverse=True)
    
    with open("scratch/test_report_frequency.txt", "w", encoding="utf-8") as f:
        f.write(f"Total reports analyzed: {len(reports)}\n")
        f.write(f"Unique stocks mentioned: {len(freq_list)}\n\n")
        
        for idx, item in enumerate(freq_list[:20]):
            f.write(f"Rank {idx+1}: {item['stock_name']} ({item['ticker']}) - Mentioned {item['count']} times\n")
            f.write(f"  Latest Report: {item['latest_title']}\n")
            f.write(f"  Brokerage: {item['latest_brokerage']} | Date: {item['latest_date']}\n")
            f.write("-" * 50 + "\n")
            
    print("Done. Saved to scratch/test_report_frequency.txt")

if __name__ == "__main__":
    test_frequency()
