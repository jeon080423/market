import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_research_parse():
    url = "https://finance.naver.com/research/company_list.naver"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content.decode('euc-kr', 'replace'), 'html.parser')
        
        table = soup.find('table', {'class': 'type_1'})
        if not table:
            print("Table of class 'type_1' not found.")
            return
            
        rows = table.find_all('tr')
        parsed_rows = []
        for i, r in enumerate(rows):
            cols = r.find_all('td')
            # The columns in company_list are usually:
            # 0: Stock name (a link or text)
            # 1: Title (a link)
            # 2: Brokerage
            # 3: Attachment (PDF icon)
            # 4: Date
            # 5: Views
            if len(cols) >= 5:
                # Stock name might be inside an 'a' tag or directly in td
                stock_name_td = cols[0]
                stock_a = stock_name_td.find('a')
                stock_name = stock_a.get_text(strip=True) if stock_a else stock_name_td.get_text(strip=True)
                
                # We can extract the stock ticker from the stock link href
                ticker = ""
                if stock_a and 'code=' in stock_a.get('href', ''):
                    ticker = stock_a['href'].split('code=')[-1]
                
                title_td = cols[1]
                title_a = title_td.find('a')
                title = title_a.get_text(strip=True) if title_a else title_td.get_text(strip=True)
                
                # Report PDF link
                pdf_link = ""
                if title_a:
                    pdf_link = title_a.get('href', '')
                    
                brokerage = cols[2].get_text(strip=True)
                date = cols[4].get_text(strip=True)
                views = cols[5].get_text(strip=True) if len(cols) > 5 else ""
                
                if stock_name and title:
                    parsed_rows.append({
                        "stock_name": stock_name,
                        "ticker": ticker,
                        "title": title,
                        "brokerage": brokerage,
                        "date": date,
                        "views": views
                    })
                    
        with open("scratch/test_research_parse.txt", "w", encoding="utf-8") as f:
            f.write(f"Total reports parsed: {len(parsed_rows)}\n\n")
            for idx, r in enumerate(parsed_rows):
                f.write(f"Report {idx+1}:\n")
                f.write(f"  Stock: {r['stock_name']} ({r['ticker']})\n")
                f.write(f"  Title: {r['title']}\n")
                f.write(f"  Brokerage: {r['brokerage']}\n")
                f.write(f"  Date: {r['date']}\n")
                f.write(f"  Views: {r['views']}\n")
                f.write("-" * 30 + "\n")
        print("Done. Saved to scratch/test_research_parse.txt")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_research_parse()
