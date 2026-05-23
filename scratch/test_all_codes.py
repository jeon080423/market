import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_all_codes():
    # Codes to test
    # Naver investor codes:
    # 9000: Foreigner? 3000: Institution? Let's check.
    codes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    
    results = []
    for code in codes:
        url = f"https://finance.naver.com/sise/sise_deal_rank.naver?sosok=0&investor_code={code}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.raise_for_status()
            text = res.content.decode('euc-kr', 'replace')
            soup = BeautifulSoup(text, 'html.parser')
            
            # Find all text and see if "외국인" or "기관" is in it
            has_foreigner = "외국인" in text
            has_institution = "기관" in text
            has_individual = "개인" in text
            
            # Let's find headings or list names
            headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            
            # Find divs with class or id that might contain the header
            divs = [d.get_text(strip=True) for d in soup.find_all('div') if '순매수' in d.get_text() or '매매' in d.get_text()]
            div_snippet = divs[0][:150] if divs else "No matching div"
            
            results.append(f"Code {code}: foreigner={has_foreigner}, institution={has_institution}, individual={has_individual}\n"
                           f"  Headings: {headings}\n"
                           f"  Div Snippet: {div_snippet}\n")
        except Exception as e:
            results.append(f"Code {code}: Error: {e}\n")
            
    with open("scratch/test_all_codes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print("Saved results to scratch/test_all_codes.txt")

if __name__ == "__main__":
    test_all_codes()
