import requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def inspect_kosdaq():
    url = "https://finance.naver.com/sise/sise_deal_rank.naver?sosok=1"
    try:
        res = requests.get(url, headers=HEADERS, timeout=5)
        res.raise_for_status()
        text = res.content.decode('euc-kr', 'replace')
        print(f"Status Code: {res.status_code}")
        print(f"HTML Length: {len(text)}")
        print("First 2000 characters of HTML:")
        print(text[:2000])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    inspect_kosdaq()
