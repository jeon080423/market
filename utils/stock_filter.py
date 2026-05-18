import os
import re
import pandas as pd
import streamlit as st
import FinanceDataReader as fdr

EN_STOCK_MAP = {
    "Samsung": "삼성전자",
    "SK Hynix": "SK하이닉스",
    "LG Energy": "LG에너지솔루션",
    "Hyundai Motor": "현대차",
    "Hyundai": "현대차",
    "Kia": "기아",
    "Kakao": "카카오",
    "Naver": "NAVER",
    "Celltrion": "셀트리온",
    "POSCO": "POSCO홀딩스",
    "KB Financial": "KB금융",
    "Shinhan": "신한지주",
    "Samsung C&T": "삼성물산",
    "Samsung Bio": "삼성바이오로직스",
    "Samsung SDI": "삼성SDI",
    "LG Chem": "LG화학",
    "LG Electronics": "LG전자",
    "Hana Financial": "하나금융지주",
    "SK Innovation": "SK이노베이션",
    "Korea Electric": "한국전력",
    "Ecopro": "에코프로",
    "Ecopro BM": "에코프로비엠",
    "L&F": "엘앤에프",
    "HLB": "HLB",
    "Alteogen": "알테오젠",
    "HPSP": "HPSP",
    "Rainbow Robotics": "레인보우로보틱스",
    "Doosan": "두산에너빌리티",
    "Samsung Electro-Mechanics": "삼성전기",
    "Samsung Heavy": "삼성중공업",
    "Korean Air": "대한항공",
    "Meritz": "메리츠금융지주",
    "HD Hyundai": "HD현대",
    "Hanwha Aerospace": "한화에어로스페이스",
    "Hanwha": "한화",
    "SK Telecom": "SK텔레콤",
    "KT": "KT",
    "S-Oil": "S-Oil",
    "Korea Zinc": "고려아연",
    "Coway": "코웨이",
    "Netmarble": "넷마블",
    "KOGAS": "한국가스공사",
    "Orion": "오리온",
    "Amorepacific": "아모레퍼시픽",
    "Yuhan": "유한양행"
}

# Regex global cache
_compiled_names_regex = None
_compiled_tickers_regex = None
_name_to_ticker_map = None

@st.cache_data(ttl=3600)
def load_krx_stocks() -> pd.DataFrame:
    """
    KOSPI/KOSDAQ 종목 목록 (KRX)
    - FinanceDataReader로 실시간 조회 후 CSV로 백업
    - 조회 실패 시 로컬 CSV 폴백
    - 반환: DataFrame [ticker, name, market, changes, chg_rate]
    """
    os.makedirs("data", exist_ok=True)
    fallback_path = os.path.join("data", "krx_stocks.csv")
    
    try:
        # Fetch live KOSPI & KOSDAQ listings
        kospi = fdr.StockListing("KOSPI")[["Code", "Name", "Changes", "ChgRate"]].assign(market="KOSPI")
        kosdaq = fdr.StockListing("KOSDAQ")[["Code", "Name", "Changes", "ChgRate"]].assign(market="KOSDAQ")
        
        df = pd.concat([kospi, kosdaq], ignore_index=True)
        df = df.rename(columns={
            "Code": "ticker",
            "Name": "name",
            "Changes": "changes",
            "ChgRate": "chg_rate"
        })
        
        # Clean numerical columns
        df["changes"] = pd.to_numeric(df["changes"], errors="coerce").fillna(0)
        df["chg_rate"] = pd.to_numeric(df["chg_rate"], errors="coerce").fillna(0.0)
        
        # Backup to CSV
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        return df
    except Exception as e:
        # Absolute minimal fallback (Top 50 major stocks to ensure wide coverage)
        minimal_stocks = [
            {"ticker": "005930", "name": "삼성전자", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "000660", "name": "SK하이닉스", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "373220", "name": "LG에너지솔루션", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "207940", "name": "삼성바이오로직스", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "005380", "name": "현대차", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "000270", "name": "기아", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "068270", "name": "셀트리온", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "005490", "name": "POSCO홀딩스", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "035420", "name": "NAVER", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "035720", "name": "카카오", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "006400", "name": "삼성SDI", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "051910", "name": "LG화학", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "012330", "name": "현대모비스", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "055550", "name": "신한지주", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "105560", "name": "KB금융", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "086790", "name": "하나금융지주", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "028260", "name": "삼성물산", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "066570", "name": "LG전자", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "096770", "name": "SK이노베이션", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "015760", "name": "한국전력", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "003550", "name": "LG", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "034730", "name": "SK", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "009150", "name": "삼성전기", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "010140", "name": "삼성중공업", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "003490", "name": "대한항공", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "138040", "name": "메리츠금융지주", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "086520", "name": "에코프로", "market": "KOSDAQ", "changes": 0, "chg_rate": 0.0},
            {"ticker": "247540", "name": "에코프로비엠", "market": "KOSDAQ", "changes": 0, "chg_rate": 0.0},
            {"ticker": "066970", "name": "엘앤에프", "market": "KOSDAQ", "changes": 0, "chg_rate": 0.0},
            {"ticker": "028300", "name": "HLB", "market": "KOSDAQ", "changes": 0, "chg_rate": 0.0},
            {"ticker": "196170", "name": "알테오젠", "market": "KOSDAQ", "changes": 0, "chg_rate": 0.0},
            {"ticker": "403870", "name": "HPSP", "market": "KOSDAQ", "changes": 0, "chg_rate": 0.0},
            {"ticker": "387320", "name": "레인보우로보틱스", "market": "KOSDAQ", "changes": 0, "chg_rate": 0.0},
            {"ticker": "034020", "name": "두산에너빌리티", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "307900", "name": "한화에어로스페이스", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "017670", "name": "SK텔레콤", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "030200", "name": "KT", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "010950", "name": "S-Oil", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "010130", "name": "고려아연", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "000810", "name": "삼성화재", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "032830", "name": "삼성생명", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "003670", "name": "포스코퓨처엠", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "090430", "name": "아모레퍼시픽", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "000100", "name": "유한양행", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "036570", "name": "엔씨소프트", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "251270", "name": "넷마블", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "005830", "name": "DB손해보험", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "000720", "name": "현대건설", "market": "KOSPI", "changes": 0, "chg_rate": 0.0},
            {"ticker": "008770", "name": "호텔신라", "market": "KOSPI", "changes": 0, "chg_rate": 0.0}
        ]
        
        # 실시간 야후 파이낸스 주가 업데이트 연동 (상승/하락 분리 고도화)
        try:
            import yfinance as yf
            tickers_list = []
            for s in minimal_stocks:
                suffix = ".KS" if s["market"] == "KOSPI" else ".KQ"
                tickers_list.append(f"{s['ticker']}{suffix}")
            
            # 5일 가격 데이터 초고속 배치 수집
            tickers_str = " ".join(tickers_list)
            data = yf.download(tickers_str, period="5d", interval="1d", progress=False)
            
            if "Close" in data:
                close_data = data["Close"]
                for s in minimal_stocks:
                    suffix = ".KS" if s["market"] == "KOSPI" else ".KQ"
                    full_ticker = f"{s['ticker']}{suffix}"
                    if full_ticker in close_data:
                        series = close_data[full_ticker].dropna()
                        if len(series) >= 2:
                            prev_price = float(series.iloc[-2])
                            curr_price = float(series.iloc[-1])
                            if prev_price > 0:
                                s["changes"] = curr_price - prev_price
                                s["chg_rate"] = ((curr_price - prev_price) / prev_price) * 100
        except Exception:
            pass
            
        return pd.DataFrame(minimal_stocks)

def filter_by_price_direction(df: pd.DataFrame, direction: str = "up") -> pd.DataFrame:
    """
    - direction: "up" (상승) or "down" (하락)
    - 상승: chg_rate > 0
    - 하락: chg_rate < 0
    """
    # Safeguard copying
    df_filtered = df.copy()
    df_filtered["chg_rate"] = pd.to_numeric(df_filtered["chg_rate"], errors="coerce").fillna(0.0)
    
    if direction == "up":
        filtered = df_filtered[df_filtered["chg_rate"] > 0]
        if filtered.empty:
            return df_filtered  # 실시간 연결 실패 시 변동률이 0이므로, 전체 목록을 폴백으로 유지
        return filtered
    elif direction == "down":
        filtered = df_filtered[df_filtered["chg_rate"] < 0]
        if filtered.empty:
            return df_filtered  # 실시간 연결 실패 시 변동률이 0이므로, 전체 목록을 폴백으로 유지
        return filtered
    return df_filtered

def init_compiled_patterns(stock_list: list[dict]):
    """
    Compiles and caches the name and ticker patterns for fast matching.
    """
    global _compiled_names_regex, _compiled_tickers_regex, _name_to_ticker_map
    if _compiled_names_regex is not None:
        return

    name_to_ticker = {}
    
    for stock in stock_list:
        ticker = stock.get("ticker")
        name = stock.get("name")
        if not ticker or not name:
            continue
            
        # Lowercase mapping for case-insensitive lookup
        name_to_ticker[name.lower()] = ticker
        name_to_ticker[ticker] = ticker
        
        # Add English mapping if exists
        for eng, kor in EN_STOCK_MAP.items():
            if kor.lower() == name.lower():
                name_to_ticker[eng.lower()] = ticker

    # Sort names by length descending to prevent partial match issues (e.g. '삼성전자우' matches '삼성전자' first if not sorted)
    sorted_names = sorted(name_to_ticker.keys(), key=len, reverse=True)
    
    # 1. Names pattern: matches Korean names or English aliases. Case-insensitive.
    # Exclude pure number tickers from names pattern
    name_keys = [k for k in sorted_names if not k.isdigit()]
    _compiled_names_regex = re.compile("|".join(re.escape(k) for k in name_keys), re.IGNORECASE)
    
    # 2. Tickers pattern: matches 6-digit tickers strictly with boundary conditions.
    ticker_keys = [k for k in sorted_names if k.isdigit()]
    _compiled_tickers_regex = re.compile(r'(?<!\d)(' + "|".join(re.escape(t) for t in ticker_keys) + r')(?!\d)')
    
    _name_to_ticker_map = name_to_ticker

def count_stock_mentions(text: str, stock_list: list[dict]) -> dict[str, int]:
    """
    - stock_list: [{"name": "삼성전자", "ticker": "005930", "market": "KOSPI"}, ...]
    - 종목명 및 티커를 정규식으로 검색 (단어 경계 처리)
    - 영문 채널용: 영문 종목명도 매핑 테이블로 추가 (예: "Samsung" → "삼성전자")
    - 반환: {"005930": 3, "000660": 1, ...}
    """
    if not text:
        return {}

    # Initialize compiled patterns
    init_compiled_patterns(stock_list)
    
    mentions = {}
    
    # Find all stock names/aliases matches
    if _compiled_names_regex:
        name_matches = _compiled_names_regex.findall(text)
        for m in name_matches:
            ticker = _name_to_ticker_map.get(m.lower())
            if ticker:
                mentions[ticker] = mentions.get(ticker, 0) + 1

    # Find all ticker number matches
    if _compiled_tickers_regex:
        ticker_matches = _compiled_tickers_regex.findall(text)
        for m in ticker_matches:
            ticker = _name_to_ticker_map.get(m)
            if ticker:
                mentions[ticker] = mentions.get(ticker, 0) + 1

    return mentions
