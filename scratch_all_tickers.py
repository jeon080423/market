import pandas as pd
import yfinance as yf

tickers = {
    "kospi": "^KS11",
    "sp500": "^GSPC",
    "fx": "KRW=X",
    "us10y": "^TNX",
    "us2y": "^IRX",
    "vix": "^VIX",
    "copper": "HG=F",
    "freight": "BDRY",
    "wti": "CL=F",
    "dxy": "DX-Y.NYB",
    "eem": "EEM"
}
panic_tickers = {
    "gold": "GC=F",
    "jpy_krw": "JPYKRW=X",
    "usd_chf": "CHF=X",
    "vvix": "^VVIX"
}
all_t = list(tickers.values()) + list(panic_tickers.values())

data = yf.download(all_t, period="5d", progress=False)['Close']
for t in all_t:
    if t not in data.columns:
        print(f"MISSING COLUMN: {t}")
    elif data[t].dropna().empty:
        print(f"EMPTY DATA: {t}")
    else:
        print(f"Success: {t}")
