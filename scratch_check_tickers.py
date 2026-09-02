import sys
import pandas as pd
import yfinance as yf

tickers = {
    "kospi": "^KS11", "sp500": "^GSPC", "fx": "KRW=X",
    "us10y": "^TNX", "us2y": "^IRX", "vix": "^VIX",
    "copper": "HG=F", "freight": "BDRY", "wti": "CL=F",
    "dxy": "DX-Y.NYB", "eem": "EEM"
}

for name, ticker in tickers.items():
    print(f"Downloading {name} ({ticker})...")
    df = yf.download(ticker, period="5d", progress=False)
    if df.empty:
        print(f"FAILED to download: {name} ({ticker})")
    else:
        print(f"Success: {name}")

