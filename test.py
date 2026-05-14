import yfinance as yf
import pandas as pd

try:
    df = yf.download('DX-Y.NYB', period='1mo')
    print("DXY head:")
    print(df.head())
    print("Empty?", df.empty)
except Exception as e:
    print("Error:", e)
