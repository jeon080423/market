import sys
import pandas as pd
import yfinance as yf

# ks_s calculation logic from app.py
def get_data(ticker, period="3y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        return df['Close'].dropna()
    except:
        return pd.Series()

ks_s = get_data("^KS11")
print("ks_s tail:\n", ks_s.tail())

if not ks_s.empty:
    ma20 = ks_s.rolling(window=20).mean()
    print("ma20 tail:\n", ma20.tail())
    
    # check the t calculation for the last element
    d = ks_s.index[-1]
    val_ks = float(ks_s.loc[d].iloc[0] if isinstance(ks_s.loc[d], pd.Series) else ks_s.loc[d])
    val_ma = float(ma20.loc[d].iloc[0] if isinstance(ma20.loc[d], pd.Series) else ma20.loc[d])
    
    print("last ks:", val_ks)
    print("last ma20:", val_ma)
    
    t = max(0.0, min(100.0, float(100 - (val_ks / val_ma - 0.9) * 500))) if d in ma20.index and pd.notna(val_ma) and val_ma != 0 else 50.0
    print("t:", t)
