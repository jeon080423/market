import yfinance as yf

def check_prices():
    tickers = ["005935.KS", "310970.KS", "196170.KQ", "247540.KQ", "034730.KS"]
    for t in tickers:
        try:
            ticker_data = yf.Ticker(t)
            history = ticker_data.history(period="1d")
            if not history.empty:
                print(f"Ticker {t}: Close Price = {history['Close'].iloc[-1]}")
            else:
                print(f"Ticker {t}: No history data")
        except Exception as e:
            print(f"Ticker {t}: Error: {e}")

if __name__ == "__main__":
    check_prices()
