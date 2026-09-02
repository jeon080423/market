import re

with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

kospi_target = '''        kospi_data.columns = [tickers["kospi"]]
        if kospi_data.empty:
            raise ValueError("KOSPI data is empty")
    except Exception as e:
        raise ValueError(f"KOSPI download failed: {e}")'''

kospi_rep = '''        kospi_data.columns = [tickers["kospi"]]
        if kospi_data.empty:
            raise ValueError("KOSPI data is empty")
    except Exception as e:
        import FinanceDataReader as fdr
        try:
            fdr_ks = fdr.DataReader("KS11", start_date, end_date)
            if not fdr_ks.empty:
                kospi_data = fdr_ks[['Close']]
                kospi_data.columns = [tickers["kospi"]]
            else:
                raise ValueError("FDR KOSPI is empty")
        except Exception as e2:
            raise ValueError(f"KOSPI download failed: {e} | {e2}")'''

fx_target = '''        if data.empty:
            raise ValueError("other_tickers data is empty")
    except Exception as e:
        raise ValueError(f"other_tickers download failed: {e}")'''

fx_rep = '''        if data.empty:
            raise ValueError("other_tickers data is empty")
    except Exception as e:
        raise ValueError(f"other_tickers download failed: {e}")
        
    # FinanceDataReader fallback for FX if missing from data
    if tickers["fx"] not in data.columns or data[tickers["fx"]].dropna().empty:
        import FinanceDataReader as fdr
        try:
            fdr_fx = fdr.DataReader("USD/KRW", start_date, end_date)
            if not fdr_fx.empty:
                fdr_fx = fdr_fx[['Close']]
                fdr_fx.columns = [tickers["fx"]]
                if tickers["fx"] in data.columns:
                    data = data.drop(columns=[tickers["fx"]])
                data = data.join(fdr_fx, how='outer')
        except: pass'''

if kospi_target in code:
    code = code.replace(kospi_target, kospi_rep)
if fx_target in code:
    code = code.replace(fx_target, fx_rep)
    
with open("app.py", "w", encoding="utf-8") as f:
    f.write(code)
print("FDR fallback added")
