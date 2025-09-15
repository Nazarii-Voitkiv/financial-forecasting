import os
import pandas as pd

DATA_RAW = "data/raw/usdpln_yahoo_daily.csv"

def load_or_fetch():
    import yfinance as yf
    if os.path.exists(DATA_RAW):
        try:
            df = pd.read_csv(DATA_RAW, parse_dates=["Date"], index_col="Date")
        except Exception:
            try:
                tmp = pd.read_csv(DATA_RAW, header=2)
                if "Date" in tmp.columns:
                    tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
                    tmp = tmp.dropna(subset=["Date"]).set_index("Date").sort_index()
                    if "Close" not in tmp.columns:
                        if "Unnamed: 3" in tmp.columns:
                            tmp = tmp.rename(columns={"Unnamed: 3": "Close"})
                        elif "Adj Close" in tmp.columns:
                            tmp["Close"] = tmp["Adj Close"]
                    df = tmp
                else:
                    df = None
            except Exception:
                df = None
        if df is not None:
            return df
    df = yf.download("USDPLN=X", interval="1d", auto_adjust=False, progress=False).dropna().sort_index()
    df.index.name = "Date"
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    df.to_csv(DATA_RAW)
    return df

def prepare_series(df: pd.DataFrame) -> pd.Series:
    s = df["Close"].astype(float).sort_index()
    s = s.asfreq("B").interpolate()
    return s

