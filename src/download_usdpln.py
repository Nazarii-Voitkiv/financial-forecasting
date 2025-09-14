import argparse
import os

import pandas as pd
import yfinance as yf

def download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df = df.dropna().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Brak DatetimeIndex")
    df.index.name = "Date"
    return df

def main():
    parser = argparse.ArgumentParser(description="Pobierz dane USD/PLN z Yahoo Finance")
    parser.add_argument("--symbol", default="USDPLN=X")
    parser.add_argument("--period", default="max")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--out", default="data/raw/usdpln_yahoo_daily.csv")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = download(args.symbol, args.period, args.interval)
    df.to_csv(args.out)
    start = df.index.min().date()
    end = df.index.max().date()
    print(f"Zapisano {len(df)} wierszy dla {args.symbol} [{start}..{end}] -> {args.out}")

if __name__ == "__main__":
    main()
