import os
from typing import Tuple

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

DATA_RAW = "data/raw/usdpln_yahoo_daily.csv"
REPORT_PLOT_2025 = "data/reports/arima_usdpln_updown_2025.png"

def fetch_usdpln() -> pd.DataFrame:
    df = yf.download("USDPLN=X", interval="1d", auto_adjust=False, progress=False)
    df = df.dropna().sort_index()
    df.index.name = "Date"
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    df.to_csv(DATA_RAW)
    return df

def load_or_fetch() -> pd.DataFrame:
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
                    df = fetch_usdpln()
            except Exception:
                df = fetch_usdpln()
    else:
        df = fetch_usdpln()
    return df

def prepare_series(df: pd.DataFrame) -> pd.Series:
    s = df["Close"].astype(float).sort_index()
    s = s.asfreq("B").interpolate()
    return s

def arima_predict_direction_2025(series: pd.Series, order=(1, 1, 1)) -> Tuple[pd.Series, float]:
    series = series.copy().asfreq("B").interpolate()
    train_end = pd.Timestamp("2024-12-31")
    if series.index.min() > train_end:
        raise ValueError("Za mało historii: brak danych przed 2025")
    test_year = 2025
    test_idx = series.index[series.index.year == test_year]
    if len(test_idx) == 0:
        raise ValueError("Brak danych na rok 2025")
    preds = []
    n = len(test_idx)
    last_print = None
    for i, d in enumerate(test_idx):
        prev_idx_pos = series.index.get_loc(d) - 1
        if prev_idx_pos < 0:
            continue
        prev_date = series.index[prev_idx_pos]
        train_series = series.loc[:prev_date]
        model = ARIMA(train_series, order=order)
        fitted = model.fit()
        pred = fitted.forecast(steps=1).iloc[-1]
        preds.append(pred)
        remaining = int(round(100 * (1 - (i + 1) / n)))
        if last_print is None or remaining != last_print:
            print(f"Pozostało: {remaining}%")
            last_print = remaining
    y_pred = pd.Series(preds, index=test_idx, name="ARIMA_forecast")
    prev_actual = series.shift(1).loc[test_idx]
    true_up = series.loc[test_idx] > prev_actual
    pred_up = y_pred > prev_actual
    accuracy = float((pred_up == true_up).mean())
    return y_pred, accuracy

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/reports", exist_ok=True)
    df = load_or_fetch()
    s = prepare_series(df)
    print("ARIMA(1,1,1) USD/PLN — kierunek 2025")
    forecast_2025, acc = arima_predict_direction_2025(s, order=(1, 1, 1))
    pct = acc * 100
    print(f"Dokładność up/down 2025: {pct:.2f}% (n={len(forecast_2025)})")
    plt.figure(figsize=(12, 5))
    actual_2025 = s.loc[forecast_2025.index]
    plt.plot(actual_2025.index, actual_2025.values, label="Rzeczywiste 2025")
    plt.plot(forecast_2025.index, forecast_2025.values, label="Prognoza ARIMA (1-krok)")
    plt.title("USD/PLN — ARIMA(1,1,1) prognozy jednodniowe 2025")
    plt.xlabel("Data")
    plt.ylabel("Zamknięcie")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_PLOT_2025, dpi=150)
    print(f"Wykres zapisano: {REPORT_PLOT_2025}")

if __name__ == "__main__":
    main()
