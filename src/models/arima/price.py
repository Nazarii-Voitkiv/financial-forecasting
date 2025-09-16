import os
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

DATA_RAW = "data/raw/usdpln_yahoo_daily.csv"
REPORT_PLOT_2025 = "data/reports/arima/price_2025.png"
METRICS_CSV = "data/reports/arima/metrics_price_2025.csv"
TRAIN_END = pd.Timestamp("2024-12-31")

# Konfiguracja
AUTO_AIC = False
ORDER = (1, 1, 1)
TOL_ABS = 0.01
TOL_PCT = 0.0

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
    s = s.asfreq("B").ffill().bfill()
    return s

def select_arima_order(series: pd.Series, p_range=(0,1,2), d_range=(0,1), q_range=(0,1,2)) -> Tuple[int,int,int]:
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    best_aic = float("inf")
    best = (1,1,1)
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p==0 and d==0 and q==0:
                    continue
                try:
                    m = ARIMA(series, order=(p,d,q))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        res = m.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best = (p,d,q)
                except Exception:
                    continue
    return best

def arima_forecast_values_2025(series: pd.Series, order=(1, 1, 1)) -> Tuple[pd.Series, pd.Series]:
    series = series.copy().asfreq("B").ffill().bfill()
    if series.index.min() > TRAIN_END:
        raise ValueError("Za mało historii: brak danych przed 2025")
    test_idx = series.index[series.index.year == 2025]
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
        import warnings as _w
        from statsmodels.tools.sm_exceptions import ConvergenceWarning as _CW
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            _w.simplefilter("ignore", _CW)
            fitted = model.fit()
        pred = fitted.forecast(steps=1).iloc[-1]
        preds.append(pred)
        remaining = int(round(100 * (1 - (i + 1) / n)))
        if last_print is None or remaining != last_print:
            print(f"Pozostało: {remaining}%")
            last_print = remaining
    y_pred = pd.Series(preds, index=test_idx, name="ARIMA_forecast")
    y_true = series.loc[test_idx]
    return y_pred, y_true

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/reports/arima", exist_ok=True)
    df = load_or_fetch()
    s = prepare_series(df)
    order = ORDER
    if AUTO_AIC:
        base_train = s.loc[:TRAIN_END]
        order = select_arima_order(base_train)
        print(f"Wybrany porządek (AIC): {order}")
    print(f"ARIMA{order} USD/PLN — wartość 2025")
    y_pred, y_true = arima_forecast_values_2025(s, order=order)
    if TOL_PCT > 0:
        tol = y_true.abs() * (TOL_PCT / 100.0)
        tol_desc = f"{TOL_PCT:.3g}%"
    else:
        tol = pd.Series(TOL_ABS, index=y_true.index)
        tol_desc = f"{TOL_ABS} PLN"
    correct = ((y_true - y_pred).abs() <= tol).astype(int)
    acc = float(correct.mean())
    pct = acc * 100
    print(f"Dokładność (tolerancja {tol_desc}) 2025: {pct:.2f}% (n={len(y_true)})")
    import pandas as _pd
    _pd.DataFrame([{"metric":"tolerance_accuracy","value":acc,"n":len(y_true),"order":str(order),"tolerance":tol_desc}]).to_csv(METRICS_CSV, index=False)
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label="Rzeczywiste 2025")
    plt.plot(y_pred.index, y_pred.values, label="Prognoza ARIMA (1-krok)")
    plt.title(f"USD/PLN — ARIMA{order} prognozy wartości 2025")
    plt.xlabel("Data")
    plt.ylabel("Zamknięcie")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_PLOT_2025, dpi=150)
    print(f"Wykres zapisano: {REPORT_PLOT_2025}")

if __name__ == "__main__":
    main()
