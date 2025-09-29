import os
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

from ..common import load_or_fetch, prepare_close_series

EVAL_YEAR = 2025
REPORT_PLOT_2025 = f"data/reports/arima/updown_{EVAL_YEAR}.png"
METRICS_CSV = f"data/reports/arima/metrics_updown_{EVAL_YEAR}.csv"
TRAIN_END = pd.Timestamp(f"{EVAL_YEAR-1}-12-31")

# Konfiguracja
AR_LAGS = 10

def prepare_series(df: pd.DataFrame) -> pd.Series:
    return prepare_close_series(df, fill_method="ffill")

def walk_forward_autoreg(series: pd.Series, lags: int) -> pd.Series:
    series = series.copy().asfreq("B").ffill().bfill()
    if series.index.min() > TRAIN_END:
        raise ValueError(f"Za mało historii: brak danych przed {EVAL_YEAR}")
    test_idx = series.index[series.index.year == EVAL_YEAR]
    if len(test_idx) == 0:
        raise ValueError(f"Brak danych na rok {EVAL_YEAR}")
    preds = []
    for date in test_idx:
        pos = series.index.get_loc(date)
        train_series = series.iloc[:pos]
        if len(train_series) <= lags:
            raise ValueError("Za mało danych do dopasowania AutoReg")
        model = AutoReg(train_series, lags=lags, old_names=False).fit()
        forecast = model.forecast(steps=1)
        preds.append(float(forecast.iloc[0]))
    return pd.Series(preds, index=test_idx, name="AutoReg_forecast")

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/reports/arima", exist_ok=True)
    df = load_or_fetch()
    s = prepare_series(df)
    print(f"AutoReg({AR_LAGS}) USD/PLN — kierunek {EVAL_YEAR}")
    forecast_2025 = walk_forward_autoreg(s, lags=AR_LAGS)
    prev_actual = s.shift(1).loc[forecast_2025.index]
    true_up = (s.loc[forecast_2025.index] > prev_actual)
    pred_up = forecast_2025 > prev_actual
    acc = float((pred_up == true_up).mean())
    pct = acc * 100
    print(f"Dokładność up/down {EVAL_YEAR}: {pct:.2f}% (n={len(forecast_2025)})")
    # metrics csv
    import pandas as _pd
    _pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "value": acc,
                "n": len(forecast_2025),
                "order": f"AR({AR_LAGS})",
                "year": int(EVAL_YEAR),
                "mode": "walk",
                "model": "autoreg_updown",
            }
        ]
    ).to_csv(METRICS_CSV, index=False)
    plt.figure(figsize=(12, 5))
    actual_2025 = s.loc[forecast_2025.index]
    plt.plot(actual_2025.index, actual_2025.values, label=f"Rzeczywiste {EVAL_YEAR}")
    plt.plot(forecast_2025.index, forecast_2025.values, label="Prognoza AutoReg (walk-forward)")
    plt.title(f"USD/PLN — AutoReg({AR_LAGS}) prognozy jednodniowe {EVAL_YEAR} (Dokładność: {acc*100:.2f}%)")
    ax = plt.gca()
    ax.set_xlabel("Data")
    ax.set_ylabel("Zamknięcie")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_PLOT_2025, dpi=150)
    print(f"Wykres zapisano: {REPORT_PLOT_2025}")

if __name__ == "__main__":
    main()
