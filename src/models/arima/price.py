import os
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

from ..common import evaluate_tolerance, load_or_fetch, prepare_close_series, regression_metrics

EVAL_YEAR = 2025
REPORT_PLOT_2025 = f"data/reports/arima/price_{EVAL_YEAR}.png"
METRICS_CSV = f"data/reports/arima/metrics_price_{EVAL_YEAR}.csv"
TRAIN_END = pd.Timestamp(f"{EVAL_YEAR-1}-12-31")

# Konfiguracja
AR_LAGS = 10
TOL_ABS = 0.01
TOL_PCT = 0.0


def prepare_series(df: pd.DataFrame) -> pd.Series:
    return prepare_close_series(df, fill_method="ffill")


def walk_forward_autoreg(series: pd.Series, lags: int) -> Tuple[pd.Series, pd.Series]:
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

    y_pred = pd.Series(preds, index=test_idx, name="AutoReg_forecast")
    y_true = series.loc[test_idx]
    return y_pred, y_true

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/reports/arima", exist_ok=True)
    df = load_or_fetch()
    s = prepare_series(df)
    print(f"AutoReg({AR_LAGS}) USD/PLN — wartość {EVAL_YEAR}")
    y_pred, y_true = walk_forward_autoreg(s, lags=AR_LAGS)
    acc, tol_desc = evaluate_tolerance(y_true, y_pred, TOL_ABS, TOL_PCT)
    pct = acc * 100
    print(f"Dokładność (tolerancja {tol_desc}) {EVAL_YEAR}: {pct:.2f}% (n={len(y_true)})")
    import pandas as _pd
    metrics = regression_metrics(y_true, y_pred)
    _pd.DataFrame(
        [
            {
                **metrics,
                "tolerance_accuracy": acc,
                "tolerance": tol_desc,
                "n": len(y_true),
                "order": f"AR({AR_LAGS})",
                "year": int(EVAL_YEAR),
                "mode": "walk",
                "model": "autoreg_price",
            }
        ]
    ).to_csv(METRICS_CSV, index=False)
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label=f"Rzeczywiste {EVAL_YEAR}")
    plt.plot(y_pred.index, y_pred.values, label="Prognoza AutoReg (walk-forward)")
    plt.title(f"USD/PLN — AutoReg({AR_LAGS}) walk-forward {EVAL_YEAR} (Dokładność: {acc*100:.2f}%)")
    ax = plt.gca()
    ax.set_xlabel("Data")
    ax.set_ylabel("Zamknięcie")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_PLOT_2025, dpi=150)
    print(f"Wykres zapisano: {REPORT_PLOT_2025}")

if __name__ == "__main__":
    main()
