import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
except Exception as _e:
    import sys
    import platform

    print("XGBoost nie jest gotowy (brak biblioteki libomp/xgboost).")
    print("Na macOS zainstaluj:")
    print("  brew install libomp")
    print("potem (w wirtualnym środowisku):")
    print("  pip install --upgrade xgboost")
    if platform.system() == "Darwin":
        print("Jeśli nadal błąd, ustaw ścieżkę bibliotek (Apple Silicon):")
        print("  export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH")
        print("lub (Intel):")
        print("  export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH")
    sys.exit(1)

from ..common import evaluate_tolerance, load_or_fetch, prepare_close_series, regression_metrics
from ..features import FeatureConfig, make_features


REPORT_DIR = "data/reports/xgb"
EVAL_YEAR = 2025
METRICS_CSV = os.path.join(REPORT_DIR, f"metrics_price_{EVAL_YEAR}.csv")
TRAIN_END = pd.Timestamp(f"{EVAL_YEAR-1}-12-31")


# Konfiguracja
N_EST = int(os.getenv("XGB_N_EST", "1200"))
MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", "4"))
ETA = float(os.getenv("XGB_ETA", "0.03"))
SUBSAMPLE = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
COLSAMPLE = float(os.getenv("XGB_COLSAMPLE", "0.6"))
REG_LAMBDA = float(os.getenv("XGB_REG_LAMBDA", "2.0"))
SEED = 42
WALK_FORWARD = bool(int(os.getenv("XGB_WALK_FORWARD", "1")))
WALK_STEP = int(os.getenv("XGB_WALK_STEP", "20"))
TOL_ABS = 0.01
TOL_PCT = 0.0
FEATURE_CONFIG = FeatureConfig()


def prepare_series(df: pd.DataFrame) -> pd.Series:
    return prepare_close_series(df, fill_method="ffill")


def build_dataset(close: pd.Series, feature_config: Optional[FeatureConfig] = None) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(close, feature_config)
    y = np.log(close.shift(-1)) - np.log(close)
    df = X.copy()
    df["target"] = y
    df = df.dropna()
    return df.drop(columns=["target"]), df["target"]


def _xgb_regressor(params: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        objective="reg:squarederror",
        tree_method=params.get("tree_method", "hist"),
        random_state=params.get("seed", 42),
        n_jobs=-1,
    )


def train_once_predict_2025(
    close: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
):
    train_mask = X.index.year < EVAL_YEAR
    test_mask = X.index.year == EVAL_YEAR
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test = X.loc[test_mask]

    model = _xgb_regressor(params)
    model.fit(X_train, y_train)

    ret_pred = pd.Series(model.predict(X_test), index=X_test.index)
    y_pred = close.loc[X_test.index] * np.exp(ret_pred)
    y_true = close.shift(-1).loc[X_test.index]
    return y_pred, y_true, model


def walk_forward_predict_2025(close, X, y, params):
    test_idx = X.index[X.index.year == EVAL_YEAR]
    preds = []
    n = len(test_idx)
    last_print = None
    model = None

    for i, d in enumerate(test_idx):
        if (model is None) or (i % WALK_STEP == 0):
            prev_date = X.index[X.index.get_loc(d) - 1]
            mask = X.index <= prev_date
            Xt, yt = X.loc[mask], y.loc[mask]
            model = _xgb_regressor(params)
            model.fit(Xt, yt)

        r = float(model.predict(X.loc[[d]])[0])
        preds.append(float(close.loc[d]) * float(np.exp(r)))

        remaining = int(round(100 * (1 - (i + 1) / n)))
        if last_print is None or remaining != last_print:
            print(f"Pozostało: {remaining}%")
            last_print = remaining

    y_pred = pd.Series(preds, index=test_idx)
    y_true = close.shift(-1).loc[test_idx]
    return y_pred, y_true


def plot_predictions(close: pd.Series, y_pred: pd.Series, y_true: pd.Series, acc: float, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label=f"Rzeczywiste {EVAL_YEAR}")
    plt.plot(y_pred.index, y_pred.values, label="Prognoza XGB")
    plt.title(f"USD/PLN — XGBoost wartość {EVAL_YEAR} (Dokładność: {acc*100:.2f}%)")
    ax = plt.gca()
    ax.set_xlabel("Data")
    ax.set_ylabel("Zamknięcie")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wykres zapisano: {out_png}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_or_fetch()
    close = prepare_series(df)
    X, y = build_dataset(close, FEATURE_CONFIG)

    params = dict(
        n_estimators=N_EST,
        max_depth=MAX_DEPTH,
        learning_rate=ETA,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        reg_lambda=REG_LAMBDA,
        seed=SEED,
    )

    if WALK_FORWARD:
        y_pred, y_true = walk_forward_predict_2025(close, X, y, params)
    else:
        y_pred, y_true, _ = train_once_predict_2025(close, X, y, params)

    acc, desc = evaluate_tolerance(y_true, y_pred, TOL_ABS, TOL_PCT)
    print(f"Dokładność (tolerancja {desc}) {EVAL_YEAR}: {acc*100:.2f}% (n={len(y_true)})")
    metrics = regression_metrics(y_true, y_pred)
    pd.DataFrame(
        [
            {
                **metrics,
                "tolerance_accuracy": acc,
                "tolerance": desc,
                "n": int(len(y_true)),
                "mode": "walk" if WALK_FORWARD else "train_once",
                "year": int(EVAL_YEAR),
                "model": "xgb_price",
                "family": "xgb",
                "task": "price",
            }
        ]
    ).to_csv(METRICS_CSV, index=False)
    plot_predictions(close, y_pred, y_true, acc, os.path.join(REPORT_DIR, f"price_{EVAL_YEAR}.png"))


if __name__ == "__main__":
    main()