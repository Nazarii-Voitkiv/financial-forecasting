import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from ..common import evaluate_tolerance, load_or_fetch, prepare_close_series, regression_metrics
from ..features import FeatureConfig, make_features


REPORT_DIR = "data/reports/rf"
EVAL_YEAR = 2025
METRICS_CSV = os.path.join(REPORT_DIR, f"metrics_price_{EVAL_YEAR}.csv")


# Konfiguracja
N_EST = int(os.getenv("RF_N_EST", "200"))
MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "8"))
SEED = 42
WALK_FORWARD = bool(int(os.getenv("RF_WALK_FORWARD", "1")))
WALK_STEP = max(1, int(os.getenv("RF_WALK_STEP", "20")))
TOL_ABS = 0.01
TOL_PCT = 0.0
FEATURE_CONFIG = FeatureConfig(
    lag_returns=5, momentum_windows=(), sma_windows=(5,), volatility_windows=(), rsi_windows=()
)


def prepare_series(df: pd.DataFrame) -> pd.Series:
    return prepare_close_series(df, fill_method="interpolate")


def build_dataset(close: pd.Series, feature_config: Optional[FeatureConfig] = None) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(close, feature_config)
    y = np.log(close.shift(-1)) - np.log(close)
    df = X.copy()
    df["target"] = y
    df = df.dropna()
    return df.drop(columns=["target"]), df["target"]


def train_once_predict_2025(
    close: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> tuple[pd.Series, pd.Series, RandomForestRegressor]:
    train_mask = X.index.year < EVAL_YEAR
    test_mask = X.index.year == EVAL_YEAR
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_ret_pred = pd.Series(model.predict(X_test), index=X_test.index)
    y_pred = close.loc[X_test.index] * np.exp(y_ret_pred)
    y_true = close.shift(-1).loc[X_test.index]
    return y_pred, y_true, model


def walk_forward_predict_2025(
    close: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> tuple[pd.Series, pd.Series]:
    test_idx = X.index[X.index.year == EVAL_YEAR]
    preds = []
    n = len(test_idx)
    last_print = None
    model: Optional[RandomForestRegressor] = None
    for i, d in enumerate(test_idx):
        prev_date = X.index[X.index.get_loc(d) - 1]
        train_mask = X.index <= prev_date
        if model is None or i % WALK_STEP == 0:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X.loc[train_mask], y.loc[train_mask])
        ret_pred = float(model.predict(X.loc[[d]])[0])
        price_pred = float(close.loc[d]) * float(np.exp(ret_pred))
        preds.append(price_pred)
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
    plt.plot(y_pred.index, y_pred.values, label="Prognoza RF")
    plt.title(f"USD/PLN — Random Forest wartość {EVAL_YEAR} (Dokładność: {acc*100:.2f}%)")
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
    if WALK_FORWARD:
        y_pred, y_true = walk_forward_predict_2025(close, X, y, N_EST, MAX_DEPTH, SEED)
    else:
        y_pred, y_true, _ = train_once_predict_2025(close, X, y, N_EST, MAX_DEPTH, SEED)
    acc, desc = evaluate_tolerance(y_true, y_pred, TOL_ABS, TOL_PCT)
    print(f"Dokładność (tolerancja {desc}) {EVAL_YEAR}: {acc*100:.2f}% (n={len(y_true)})")
    metrics = regression_metrics(y_true, y_pred)
    # Add family/task metadata and write metrics
    pd.DataFrame(
        [
            {
                **metrics,
                "tolerance_accuracy": float(acc),
                "tolerance": desc,
                "n": int(len(y_true)),
                "mode": "walk" if WALK_FORWARD else "train_once",
                "year": int(EVAL_YEAR),
                "model": "rf_price",
                "family": "rf",
                "task": "price",
            }
        ]
    ).to_csv(METRICS_CSV, index=False)

    # Save feature importances for regressor
    try:
        imp_df = pd.DataFrame({"feature": X.columns, "importance": _.feature_importances_})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_csv = os.path.join(REPORT_DIR, f"feature_importance_{EVAL_YEAR-1}.csv")
        os.makedirs(os.path.dirname(imp_csv), exist_ok=True)
        imp_df.to_csv(imp_csv, index=False)
        print(f"Zapisano ważności cech: {imp_csv}")
    except Exception:
        # If model object not present (e.g., walk-forward mode without final model), skip
        pass
    plot_predictions(close, y_pred, y_true, acc, os.path.join(REPORT_DIR, f"price_{EVAL_YEAR}.png"))


if __name__ == "__main__":
    main()
