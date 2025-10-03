import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    from xgboost.callback import EarlyStopping
except ImportError:
    print("XGBoost не встановлено або виникла помилка імпорту. Встановіть його: pip install xgboost")
    exit(1)

from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from ..common import evaluate_tolerance, load_or_fetch, prepare_close_series, regression_metrics
from ..features import FeatureConfig, make_features

# --- Константи ---
FAMILY = "xgb"
REPORT_DIR = f"data/reports/{FAMILY}"
EVAL_YEAR = 2025
WALK_FORWARD = True
WALK_STEP = 20
TOL_ABS = 0.01
TOL_PCT = 0.0
SEED = 42
AR_LAGS = 10
OOF_SPLITS = 5
MIN_TRAIN_SAMPLES = 250
MIN_VAL_SAMPLES = 30
AR_MIN_HISTORY = 60

FEATURE_CONFIG = FeatureConfig(
    lag_returns=5,
    momentum_windows=(),
    sma_windows=(5, 20),
    volatility_windows=(),
    rsi_windows=(14,)
)

PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 2.0,
    "n_jobs": -1,
    "random_state": SEED,
}

EARLY_STOPPING_ROUNDS = 200
VAL_DAYS = 120


def prepare_series(df: pd.DataFrame) -> pd.Series:
    """Готує часовий ряд ціни закриття."""
    return prepare_close_series(df, fill_method="ffill")


def build_dataset(close: pd.Series, feature_config: Optional[FeatureConfig] = None) -> tuple[pd.DataFrame, pd.Series]:
    """Створює датасет X, y, де y - логарифм дохідності."""
    X = make_features(close, feature_config)
    y = np.log(close.shift(-1)) - np.log(close)
    df = pd.concat([X, y.rename("target")], axis=1)
    df = df.dropna()
    return df.drop(columns=["target"]), df["target"]


def _get_model() -> XGBRegressor:
    return XGBRegressor(**PARAMS)


def _fit_with_early_stopping(model: XGBRegressor, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Fits an XGBRegressor with optional early stopping on the most recent window."""

    if X_train.empty:
        raise ValueError("Training set is empty.")

    end_train_date = X_train.index.max()
    val_start_date = end_train_date - pd.Timedelta(days=VAL_DAYS)
    val_mask = X_train.index >= val_start_date

    use_val = 0 < val_mask.sum() < len(X_train) and val_mask.sum() >= MIN_VAL_SAMPLES
    eval_set = [(X_train.loc[val_mask], y_train.loc[val_mask])] if use_val else None

    try:
        if eval_set is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=[EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
    except TypeError:
        model.fit(X_train, y_train)


def _autoreg_one_step_forecasts(close: pd.Series, lags: int) -> pd.Series:
    """Generates one-step ahead AutoReg forecasts for each timestamp."""

    series = close.asfreq("B").ffill().bfill()
    idx = series.index
    forecasts: list[float] = []
    forecast_idx: list[pd.Timestamp] = []
    min_history = max(AR_MIN_HISTORY, lags + 5)

    for i in range(len(idx)):
        history = series.iloc[:i]
        if history.size < min_history:
            continue
        try:
            model = AutoReg(history, lags=lags, old_names=False).fit()
        except ValueError:
            continue
        pred = float(model.forecast(steps=1).iloc[0])
        forecasts.append(pred)
        forecast_idx.append(idx[i])

    return pd.Series(forecasts, index=pd.Index(forecast_idx, name=series.index.name))


def _generate_xgb_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = OOF_SPLITS,
) -> pd.Series:
    """Creates time-series out-of-fold predictions for the base XGB model."""

    if len(X_train) < (MIN_TRAIN_SAMPLES + 1):
        return pd.Series(index=X_train.index, dtype=float)

    n_splits = min(n_splits, len(X_train) - 1)
    if n_splits < 2:
        return pd.Series(index=X_train.index, dtype=float)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    preds = pd.Series(index=X_train.index, dtype=float)

    for fold, (idx_train, idx_val) in enumerate(splitter.split(X_train), start=1):
        X_tr = X_train.iloc[idx_train]
        y_tr = y_train.iloc[idx_train]
        X_val = X_train.iloc[idx_val]

        if len(X_tr) < MIN_TRAIN_SAMPLES:
            continue

        model = _get_model()
        _fit_with_early_stopping(model, X_tr, y_tr)
        preds.iloc[idx_val] = model.predict(X_val)
        print(f"OOF fold {fold}/{n_splits}: fitted on {len(X_tr)} obs, predicted {len(X_val)} obs")

    return preds


def train_once_predict_2025(
    close: pd.Series, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.Series, pd.Series, XGBRegressor, pd.Series]:
    """Fits a single XGB model on pre-2025 data and predicts 2025 prices."""

    train_mask = X.index.year < EVAL_YEAR
    test_mask = X.index.year == EVAL_YEAR
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test = X.loc[test_mask]

    model = _get_model()

    print("Training single XGB base model...")
    _fit_with_early_stopping(model, X_train, y_train)

    ret_pred = pd.Series(model.predict(X_test), index=X_test.index)
    y_pred = close.loc[X_test.index] * np.exp(ret_pred)
    y_true = close.shift(-1).loc[X_test.index]

    return y_pred, y_true, model, ret_pred


def walk_forward_predict_2025(
    close: pd.Series, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Predicts 2025 prices with periodic refits and returns log-return forecasts."""

    test_dates = X[X.index.year == EVAL_YEAR].index
    price_predictions: list[float] = []
    ret_predictions: list[float] = []
    model: Optional[XGBRegressor] = None

    print("Starting walk-forward prediction...")
    for i, date in enumerate(test_dates):
        if model is None or i % WALK_STEP == 0:
            print(f"Refitting model at step {i}/{len(test_dates)} ({date.date()})...")
            train_mask = X.index < date
            X_train, y_train = X.loc[train_mask], y.loc[train_mask]

            model = _get_model()
            _fit_with_early_stopping(model, X_train, y_train)

        ret_pred = float(model.predict(X.loc[[date]])[0])
        price_pred = close.loc[date] * np.exp(ret_pred)
        ret_predictions.append(ret_pred)
        price_predictions.append(price_pred)

    y_pred = pd.Series(price_predictions, index=test_dates)
    ret_series = pd.Series(ret_predictions, index=test_dates)
    y_true = close.shift(-1).loc[test_dates]
    return y_pred, y_true, ret_series


def plot_predictions(y_true: pd.Series, y_pred: pd.Series, acc: float, out_png: str):
    """Saves a plot comparing stacked-ensemble predictions with the ground truth."""

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(14, 7))
    plt.plot(y_true.index, y_true, label='Rzeczywiste 2025', color='blue')
    plt.plot(
        y_pred.index,
        y_pred,
        label='Prognoza Stacked (ARIMA + XGB)',
        color='orange',
        linestyle='--',
    )
    plt.title(f"USD/PLN — Stacked (ARIMA + XGB) wartość {EVAL_YEAR} (Tolerance acc: {acc:.2%})")
    plt.xlabel("Data")
    plt.ylabel("Cena zamknięcia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wykres zapisano: {out_png}")


def main():
    """Entry point building a stacked ARIMA + XGB ensemble for price forecasting."""

    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_or_fetch()
    close = prepare_series(df)
    X, y = build_dataset(close, FEATURE_CONFIG)

    train_mask = X.index.year < EVAL_YEAR
    test_mask = X.index.year == EVAL_YEAR
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]

    # Build ARIMA-derived feature (log-return implied by AutoReg forecast)
    print("Generating AutoReg forecasts for meta features...")
    arima_forecast = _autoreg_one_step_forecasts(close, AR_LAGS)
    current_close = close.reindex(X.index)
    arima_ret = (np.log(arima_forecast.reindex(X.index)) - np.log(current_close)).replace([np.inf, -np.inf], np.nan)

    # Out-of-fold XGB predictions to avoid leakage in meta training
    print("Creating out-of-fold XGB predictions for stacking...")
    oof_preds = _generate_xgb_oof_predictions(X_train, y_train)
    oof_full = pd.Series(index=X.index, dtype=float)
    oof_full.loc[oof_preds.index] = oof_preds

    meta_train = pd.DataFrame({
        "arima_ret": arima_ret.loc[train_mask],
        "xgb_ret": oof_full.loc[train_mask],
    }).dropna()
    meta_target = y.loc[meta_train.index]

    meta_model: Optional[LinearRegression]
    if len(meta_train) < MIN_TRAIN_SAMPLES:
        print("Meta-training data insufficient; fallback to base XGB predictions.")
        meta_model = None
    else:
        meta_model = LinearRegression()
        meta_model.fit(meta_train, meta_target)
        print(f"Meta model fitted on {len(meta_train)} observations.")

    # Base predictions on the evaluation window
    if WALK_FORWARD:
        _, y_true_prices, base_ret_pred = walk_forward_predict_2025(close, X, y)
    else:
        _, y_true_prices, _, base_ret_pred = train_once_predict_2025(close, X, y)

    meta_index = X.index[test_mask]
    meta_features_test = pd.DataFrame(index=meta_index)
    meta_features_test["arima_ret"] = arima_ret.loc[meta_index]
    meta_features_test["xgb_ret"] = base_ret_pred.reindex(meta_index)

    meta_ret_pred = pd.Series(index=meta_index, dtype=float)
    if meta_model is not None:
        valid_idx = meta_features_test.dropna().index
        if len(valid_idx) > 0:
            meta_ret_pred.loc[valid_idx] = meta_model.predict(meta_features_test.loc[valid_idx])
            print(f"Meta predictions generated for {len(valid_idx)} of {len(meta_index)} test days.")
        missing_idx = meta_ret_pred[meta_ret_pred.isna()].index
        if len(missing_idx) > 0:
            print(f"Falling back to base XGB for {len(missing_idx)} missing stacked predictions.")
            meta_ret_pred.loc[missing_idx] = base_ret_pred.reindex(missing_idx)
    else:
        meta_ret_pred = base_ret_pred.reindex(meta_index)

    y_true_prices = y_true_prices.reindex(meta_index)
    meta_price_pred = close.loc[meta_ret_pred.index] * np.exp(meta_ret_pred)

    valid_eval_idx = meta_price_pred.index[meta_price_pred.notna() & y_true_prices.notna()]
    meta_price_pred = meta_price_pred.loc[valid_eval_idx]
    y_true_prices = y_true_prices.loc[valid_eval_idx]

    acc, desc = evaluate_tolerance(y_true_prices, meta_price_pred, TOL_ABS, TOL_PCT)
    metrics = regression_metrics(y_true_prices, meta_price_pred)

    print(f"Dokładność stacked (tolerancja {desc}) {EVAL_YEAR}: {acc*100:.2f}% (n={len(y_true_prices)})")

    metrics_df = pd.DataFrame([{
        **metrics,
        "tolerance_accuracy": acc,
        "tolerance": desc,
        "n": len(y_true_prices),
        "mode": "walk_forward" if WALK_FORWARD else "train_once",
        "year": EVAL_YEAR,
        "family": FAMILY,
        "task": "price",
        "model": "stacked_arima_xgb",
    }])

    metrics_path = os.path.join(REPORT_DIR, f"metrics_price_{EVAL_YEAR}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metryki zapisano: {metrics_path}")

    plot_path = os.path.join(REPORT_DIR, f"price_{EVAL_YEAR}.png")
    plot_predictions(y_true_prices, meta_price_pred, acc, plot_path)

if __name__ == "__main__":
    main()
