import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, roc_curve
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
    from xgboost.callback import EarlyStopping
except ImportError:
    print("XGBoost nie jest zainstalowany lub wystąpił błąd importu. Zainstaluj go: pip install xgboost")
    exit(1)

from ..common import load_or_fetch, prepare_close_series
from ..features import FeatureConfig, make_features

# --- Stałe ---
FAMILY = "xgb"
REPORT_DIR = f"data/reports/{FAMILY}"
EVAL_YEAR = 2025
WALK_FORWARD = True
WALK_STEP = 20
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
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 2.0,
    "n_jobs": -1,
    "random_state": SEED,
    "eval_metric": "logloss",
}

EARLY_STOPPING_ROUNDS = 200
VAL_DAYS = 120


def prepare_series(df: pd.DataFrame) -> pd.Series:
    """Przygotowuje szereg czasowy ceny zamknięcia."""
    return prepare_close_series(df, fill_method="ffill")


def build_dataset(close: pd.Series, feature_config: Optional[FeatureConfig] = None) -> tuple[pd.DataFrame, pd.Series]:
    """Tworzy zbiór danych X, y, gdzie y to binarny kierunek ruchu."""
    X = make_features(close, feature_config)
    y = (close.shift(-1) > close).astype(int)
    df = pd.concat([X, y.rename("target")], axis=1)
    df = df.dropna()
    return df.drop(columns=["target"]), df["target"]


def _get_model() -> XGBClassifier:
    return XGBClassifier(**PARAMS)


def _fit_with_early_stopping(model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Fits an XGBClassifier with optional early stopping on a validation tail."""

    if X_train.empty:
        raise ValueError("Training set is empty.")

    end_train = X_train.index.max()
    val_start = end_train - pd.Timedelta(days=VAL_DAYS)
    val_mask = X_train.index >= val_start

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
    """Generates one-step-ahead AutoReg forecasts for each timestamp."""

    series = close.asfreq("B").ffill().bfill()
    idx = series.index
    preds: list[float] = []
    pred_idx: list[pd.Timestamp] = []
    min_history = max(AR_MIN_HISTORY, lags + 5)

    for i in range(len(idx)):
        history = series.iloc[:i]
        if history.size < min_history:
            continue
        try:
            model = AutoReg(history, lags=lags, old_names=False).fit()
        except ValueError:
            continue
        preds.append(float(model.forecast(steps=1).iloc[0]))
        pred_idx.append(idx[i])

    return pd.Series(preds, index=pd.Index(pred_idx, name=series.index.name))


def _generate_xgb_oof_probabilities(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = OOF_SPLITS,
) -> pd.Series:
    """Creates time-series out-of-fold probability forecasts for the base classifier."""

    if len(X_train) < (MIN_TRAIN_SAMPLES + 1):
        return pd.Series(index=X_train.index, dtype=float)

    n_splits = min(n_splits, len(X_train) - 1)
    if n_splits < 2:
        return pd.Series(index=X_train.index, dtype=float)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    probs = pd.Series(index=X_train.index, dtype=float)

    for fold, (idx_tr, idx_val) in enumerate(splitter.split(X_train), start=1):
        X_tr = X_train.iloc[idx_tr]
        y_tr = y_train.iloc[idx_tr]
        X_val = X_train.iloc[idx_val]

        if len(X_tr) < MIN_TRAIN_SAMPLES or y_tr.nunique() < 2:
            continue

        model = _get_model()
        _fit_with_early_stopping(model, X_tr, y_tr)
        probs.iloc[idx_val] = model.predict_proba(X_val)[:, 1]
        print(f"OOF fold {fold}/{n_splits}: fitted on {len(X_tr)} obs, predicted {len(X_val)} obs")

    return probs


def _calibrate_from_probs(
    probs: pd.Series,
    y_true: pd.Series,
    default_thresh: float = 0.5,
) -> float:
    """Calibrates a probability threshold using Youden index on the latest year - 1."""

    if probs.empty:
        return default_thresh

    mask = (probs.index.year == (EVAL_YEAR - 1)) & probs.notna() & y_true.loc[probs.index].notna()
    if mask.sum() < MIN_VAL_SAMPLES:
        return default_thresh

    val_probs = probs.loc[mask]
    val_true = y_true.loc[val_probs.index].values
    fpr, tpr, thresholds = roc_curve(val_true, val_probs)
    if len(thresholds) == 0:
        return default_thresh
    best = thresholds[np.argmax(tpr - fpr)]
    return float(best)

def _calibrate_threshold(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    """Kalibruje próg na podstawie indeksu Youdena na ostatnich VAL_DAYS dniach 2024 roku."""
    val_mask_all = X.index.year == (EVAL_YEAR - 1)
    if not val_mask_all.any():
        return 0.5
    
    last_date = X[val_mask_all].index.max()
    val_start_date = last_date - pd.Timedelta(days=VAL_DAYS)
    val_mask = (X.index >= val_start_date) & (X.index <= last_date)
    
    if not val_mask.any():
        return 0.5

    val_probs = model.predict_proba(X.loc[val_mask])[:, 1]
    val_true = y.loc[val_mask].values
    fpr, tpr, thresholds = roc_curve(val_true, val_probs)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    return float(best_thresh)


def train_once_predict_2025(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, float, XGBClassifier]:
    """Fits a single base classifier on pre-2025 data and predicts 2025."""
    train_mask = X.index.year < EVAL_YEAR
    test_mask = X.index.year == EVAL_YEAR
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test = X.loc[test_mask]

    model = _get_model()

    print("Training single base classifier...")
    _fit_with_early_stopping(model, X_train, y_train)

    used_thresh = _calibrate_threshold(model, X, y)
    probs = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    y_pred = (probs >= used_thresh).astype(int)

    return pd.Series(y_pred, index=X_test.index), y.loc[test_mask], probs, used_thresh, model


def walk_forward_predict_2025(
    X: pd.DataFrame, y: pd.Series, threshold: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Performs walk-forward inference returning probabilities alongside class labels."""

    test_dates = X[X.index.year == EVAL_YEAR].index
    predictions: list[int] = []
    probabilities: list[float] = []
    model: Optional[XGBClassifier] = None

    print("Starting walk-forward prediction...")
    for i, date in enumerate(test_dates):
        if model is None or i % WALK_STEP == 0:
            print(f"Refitting model at step {i}/{len(test_dates)} ({date.date()})...")
            train_mask = X.index < date
            X_train, y_train = X.loc[train_mask], y.loc[train_mask]
            model = _get_model()
            _fit_with_early_stopping(model, X_train, y_train)

        prob = float(model.predict_proba(X.loc[[date]])[0, 1])
        probabilities.append(prob)
        predictions.append(1 if prob >= threshold else 0)

    y_pred = pd.Series(predictions, index=test_dates)
    prob_series = pd.Series(probabilities, index=test_dates)
    y_true = y.loc[test_dates]
    return y_pred, y_true, prob_series


def bootstrap_ci_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, block_size: int = 5) -> tuple[float, float]:
    """Oblicza 95% przedział ufności dla accuracy z blokowym bootstrapem."""
    rng = np.random.default_rng(SEED)
    accuracies = []
    n = len(y_true)
    for _ in range(n_boot):
        indices = []
        while len(indices) < n:
            start = rng.integers(0, n - block_size)
            indices.extend(range(start, start + block_size))
        indices = indices[:n]
        boot_true = y_true[indices]
        boot_pred = y_pred[indices]
        accuracies.append(accuracy_score(boot_true, boot_pred))
    return np.percentile(accuracies, [2.5, 97.5])


def feature_importance_report(model: XGBClassifier, feature_names: list[str], out_csv: str):
    """Zapisuje raport o ważności cech."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    imp.to_csv(out_csv, index=False)
    print(f"Raport o ważności cech zapisano: {out_csv}")


def plot_predictions(close: pd.Series, y_pred: pd.Series, y_true: pd.Series, out_png: str):
    """Zapisuje wykres z prognozami UP/DOWN dla modelu stacked."""

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(14, 7))
    c_test = close.loc[y_pred.index]
    plt.plot(c_test.index, c_test, label='Cena USD/PLN 2025', color='black')

    up_days = y_pred[y_pred == 1].index
    down_days = y_pred[y_pred == 0].index
    plt.scatter(up_days, c_test.loc[up_days], color='green', marker='^', s=50, label='Prognoza Stacked UP')
    plt.scatter(down_days, c_test.loc[down_days], color='red', marker='v', s=50, label='Prognoza Stacked DOWN')

    acc = accuracy_score(y_true, y_pred)
    plt.title(f"Prognoza kierunku USD/PLN (Stacked ARIMA + XGB) - {EVAL_YEAR} (Accuracy: {acc:.2%})")
    plt.xlabel("Data")
    plt.ylabel("Cena zamknięcia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wykres prognoz zapisano: {out_png}")


def main():
    """Główna funkcja uruchamiająca stacked ensemble ARIMA + XGB dla klasyfikacji kierunku."""

    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_or_fetch()
    close = prepare_series(df)
    X, y = build_dataset(close, FEATURE_CONFIG)

    train_mask = X.index.year < EVAL_YEAR
    test_mask = X.index.year == EVAL_YEAR
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]

    # Base model for calibration and feature importances
    print("Training base model for threshold calibration...")
    base_model = _get_model()
    _fit_with_early_stopping(base_model, X_train, y_train)
    base_threshold = _calibrate_threshold(base_model, X, y)
    print(f"Calibrated base threshold on 2024 data: {base_threshold:.4f}")

    print("Generating AutoReg forecasts for meta features...")
    arima_forecast = _autoreg_one_step_forecasts(close, AR_LAGS)
    current_close = close.reindex(X.index)
    arima_ret = (arima_forecast.reindex(X.index) / current_close - 1.0).replace([np.inf, -np.inf], np.nan)

    print("Preparing out-of-fold base probabilities for stacking...")
    oof_probs = _generate_xgb_oof_probabilities(X_train, y_train)
    oof_full = pd.Series(index=X.index, dtype=float)
    oof_full.loc[oof_probs.index] = oof_probs

    meta_train = pd.DataFrame({
        "xgb_prob": oof_full.loc[train_mask],
        "arima_ret": arima_ret.loc[train_mask],
    })
    meta_train["arima_signal"] = (meta_train["arima_ret"] > 0).astype(int)
    meta_train = meta_train.dropna()
    meta_target = y.loc[meta_train.index]

    meta_model: Optional[LogisticRegression]
    meta_threshold = base_threshold

    if len(meta_train) >= MIN_TRAIN_SAMPLES and meta_target.nunique() == 2:
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(meta_train, meta_target)
        meta_train_probs = pd.Series(meta_model.predict_proba(meta_train)[:, 1], index=meta_train.index)
        meta_threshold = _calibrate_from_probs(meta_train_probs, meta_target, default_thresh=base_threshold)
        if not np.isfinite(meta_threshold):
            meta_threshold = base_threshold
        print(f"Meta model fitted on {len(meta_train)} observations. Meta threshold {meta_threshold:.4f}")
    else:
        meta_model = None
        print("Meta-training data insufficient or single class; using base XGB probabilities only.")

    if WALK_FORWARD:
        _, y_test, base_probs = walk_forward_predict_2025(X, y, base_threshold)
        model_for_importance = base_model
    else:
        _, y_test, base_probs, used_thresh, model_for_importance = train_once_predict_2025(X, y)
        base_threshold = used_thresh
        if meta_model is None:
            meta_threshold = base_threshold

    meta_index = X.index[test_mask]
    meta_features_test = pd.DataFrame(index=meta_index)
    meta_features_test["xgb_prob"] = base_probs.reindex(meta_index)
    meta_features_test["arima_ret"] = arima_ret.loc[meta_index]
    meta_features_test["arima_signal"] = (meta_features_test["arima_ret"] > 0).astype(int)

    meta_probs = pd.Series(index=meta_index, dtype=float)
    if meta_model is not None:
        valid_idx = meta_features_test.dropna().index
        if len(valid_idx) > 0:
            meta_probs.loc[valid_idx] = meta_model.predict_proba(meta_features_test.loc[valid_idx])[:, 1]
            print(f"Meta probabilities generated for {len(valid_idx)} of {len(meta_index)} test days.")
        missing_idx = meta_probs[meta_probs.isna()].index
        if len(missing_idx) > 0:
            print(f"Filling {len(missing_idx)} missing stacked probabilities with base XGB outputs.")
            meta_probs.loc[missing_idx] = base_probs.reindex(missing_idx)
    else:
        meta_probs = base_probs.reindex(meta_index)

    meta_probs = meta_probs.fillna(base_probs.reindex(meta_index))
    y_test = y_test.reindex(meta_index)

    final_probs = meta_probs.loc[meta_index]
    valid_eval_idx = final_probs.index[final_probs.notna() & y_test.notna()]
    final_probs = final_probs.loc[valid_eval_idx]
    y_test = y_test.loc[valid_eval_idx]

    final_threshold = meta_threshold if meta_model is not None else base_threshold
    final_preds = (final_probs >= final_threshold).astype(int)

    acc = accuracy_score(y_test, final_preds)
    bacc = balanced_accuracy_score(y_test, final_preds)
    mcc = matthews_corrcoef(y_test, final_preds)
    try:
        auc = roc_auc_score(y_test, final_probs)
    except ValueError:
        auc = np.nan

    acc_lo, acc_hi = bootstrap_ci_accuracy(y_test.values, final_preds.values)

    print(f"\n--- Final Metrics for {EVAL_YEAR} (Stacked) ---")
    print(f"Accuracy: {acc:.2%} (95% CI: {acc_lo:.2%} - {acc_hi:.2%})")
    print(f"Balanced Accuracy: {bacc:.2%}")
    print(f"MCC: {mcc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"UP rate (true): {y_test.mean():.3f}, (pred): {final_preds.mean():.3f}")
    print(f"Final decision threshold: {final_threshold:.4f}")

    metrics_df = pd.DataFrame([{
        "accuracy": float(acc),
        "acc_lo": float(acc_lo),
        "acc_hi": float(acc_hi),
        "balanced_accuracy": float(bacc),
        "mcc": float(mcc),
        "roc_auc": float(auc),
        "n": int(len(y_test)),
        "threshold": float(final_threshold),
        "mode": "walk_forward" if WALK_FORWARD else "train_once",
        "year": EVAL_YEAR,
        "family": FAMILY,
        "task": "updown",
        "model": "stacked_arima_xgb",
    }])

    metrics_path = os.path.join(REPORT_DIR, f"metrics_updown_{EVAL_YEAR}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetryki zapisano: {metrics_path}")

    imp_path = os.path.join(REPORT_DIR, f"feature_importance_{EVAL_YEAR-1}.csv")
    feature_importance_report(model_for_importance, list(X.columns), imp_path)

    plot_path = os.path.join(REPORT_DIR, f"updown_{EVAL_YEAR}.png")
    plot_predictions(close, final_preds, y_test, plot_path)

if __name__ == "__main__":
    main()
