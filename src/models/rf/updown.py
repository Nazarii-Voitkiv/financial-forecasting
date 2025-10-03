import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score

from ..common import load_or_fetch, prepare_close_series
from ..features import FeatureConfig, make_features


REPORT_DIR = "data/reports/rf"
EVAL_YEAR = 2025
METRICS_CSV = os.path.join(REPORT_DIR, f"metrics_updown_{EVAL_YEAR}.csv")


# Konfiguracja
N_EST = int(os.getenv("RF_N_EST", "200"))
MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "8"))
SEED = 42
WALK_FORWARD = bool(int(os.getenv("RF_WALK_FORWARD", "1")))
WALK_STEP = max(1, int(os.getenv("RF_WALK_STEP", "20")))
THRESH = 0.5
OPT_THRESH = True
FEATURE_CONFIG = FeatureConfig(
    lag_returns=5, momentum_windows=(), sma_windows=(5,), volatility_windows=(), rsi_windows=()
)


def prepare_series(df: pd.DataFrame) -> pd.Series:
    return prepare_close_series(df, fill_method="interpolate")


def build_dataset(close: pd.Series, feature_config: Optional[FeatureConfig] = None) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(close, feature_config)
    y = (close.shift(-1) > close).astype(int)
    df = X.copy()
    df["target"] = y
    df = df.dropna()
    return df.drop(columns=["target"]), df["target"]


def _calibrate_threshold(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    base_threshold: float,
) -> float:
    val_mask_all = X.index.year == (EVAL_YEAR - 1)
    if not val_mask_all.any():
        return base_threshold
    last_date = X.index[val_mask_all].max()
    val_mask = (X.index >= last_date - pd.Timedelta(days=120)) & val_mask_all
    if not val_mask.any():
        return base_threshold
    val_probs = model.predict_proba(X.loc[val_mask])[:, 1]
    val_true = y.loc[val_mask].values
    grid = np.linspace(0.3, 0.7, 41)
    best_thresh, best_acc = base_threshold, -1.0
    for t in grid:
        acc = ((val_probs >= t).astype(int) == val_true).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(t)
    return best_thresh


def train_once_predict_2025(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    thresh: float = 0.5,
    optimize_thresh: bool = False,
) -> tuple[pd.Series, pd.Series, np.ndarray, float, RandomForestClassifier]:
    train_mask = X.index.year < EVAL_YEAR
    test_mask = X.index.year == EVAL_YEAR
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    best_thresh = thresh
    if optimize_thresh:
        best_thresh = _calibrate_threshold(model, X, y, thresh)
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= best_thresh).astype(int)
    return pd.Series(y_pred, index=X_test.index), y_test, probs, best_thresh, model


def walk_forward_predict_2025(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    threshold: float,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    test_idx = X.index[X.index.year == EVAL_YEAR]
    preds: list[int] = []
    probs: list[float] = []
    n = len(test_idx)
    last_print = None
    model: Optional[RandomForestClassifier] = None
    for i, d in enumerate(test_idx):
        if model is None or i % WALK_STEP == 0:
            prev_date = X.index[X.index.get_loc(d) - 1]
            train_mask = X.index <= prev_date
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            model.fit(X.loc[train_mask], y.loc[train_mask])
        p = model.predict_proba(X.loc[[d]])[0, 1]
        probs.append(float(p))
        preds.append(1 if p >= threshold else 0)
        remaining = int(round(100 * (1 - (i + 1) / n)))
        if last_print is None or remaining != last_print:
            print(f"Pozostało: {remaining}%")
            last_print = remaining
    y_pred = pd.Series(preds, index=test_idx)
    y_test = y.loc[test_idx]
    return y_pred, y_test, np.array(probs)


def bootstrap_ci_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def one_sample() -> np.ndarray:
        out: list[int] = []
        i = 0
        while i < n:
            start = rng.integers(0, n)
            end = min(start + block, n)
            out.extend(range(start, end))
            i += end - start
        return np.array(out[:n])

    accs = []
    for _ in range(n_boot):
        s = one_sample()
        accs.append((y_true[s] == y_pred[s]).mean())
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)


def feature_importance_report(
    model: RandomForestClassifier,
    feature_names: list[str],
    out_csv: str,
) -> None:
    imp = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    imp.to_csv(out_csv, index=False)
    print(f"Zapisano ważności cech: {out_csv}")


def plot_predictions(close: pd.Series, y_pred: pd.Series, y_true: pd.Series, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(12, 5))
    c = close.loc[y_pred.index]
    plt.plot(c.index, c.values, label=f"Rzeczywiste {EVAL_YEAR}")
    up_idx = y_pred.index[y_pred.values == 1]
    dn_idx = y_pred.index[y_pred.values == 0]
    plt.scatter(up_idx, c.loc[up_idx].values, marker="^", color="green", s=30, label="Pred UP")
    plt.scatter(dn_idx, c.loc[dn_idx].values, marker="v", color="red", s=30, label="Pred DOWN")
    accuracy = accuracy_score(y_true, y_pred)
    plt.title(f"USD/PLN — Random Forest kierunek {EVAL_YEAR} (Dokładność: {accuracy:.2%})")
    plt.xlabel("Data")
    plt.ylabel("Zamknięcie")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wykres zapisano: {out_png}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_or_fetch()
    close = prepare_series(df)
    X, y = build_dataset(close, FEATURE_CONFIG)

    # 1) base model trained on pre-2025 data to calibrate threshold and get feature importances
    base_model = RandomForestClassifier(
        n_estimators=N_EST,
        max_depth=MAX_DEPTH,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    mask_train_pre_2025 = X.index.year < EVAL_YEAR
    base_model.fit(X.loc[mask_train_pre_2025], y.loc[mask_train_pre_2025])
    used_thresh = _calibrate_threshold(base_model, X, y, THRESH)
    model = base_model  # for feature importance report

    if WALK_FORWARD:
        y_pred, y_test, y_prob = walk_forward_predict_2025(X, y, N_EST, MAX_DEPTH, SEED, used_thresh)
    else:
        y_pred, y_test, y_prob, used_thresh, model = train_once_predict_2025(
            X,
            y,
            N_EST,
            MAX_DEPTH,
            SEED,
            THRESH,
            OPT_THRESH,
        )
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")
    lo, hi = bootstrap_ci_accuracy(y_test.values.astype(int), y_pred.values.astype(int))
    print(
        f"Dokładność up/down {EVAL_YEAR}: {acc*100:.2f}% (95% CI {lo*100:.1f}-{hi*100:.1f}) (n={len(y_pred)}) — próg {used_thresh:.2f}"
    )
    print(f"Balanced Acc: {bacc*100:.2f}%  MCC: {mcc:.3f}  ROC-AUC: {auc:.3f}  Baseline: 50.00%")

    # Add family/task metadata and write metrics
    pd.DataFrame(
        [
            {
                "accuracy": float(acc),
                "acc_lo": lo,
                "acc_hi": hi,
                "balanced_accuracy": float(bacc),
                "mcc": float(mcc),
                "roc_auc": float(auc),
                "n": int(len(y_pred)),
                "threshold": float(used_thresh),
                "mode": "walk" if WALK_FORWARD else "train_once",
                "year": int(EVAL_YEAR),
                "family": "rf",
                "task": "updown",
            }
        ]
    ).to_csv(METRICS_CSV, index=False)

    feature_importance_report(model, list(X.columns), os.path.join(REPORT_DIR, f"feature_importance_{EVAL_YEAR-1}.csv"))
    plot_predictions(close, y_pred, y_test, os.path.join(REPORT_DIR, f"updown_{EVAL_YEAR}.png"))


if __name__ == "__main__":
    main()