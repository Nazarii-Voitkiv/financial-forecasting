import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
import yfinance as yf

DATA_RAW = "data/raw/usdpln_yahoo_daily.csv"

def load_or_fetch():
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
                    df = None
            except Exception:
                df = None
        if df is not None:
            return df
    df = yf.download("USDPLN=X", interval="1d", auto_adjust=False, progress=False).dropna().sort_index()
    df.index.name = "Date"
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    df.to_csv(DATA_RAW)
    return df

def prepare_series(df: pd.DataFrame) -> pd.Series:
    s = df["Close"].astype(float).sort_index()
    s = s.asfreq("B").interpolate()
    return s

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def make_features(close: pd.Series) -> pd.DataFrame:
    close = close.copy()
    r = close.pct_change()
    log_ret = np.log(close).diff()
    X = pd.DataFrame(index=close.index)

    for k in range(1, 11):
        X[f"ret_lag_{k}"] = r.shift(k)
        X[f"logret_lag_{k}"] = log_ret.shift(k)

    for w in (3, 5, 10, 20):
        X[f"mom_{w}"] = close.pct_change(w)

    for w in (5, 10, 20):
        sma = close.rolling(w).mean()
        X[f"sma_{w}_rel"] = close / sma - 1
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    X["ema12_rel"] = close / ema12 - 1
    X["ema26_rel"] = close / ema26 - 1

    macd = ema12 - ema26
    macd_sig = ema(macd, 9)
    X["macd"] = macd
    X["macd_sig"] = macd_sig
    X["macd_hist"] = macd - macd_sig

    for w in (5, 10, 20):
        X[f"vol_{w}"] = r.rolling(w).std()
    X["ewm_vol_10"] = r.ewm(span=10, adjust=False).std()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    width = (upper - lower) / sma20
    pb = (close - lower) / (upper - lower)
    X["bb_width20"] = width
    X["bb_percent_b20"] = pb
    X["zscore_20"] = (close - sma20) / (std20 + 1e-12)

    for w in (5, 10, 20):
        roll_max = close.rolling(w).max()
        roll_min = close.rolling(w).min()
        X[f"dist_max_{w}"] = close / roll_max - 1
        X[f"dist_min_{w}"] = close / roll_min - 1

    X["rsi14"] = rsi(close, 14)
    X["rsi7"] = rsi(close, 7)

    dow = close.index.dayofweek
    for d in range(5):
        X[f"dow_{d}"] = (dow == d).astype(int)
    m = close.index.month
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)

    return X

REPORT_DIR = "data/reports/rf"
METRICS_CSV = os.path.join(REPORT_DIR, "metrics_updown_2025.csv")

def build_dataset(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(close)
    y = (close.shift(-1) > close).astype(int)
    df = X.copy()
    df["target"] = y
    df = df.dropna()
    X = df.drop(columns=["target"]) 
    y = df["target"]
    return X, y

def train_once_predict_2025(X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int, random_state: int, thresh: float = 0.5, optimize_thresh: bool = False) -> tuple[pd.Series, pd.Series, np.ndarray, float, RandomForestClassifier]:
    train_mask = X.index.year <= 2024
    test_mask = X.index.year == 2025
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    best_thresh = thresh
    if optimize_thresh:
        # ostatnie 90 dni 2024 як валідація для підбору порогу
        val_mask = (X.index.year == 2024) & (X.index >= (X.index[X.index.year == 2024].max() - pd.Timedelta(days=120)))
        if val_mask.any():
            val_probs = model.predict_proba(X.loc[val_mask])[:, 1]
            val_true = y.loc[val_mask].values
            grid = np.linspace(0.3, 0.7, 41)
            accs = [(t, ( (val_probs >= t).astype(int) == val_true ).mean()) for t in grid]
            best_thresh = max(accs, key=lambda x: x[1])[0]
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= best_thresh).astype(int)
    return pd.Series(y_pred, index=X_test.index), y_test, probs, best_thresh, model

def walk_forward_predict_2025(X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int, random_state: int) -> tuple[pd.Series, pd.Series, np.ndarray]:
    test_idx = X.index[X.index.year == 2025]
    preds = []
    probs = []
    n = len(test_idx)
    last_print = None
    for i, d in enumerate(test_idx):
        prev_date = X.index[X.index.get_loc(d) - 1]
        train_mask = X.index <= prev_date
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1, class_weight="balanced")
        model.fit(X.loc[train_mask], y.loc[train_mask])
        p = model.predict_proba(X.loc[[d]])[0, 1]
        probs.append(p)
        preds.append(1 if p >= 0.5 else 0)
        remaining = int(round(100 * (1 - (i + 1) / n)))
        if last_print is None or remaining != last_print:
            print(f"Pozostało: {remaining}%")
            last_print = remaining
    y_pred = pd.Series(preds, index=test_idx)
    y_test = y.loc[test_idx]
    return y_pred, y_test, np.array(probs)

def bootstrap_ci_accuracy(y_true: np.ndarray, y_pred: np.ndarray, block: int = 5, n_boot: int = 1000, seed: int = 42) -> tuple[float,float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)
    def one_sample():
        out = []
        i = 0
        while i < n:
            start = rng.integers(0, n)
            end = min(start + block, n)
            out.extend(range(start, end))
            i += (end - start)
        return np.array(out[:n])
    accs = []
    for _ in range(n_boot):
        s = one_sample()
        accs.append((y_true[s] == y_pred[s]).mean())
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)

def feature_importance_report(model: RandomForestClassifier, feature_names: list[str], out_csv: str):
    imp = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    imp.to_csv(out_csv, index=False)
    print(f"Zapisano ważności cech: {out_csv}")

def plot_predictions(close: pd.Series, y_pred: pd.Series, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(12, 5))
    c = close.loc[y_pred.index]
    plt.plot(c.index, c.values, label="Rzeczywiste 2025")
    up_idx = y_pred.index[y_pred.values == 1]
    dn_idx = y_pred.index[y_pred.values == 0]
    plt.scatter(up_idx, c.loc[up_idx].values, marker="^", color="green", s=30, label="Pred UP")
    plt.scatter(dn_idx, c.loc[dn_idx].values, marker="v", color="red", s=30, label="Pred DOWN")
    plt.title("USD/PLN — Random Forest kierunek 2025")
    plt.xlabel("Data")
    plt.ylabel("Zamknięcie")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wykres zapisano: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Random Forest dla USD/PLN — up/down 2025")
    parser.add_argument("--n-est", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--opt-thresh", action="store_true")
    args = parser.parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_or_fetch()
    close = prepare_series(df)
    X, y = build_dataset(close)
    if args.walk_forward:
        y_pred, y_test, y_prob = walk_forward_predict_2025(X, y, args.n_est, args.max_depth, args.seed)
        model = RandomForestClassifier(n_estimators=args.n_est, max_depth=args.max_depth, random_state=args.seed, n_jobs=-1, class_weight="balanced")
        model.fit(X.loc[X.index.year <= 2024], y.loc[y.index.year <= 2024])
        used_thresh = 0.5
    else:
        y_pred, y_test, y_prob, used_thresh, model = train_once_predict_2025(X, y, args.n_est, args.max_depth, args.seed, args.thresh, args.opt_thresh)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float('nan')
    lo, hi = bootstrap_ci_accuracy(y_test.values.astype(int), y_pred.values.astype(int))
    print(f"Dokładność up/down 2025: {acc*100:.2f}% (95% CI {lo*100:.1f}-{hi*100:.1f}) (n={len(y_pred)}) — próg {used_thresh:.2f}")
    print(f"Balanced Acc: {bacc*100:.2f}%  MCC: {mcc:.3f}  ROC-AUC: {auc:.3f}  Baseline: 50.00%")
    # save metrics
    os.makedirs(REPORT_DIR, exist_ok=True)
    import pandas as _pd
    _pd.DataFrame([{
        "accuracy": float(acc),
        "acc_lo": lo,
        "acc_hi": hi,
        "balanced_accuracy": float(bacc),
        "mcc": float(mcc),
        "roc_auc": float(auc),
        "n": int(len(y_pred)),
        "threshold": float(used_thresh),
        "mode": "walk" if args.walk_forward else "train_once"
    }]).to_csv(METRICS_CSV, index=False)
    feature_importance_report(model, list(X.columns), os.path.join(REPORT_DIR, "feature_importance_2024.csv"))
    plot_predictions(close, y_pred, os.path.join(REPORT_DIR, "updown_2025.png"))

if __name__ == "__main__":
    main()
