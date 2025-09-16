import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
try:
    from xgboost import XGBRegressor
except Exception as _e:
    import sys, platform
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

DATA_RAW = "data/raw/usdpln_yahoo_daily.csv"
REPORT_DIR = "data/reports/xgb"
METRICS_CSV = os.path.join(REPORT_DIR, "metrics_price_2025.csv")
TRAIN_END = pd.Timestamp("2024-12-31")

# Konfiguracja
N_EST = 300
MAX_DEPTH = 6
ETA = 0.1
SUBSAMPLE = 0.8
COLSAMPLE = 0.8
REG_LAMBDA = 1.0
SEED = 42
WALK_FORWARD = True
TOL_ABS = 0.01
TOL_PCT = 0.0


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
    s = s.asfreq("B").ffill().bfill()
    return s


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
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
    X["bb_width20"] = (upper - lower) / sma20
    X["bb_percent_b20"] = (close - lower) / (upper - lower)
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


def build_dataset(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(close)
    y = (np.log(close.shift(-1)) - np.log(close))
    df = X.copy()
    df["target"] = y
    df = df.dropna()
    return df.drop(columns=["target"]), df["target"]


def train_once_predict_2025(close: pd.Series, X: pd.DataFrame, y: pd.Series, params: dict):
    train_mask = X.index.year <= 2024
    test_mask = X.index.year == 2025
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test = X.loc[test_mask]

    model = XGBRegressor(
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
    model.fit(X_train, y_train)
    ret_pred = pd.Series(model.predict(X_test), index=X_test.index)
    y_pred = close.loc[X_test.index] * np.exp(ret_pred)
    y_true = close.shift(-1).loc[X_test.index]
    return y_pred, y_true, model


def walk_forward_predict_2025(close: pd.Series, X: pd.DataFrame, y: pd.Series, params: dict):
    test_idx = X.index[X.index.year == 2025]
    preds = []
    n = len(test_idx)
    last_print = None
    for i, d in enumerate(test_idx):
        prev_date = X.index[X.index.get_loc(d) - 1]
        train_mask = X.index <= prev_date
        model = XGBRegressor(
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


def evaluate_tolerance(y_true: pd.Series, y_pred: pd.Series, tol_abs: float, tol_pct: float):
    if tol_pct > 0:
        tol = y_true.abs() * (tol_pct / 100.0)
        desc = f"{tol_pct}%"
    else:
        tol = pd.Series(tol_abs, index=y_true.index)
        desc = f"{tol_abs} PLN"
    correct = ((y_true - y_pred).abs() <= tol).astype(int)
    acc = float(correct.mean())
    return acc, desc


def plot_predictions(close: pd.Series, y_pred: pd.Series, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(12, 5))
    c = close.loc[y_pred.index]
    plt.plot(c.index, c.values, label="Rzeczywiste 2025")
    plt.plot(y_pred.index, y_pred.values, label="Prognoza XGB")
    plt.title("USD/PLN — XGBoost wartość 2025")
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
    X, y = build_dataset(close)

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
    print(f"Dokładność (tolerancja {desc}) 2025: {acc*100:.2f}% (n={len(y_true)})")
    pd.DataFrame([{"tolerance_accuracy": acc, "tolerance": desc, "n": int(len(y_true)), "mode": "walk" if WALK_FORWARD else "train_once"}]).to_csv(METRICS_CSV, index=False)
    plot_predictions(close, y_pred, os.path.join(REPORT_DIR, "price_2025.png"))


if __name__ == "__main__":
    main()
