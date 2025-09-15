import numpy as np
import pandas as pd

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

    # Lags of returns
    for k in range(1, 11):
        X[f"ret_lag_{k}"] = r.shift(k)
        X[f"logret_lag_{k}"] = log_ret.shift(k)

    # Momentum windows
    for w in (3, 5, 10, 20):
        X[f"mom_{w}"] = close.pct_change(w)

    # SMA/EMA relations
    for w in (5, 10, 20):
        sma = close.rolling(w).mean()
        X[f"sma_{w}_rel"] = close / sma - 1
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    X["ema12_rel"] = close / ema12 - 1
    X["ema26_rel"] = close / ema26 - 1

    # MACD family
    macd = ema12 - ema26
    macd_sig = ema(macd, 9)
    X["macd"] = macd
    X["macd_sig"] = macd_sig
    X["macd_hist"] = macd - macd_sig

    # Volatility proxies
    for w in (5, 10, 20):
        X[f"vol_{w}"] = r.rolling(w).std()
    X["ewm_vol_10"] = r.ewm(span=10, adjust=False).std()

    # Bollinger bands features (20)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    width = (upper - lower) / sma20
    pb = (close - lower) / (upper - lower)
    X["bb_width20"] = width
    X["bb_percent_b20"] = pb
    X["zscore_20"] = (close - sma20) / (std20 + 1e-12)

    # Rolling extremes distances
    for w in (5, 10, 20):
        roll_max = close.rolling(w).max()
        roll_min = close.rolling(w).min()
        X[f"dist_max_{w}"] = close / roll_max - 1
        X[f"dist_min_{w}"] = close / roll_min - 1

    # RSI family
    X["rsi14"] = rsi(close, 14)
    X["rsi7"] = rsi(close, 7)

    # Calendar effects
    dow = close.index.dayofweek
    for d in range(5):
        X[f"dow_{d}"] = (dow == d).astype(int)
    # Month seasonality (sin/cos)
    m = close.index.month
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)

    return X
