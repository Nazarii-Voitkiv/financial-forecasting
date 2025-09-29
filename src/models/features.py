"""Feature generation helpers shared by tree-based models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


@dataclass(frozen=True)
class FeatureConfig:
    lag_returns: int = 10
    momentum_windows: tuple[int, ...] = (3, 5, 10, 20)
    sma_windows: tuple[int, ...] = (5, 10, 20)
    volatility_windows: tuple[int, ...] = (5, 10, 20)
    bollinger_window: int = 20
    rsi_windows: tuple[int, ...] = (14, 7)


def make_features(close: pd.Series, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    config = config or FeatureConfig()
    close = close.copy()
    r = close.pct_change()
    log_ret = np.log(close).diff()
    X = pd.DataFrame(index=close.index)

    for k in range(1, config.lag_returns + 1):
        X[f"ret_lag_{k}"] = r.shift(k)
        X[f"logret_lag_{k}"] = log_ret.shift(k)

    for w in config.momentum_windows:
        X[f"mom_{w}"] = close.pct_change(w)

    for w in config.sma_windows:
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

    for w in config.volatility_windows:
        X[f"vol_{w}"] = r.rolling(w).std()
    X["ewm_vol_10"] = r.ewm(span=10, adjust=False).std()

    w = config.bollinger_window
    sma_w = close.rolling(w).mean()
    std_w = close.rolling(w).std()
    upper = sma_w + 2 * std_w
    lower = sma_w - 2 * std_w
    X[f"bb_width{w}"] = (upper - lower) / sma_w
    X[f"bb_percent_b{w}"] = (close - lower) / (upper - lower)
    X[f"zscore_{w}"] = (close - sma_w) / (std_w + 1e-12)

    for w in config.sma_windows:
        roll_max = close.rolling(w).max()
        roll_min = close.rolling(w).min()
        X[f"dist_max_{w}"] = close / roll_max - 1
        X[f"dist_min_{w}"] = close / roll_min - 1

    for w in config.rsi_windows:
        X[f"rsi{w}"] = rsi(close, w)

    dow = close.index.dayofweek
    for d in range(5):
        X[f"dow_{d}"] = (dow == d).astype(int)

    m = close.index.month
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)

    return X


__all__ = [
    "FeatureConfig",
    "ema",
    "make_features",
    "rsi",
]
