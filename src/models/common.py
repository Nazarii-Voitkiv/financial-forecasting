"""Shared data utilities for financial forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


DATA_RAW = Path("data/raw/usdpln_yahoo_daily.csv")
DEFAULT_SYMBOL = "USDPLN=X"


def _try_read_cached(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load cached CSV data handling alternate Yahoo! export formats."""
    try:
        return pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    except Exception:
        try:
            tmp = pd.read_csv(csv_path, header=2)
        except Exception:
            return None
        if "Date" not in tmp.columns:
            return None
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp = tmp.dropna(subset=["Date"]).set_index("Date").sort_index()
        if "Close" not in tmp.columns:
            if "Unnamed: 3" in tmp.columns:
                tmp = tmp.rename(columns={"Unnamed: 3": "Close"})
            elif "Adj Close" in tmp.columns:
                tmp["Close"] = tmp["Adj Close"]
        return tmp


def load_or_fetch(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = "1d",
    force_download: bool = False,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load cached FX data or download it via yfinance if necessary."""

    csv_path = Path(cache_path or DATA_RAW)
    if not force_download and csv_path.exists():
        cached = _try_read_cached(csv_path)
        if cached is not None:
            return cached

    df = yf.download(symbol, interval=interval, auto_adjust=False, progress=False)
    df = df.dropna().sort_index()
    df.index.name = "Date"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)
    return df


def prepare_close_series(
    df: pd.DataFrame,
    freq: str = "B",
    fill_method: str = "ffill",
    allow_interpolate: bool = True,
) -> pd.Series:
    """Return a cleaned close-price series on the desired calendar."""

    series = df["Close"].astype(float).sort_index()
    series = series.asfreq(freq)

    method = fill_method.lower()
    if method == "interpolate":
        series = series.interpolate(limit_direction="both")
    else:
        # defaults to forward/backward fill to avoid weekend gaps
        series = series.ffill().bfill()
        if method == "ffill_only":
            series = series.ffill()
        elif method == "bfill_only":
            series = series.bfill()
        elif method not in {"ffill", "ffill_only", "bfill_only"} and allow_interpolate:
            # fallback to interpolation only when explicitly allowed
            series = series.interpolate(limit_direction="both")

    return series


def evaluate_tolerance(
    y_true: pd.Series,
    y_pred: pd.Series,
    tol_abs: float,
    tol_pct: float,
) -> tuple[float, str]:
    """Compute tolerance accuracy and human-readable tolerance description."""

    if tol_pct > 0:
        tol = y_true.abs() * (tol_pct / 100.0)
        desc = f"{tol_pct}%"
    else:
        tol = pd.Series(tol_abs, index=y_true.index)
        desc = f"{tol_abs} PLN"
    acc = float(((y_true - y_pred).abs() <= tol).mean())
    return acc, desc


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return MAE, RMSE, and MAPE-style diagnostics for price forecasts."""

    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    diff = y_true - y_pred
    mae = float(diff.abs().mean())
    rmse = float(np.sqrt((diff**2).mean()))

    denom = y_true.replace(0.0, np.nan).abs()
    mape = float((diff.abs() / denom).dropna().mean() * 100) if denom.notna().any() else float("nan")
    medae = float(diff.abs().median())
    return {"mae": mae, "rmse": rmse, "medae": medae, "mape": mape}


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward evaluation loops."""

    year: int
    retrain_each_step: bool = True

    def test_index(self, index: pd.Index) -> pd.Index:
        return index[index.year == self.year]
