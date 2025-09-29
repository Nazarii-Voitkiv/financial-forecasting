"""Compatibility wrappers for shared RF utilities."""

from __future__ import annotations

import pandas as pd

from ..common import load_or_fetch as _load_or_fetch
from ..common import prepare_close_series as _prepare_close_series


def load_or_fetch() -> pd.DataFrame:
    return _load_or_fetch()


def prepare_series(df: pd.DataFrame) -> pd.Series:
    return _prepare_close_series(df, fill_method="interpolate")


__all__ = ["load_or_fetch", "prepare_series"]

