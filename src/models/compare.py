"""Aggregate 2025 forecasting metrics across models."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd


PRICE_PATTERN = "data/reports/**/metrics_price_2025*.csv"
SUMMARY_CSV = Path("data/reports/price_summary_2025.csv")


def load_price_metrics() -> pd.DataFrame:
    rows = []
    for path in glob.glob(PRICE_PATTERN, recursive=True):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Pominięto {path}: {exc}")
            continue
        df = df.copy()
        df["source_file"] = path
        if "model" not in df.columns:
            df["model"] = Path(path).stem.replace("metrics_price_", "")
        rows.append(df)
    if not rows:
        raise SystemExit("Brak wygenerowanych metryk. Uruchom najpierw modele.")
    merged = pd.concat(rows, ignore_index=True)
    required = {"mae", "rmse", "tolerance_accuracy", "model"}
    missing = required - set(merged.columns)
    if missing:
        raise SystemExit(f"Brak wymaganych kolumn: {missing}")
    return merged


def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("mae")
    df["rank_mae"] = range(1, len(df) + 1)
    return df


def summarize(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(subset=["model", "mae", "rmse", "source_file"])
    ranked = rank_models(df)
    os.makedirs(SUMMARY_CSV.parent, exist_ok=True)
    ranked.to_csv(SUMMARY_CSV, index=False)
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]

    print("=== Ranking modeli (MAE rosnąco) ===")
    cols = ["rank_mae", "model", "mae", "rmse", "mape", "tolerance_accuracy", "source_file"]
    def fmt(value):
        if isinstance(value, float):
            if pd.isna(value):
                return "nan"
            return f"{value:.4f}"
        return str(value)

    print(ranked[cols].to_string(index=False, formatters={c: fmt for c in cols}))
    print()
    print("Najlepszy model:")
    print(json.dumps(best[cols].to_dict(), indent=2, default=str))
    print("\nNajgorszy model:")
    print(json.dumps(worst[cols].to_dict(), indent=2, default=str))


def main() -> None:
    df = load_price_metrics()
    summarize(df)


if __name__ == "__main__":
    main()
