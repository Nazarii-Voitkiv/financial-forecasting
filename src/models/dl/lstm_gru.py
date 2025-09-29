import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..common import evaluate_tolerance, load_or_fetch, prepare_close_series, regression_metrics
from ..features import FeatureConfig, make_features
from .utils import SequenceDataset, make_sequences, standardize_dataset, apply_standardization


REPORT_DIR = "data/reports/dl"
EVAL_YEAR = 2025
METRICS_CSV = os.path.join(REPORT_DIR, f"metrics_price_{EVAL_YEAR}_rnn.csv")


@dataclass(frozen=True)
class TrainingConfig:
    window: int = 30
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    epochs: int = 80
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 8


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell: Literal["lstm", "gru"],
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


def build_dataset(window: int, feature_config: Optional[FeatureConfig] = None) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset, pd.Series]:
    df = load_or_fetch()
    close = prepare_close_series(df, fill_method="ffill")
    X_raw = make_features(close, feature_config)
    target = np.log(close.shift(-1)) - np.log(close)
    data = pd.concat([X_raw, target.rename("target")], axis=1).dropna()
    X = data.drop(columns=["target"])
    y = data["target"]

    seqs, targets, idx = make_sequences(X, y, window)
    train_mask = idx.year < EVAL_YEAR
    test_mask = idx.year == EVAL_YEAR

    last_year_mask = idx.year == (EVAL_YEAR - 1)
    if not last_year_mask.any():
        raise ValueError("Validation cutoff unavailable; insufficient historical data")
    val_cutoff = idx[last_year_mask].max()
    val_mask = (idx >= (val_cutoff - pd.Timedelta(days=120))) & train_mask
    train_core = train_mask & ~val_mask

    train_ds = SequenceDataset(seqs[train_core], targets[train_core], idx[train_core])
    val_ds = SequenceDataset(seqs[val_mask], targets[val_mask], idx[val_mask])
    test_ds = SequenceDataset(seqs[test_mask], targets[test_mask], idx[test_mask])

    return train_ds, val_ds, test_ds, close


def train_model(cell: Literal["lstm", "gru"], cfg: TrainingConfig, feature_cfg: FeatureConfig) -> tuple[pd.Series, pd.Series, dict]:
    train_ds, val_ds, test_ds, close = build_dataset(cfg.window, feature_cfg)

    train_std, mean, std = standardize_dataset(train_ds)
    val_std = apply_standardization(val_ds, mean, std)
    test_std = apply_standardization(test_ds, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceRegressor(
        input_dim=train_std._X.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        cell=cell,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_std, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_std, batch_size=cfg.batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    patience_left = cfg.patience

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
        val_loss /= len(val_std)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = model.state_dict()
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = DataLoader(test_std, batch_size=cfg.batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0).squeeze()

    y_pred_ret = pd.Series(preds, index=test_std.index)
    y_pred_price = close.loc[test_std.index] * np.exp(y_pred_ret)
    y_true = close.shift(-1).loc[test_std.index]

    acc, desc = evaluate_tolerance(y_true, y_pred_price, tol_abs=0.01, tol_pct=0.0)
    metrics = regression_metrics(y_true, y_pred_price)
    metrics.update({
        "tolerance_accuracy": float(acc),
        "tolerance": desc,
        "n": int(len(y_true)),
        "mode": "walk",
        "year": int(EVAL_YEAR),
        "model": f"{cell}_price",
        "best_val_loss": float(best_val),
    })

    return y_pred_price, y_true, metrics


def save_results(model_name: str, y_pred: pd.Series, y_true: pd.Series, metrics: dict) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    df_metrics = pd.DataFrame([metrics])
    if os.path.exists(METRICS_CSV):
        existing = pd.read_csv(METRICS_CSV)
        df_metrics = pd.concat([existing, df_metrics], ignore_index=True)
        df_metrics = df_metrics.drop_duplicates(subset=["model"], keep="last")
    df_metrics.to_csv(METRICS_CSV, index=False)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label=f"Rzeczywiste {EVAL_YEAR}")
    plt.plot(y_pred.index, y_pred.values, label=f"Prognoza {model_name.upper()}")
    plt.title(f"USD/PLN — {model_name.upper()} wartość {EVAL_YEAR} (Dokładność: {metrics['tolerance_accuracy']*100:.2f}%)")
    plt.xlabel("Data")
    plt.ylabel("Zamknięcie")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(REPORT_DIR, f"price_{EVAL_YEAR}_{model_name}.png")
    plt.savefig(out_png, dpi=150)
    print(f"Wykres zapisano: {out_png}")


def main():
    feature_cfg = FeatureConfig()
    cfg = TrainingConfig()
    for cell in ("lstm", "gru"):
        print(f"Trening modelu {cell.upper()}...")
        y_pred, y_true, metrics = train_model(cell, cfg, feature_cfg)
        save_results(cell, y_pred, y_true, metrics)
        print(
            f"{cell.upper()} — MAE: {metrics['mae']:.4f}  RMSE: {metrics['rmse']:.4f}  Tolerancja: {metrics['tolerance_accuracy']*100:.2f}%"
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
