import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..common import evaluate_tolerance, load_or_fetch, prepare_close_series, regression_metrics
from ..features import FeatureConfig, make_features
from .utils import SequenceDataset, apply_standardization, make_sequences, standardize_dataset


REPORT_DIR = "data/reports/dl"
EVAL_YEAR = 2025
METRICS_CSV = os.path.join(REPORT_DIR, f"metrics_price_{EVAL_YEAR}_transformer.csv")


@dataclass(frozen=True)
class TrainingConfig:
    window: int = 30
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    epochs: int = 70
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 6


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.device)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cfg: TrainingConfig,
    ) -> None:
        super().__init__()
        self.input_linear = nn.Linear(input_dim, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.positional = PositionalEncoding(cfg.d_model, max_len=cfg.window + 5)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(x)
        x = self.positional(x)
        x = self.encoder(x)
        out = x[:, -1, :]
        return self.head(out)


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


def train_transformer(cfg: TrainingConfig, feature_cfg: FeatureConfig) -> tuple[pd.Series, pd.Series, dict]:
    train_ds, val_ds, test_ds, close = build_dataset(cfg.window, feature_cfg)

    train_std, mean, std = standardize_dataset(train_ds)
    val_std = apply_standardization(val_ds, mean, std)
    test_std = apply_standardization(test_ds, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(train_std._X.shape[-1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_std, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_std, batch_size=cfg.batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    patience_left = cfg.patience

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
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
            preds.append(model(xb).cpu().numpy())
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
        "model": "transformer_price",
        "best_val_loss": float(best_val),
    })

    return y_pred_price, y_true, metrics


def save_results(y_pred: pd.Series, y_true: pd.Series, metrics: dict) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(METRICS_CSV, index=False)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label=f"Rzeczywiste {EVAL_YEAR}")
    plt.plot(y_pred.index, y_pred.values, label="Prognoza TRANSFORMER")
    plt.title(f"USD/PLN — Transformer wartość {EVAL_YEAR} (Dokładność: {metrics['tolerance_accuracy']*100:.2f}%)")
    plt.xlabel("Data")
    plt.ylabel("Zamknięcie")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(REPORT_DIR, f"price_{EVAL_YEAR}_transformer.png")
    plt.savefig(out_png, dpi=150)
    print(f"Wykres zapisano: {out_png}")


def main():
    feature_cfg = FeatureConfig()
    cfg = TrainingConfig()
    print("Trening modelu Transformer...")
    y_pred, y_true, metrics = train_transformer(cfg, feature_cfg)
    save_results(y_pred, y_true, metrics)
    print(
        f"Transformer — MAE: {metrics['mae']:.4f}  RMSE: {metrics['rmse']:.4f}  Tolerancja: {metrics['tolerance_accuracy']*100:.2f}%"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
