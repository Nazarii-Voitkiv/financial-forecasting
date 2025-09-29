"""Utilities for deep-learning forex forecasters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceConfig:
    window: int = 30


class SequenceDataset(Dataset):
    """Sliding-window dataset over tabular feature frames."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        index: pd.Index,
    ) -> None:
        self._X = torch.from_numpy(sequences).float()
        self._y = torch.from_numpy(targets).float().unsqueeze(-1)
        self._index = index

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y[idx]

    @property
    def index(self) -> pd.Index:
        return self._index


def make_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    window: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    if window < 1:
        raise ValueError("Window size must be positive")
    if len(X) != len(y):
        raise ValueError("Features and target must align")

    values = X.values.astype(np.float32)
    targets = y.values.astype(np.float32)
    idx = X.index

    seqs = []
    ys = []
    out_idx = []
    for i in range(window - 1, len(X)):
        seq = values[i - window + 1 : i + 1]
        target = targets[i]
        if np.isnan(seq).any() or np.isnan(target):
            continue
        seqs.append(seq)
        ys.append(target)
        out_idx.append(idx[i])

    if not seqs:
        raise ValueError("Sequence generation produced no samples; increase data window")

    return np.stack(seqs), np.array(ys), pd.Index(out_idx)


def train_val_split_by_time(
    dataset: SequenceDataset,
    val_index: pd.Index,
) -> tuple[SequenceDataset, SequenceDataset]:
    mask = dataset.index.isin(val_index)
    val_idx = np.where(mask)[0]
    train_idx = np.where(~mask)[0]
    if not len(train_idx):
        raise ValueError("Empty training set after split")
    if not len(val_idx):
        raise ValueError("Empty validation set after split")

    def sub_dataset(idxs: np.ndarray) -> SequenceDataset:
        return SequenceDataset(
            dataset._X[idxs].numpy(),
            dataset._y[idxs].squeeze(-1).numpy(),
            dataset.index[idxs],
        )

    return sub_dataset(train_idx), sub_dataset(val_idx)


def standardize_dataset(train_ds: SequenceDataset) -> tuple[SequenceDataset, torch.Tensor, torch.Tensor]:
    data = train_ds._X
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    std[std < 1e-6] = 1.0

    standardized = SequenceDataset(
        ((train_ds._X - mean) / std).cpu().numpy(),
        train_ds._y.squeeze(-1).cpu().numpy(),
        train_ds.index,
    )

    return standardized, mean.squeeze(0).cpu(), std.squeeze(0).cpu()


def apply_standardization(ds: SequenceDataset, mean: torch.Tensor, std: torch.Tensor) -> SequenceDataset:
    mean = mean.unsqueeze(0)
    std = std.unsqueeze(0)
    return SequenceDataset(
        ((ds._X - mean) / std).cpu().numpy(),
        ds._y.squeeze(-1).cpu().numpy(),
        ds.index,
    )
