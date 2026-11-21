"""Lightweight data loader for Project 2 (BRPMM) batch time series.

Each batch (``batch_id``) is one multivariate time series; calibration batches
carry an ``accepted`` / ``rejected`` label. Column names come from
``team-workspace/variable_names/process_{A,B}_variables.csv``.

Sources, in order of preference:
  1. ``team-workspace/exports/process_{p}/{split}.parquet`` (fast, float32, has ``t``)
  2. ``input-data/Project 2 - BRPMM/process {p}/{split} data/{split}_data.csv``

Override locations with env vars ``BRPMM_WORKSPACE`` and ``BRPMM_INPUT``.

Quick use::

    from data_loader import load_dataset
    ds = load_dataset("A", "calibration")      # padded 3-D arrays
    ds.X.shape, ds.y.shape                     # (n_batches, T, n_features), (n_batches,)

    df = load_long("B", "test")                # long DataFrame: batch_id, t, <features>
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

Process = Literal["A", "B"]
Split = Literal["calibration", "test"]

_ROOT = Path(__file__).resolve().parents[2]  # .../work
WORKSPACE = Path(os.environ.get("BRPMM_WORKSPACE", _ROOT / "team-workspace"))
INPUT_DIR = Path(os.environ.get("BRPMM_INPUT", _ROOT / "input-data" / "Project 2 - BRPMM"))

LABEL_MAP = {"accepted": 0, "rejected": 1}


def _check(process: str, split: str) -> tuple[str, str]:
    process = process.upper()
    if process not in ("A", "B"):
        raise ValueError(f"process must be 'A' or 'B', got {process!r}")
    if split not in ("calibration", "test"):
        raise ValueError(f"split must be 'calibration' or 'test', got {split!r}")
    return process, split


# --------------------------------------------------------------------------- #
# Raw loaders
# --------------------------------------------------------------------------- #
def load_variables(process: Process) -> list[str]:
    """Feature names in column order for a process."""
    process, _ = _check(process, "test")
    path = WORKSPACE / "variable_names" / f"process_{process}_variables.csv"
    if path.exists():
        return pd.read_csv(path).sort_values("column_index")["variable_name"].tolist()
    xlsx = INPUT_DIR / f"process {process}" / "calibration data" / "variable_names.xlsx"
    return pd.read_excel(xlsx, header=None)[0].astype(str).tolist()


def load_long(
    process: Process,
    split: Split = "calibration",
    features: Optional[list[str]] = None,
    prefer_parquet: bool = True,
) -> pd.DataFrame:
    """Long-format DataFrame: ``batch_id``, ``t`` (0-based step), then features (float32)."""
    process, split = _check(process, split)
    variables = load_variables(process)
    cols = features or variables
    unknown = set(cols) - set(variables)
    if unknown:
        raise KeyError(f"Unknown features for process {process}: {sorted(unknown)}")

    pq = WORKSPACE / "exports" / f"process_{process}" / f"{split}.parquet"
    if prefer_parquet and pq.exists():
        df = pd.read_parquet(pq, columns=["batch_id", "t", *cols])
        df["batch_id"] = df["batch_id"].astype(str)
    else:
        csv = INPUT_DIR / f"process {process}" / f"{split} data" / f"{split}_data.csv"
        df = pd.read_csv(csv, usecols=["batch_id", *cols],
                         dtype={c: np.float32 for c in cols})
        df.insert(1, "t", df.groupby("batch_id", sort=False).cumcount())
    return df[["batch_id", "t", *cols]].reset_index(drop=True)


def load_labels(process: Process) -> pd.Series:
    """Calibration labels as a Series indexed by batch_id (1 = rejected, 0 = accepted)."""
    process, _ = _check(process, "calibration")
    path = INPUT_DIR / f"process {process}" / "calibration data" / "calibration_labels.csv"
    lab = pd.read_csv(path)
    return lab.set_index("batch_id")["label"].str.strip().str.lower().map(LABEL_MAP).astype(int)


# --------------------------------------------------------------------------- #
# Model-ready arrays
# --------------------------------------------------------------------------- #
@dataclass
class BatchDataset:
    X: np.ndarray            # (n_batches, T, n_features), float32, padded
    lengths: np.ndarray      # (n_batches,) true length of each series (after truncation)
    batch_ids: list[str]
    feature_names: list[str]
    y: Optional[np.ndarray] = None   # (n_batches,) int; None for the test split
    align: Literal["start", "end"] = "start"   # where the real data sits in each row

    @property
    def mask(self) -> np.ndarray:
        """Boolean (n_batches, T): True where a timestep is real data."""
        steps = np.arange(self.X.shape[1])[None, :]
        if self.align == "end":
            return steps >= self.X.shape[1] - self.lengths[:, None]
        return steps < self.lengths[:, None]

    def __len__(self) -> int:
        return len(self.batch_ids)

    def subset(self, idx) -> "BatchDataset":
        idx = np.asarray(idx)
        return BatchDataset(self.X[idx], self.lengths[idx],
                            [self.batch_ids[i] for i in idx], self.feature_names,
                            None if self.y is None else self.y[idx], self.align)

    def to_torch(self):
        """Return a ``torch.utils.data.TensorDataset`` of (X, lengths[, y])."""
        import torch
        from torch.utils.data import TensorDataset
        tensors = [torch.from_numpy(self.X), torch.from_numpy(self.lengths)]
        if self.y is not None:
            tensors.append(torch.from_numpy(self.y))
        return TensorDataset(*tensors)


def to_arrays(
    df: pd.DataFrame,
    max_len: Optional[int] = None,
    pad_value: float = 0.0,
    align: Literal["start", "end"] = "start",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pivot a long DataFrame into a padded 3-D array.

    ``max_len`` truncates longer series (keeping the first steps for
    ``align='start'``, the last steps for ``align='end'``). Padding is added
    after the data for ``'start'`` and before it for ``'end'``.
    """
    feats = [c for c in df.columns if c not in ("batch_id", "t")]
    df = df.sort_values(["batch_id", "t"], kind="stable")
    ids, starts, counts = np.unique(df["batch_id"].to_numpy(), return_index=True, return_counts=True)
    values = df[feats].to_numpy(np.float32)

    T = int(max_len or counts.max())
    X = np.full((len(ids), T, len(feats)), pad_value, dtype=np.float32)
    lengths = np.minimum(counts, T).astype(np.int64)
    for i, (s, n, L) in enumerate(zip(starts, counts, lengths)):
        seg = values[s:s + n]
        seg = seg[:L] if align == "start" else seg[n - L:]
        if align == "start":
            X[i, :L] = seg
        else:
            X[i, T - L:] = seg
    return X, lengths, ids.astype(str).tolist()


def load_dataset(
    process: Process,
    split: Split = "calibration",
    features: Optional[list[str]] = None,
    max_len: Optional[int] = None,
    pad_value: float = 0.0,
    align: Literal["start", "end"] = "start",
) -> BatchDataset:
    """Load one process/split as padded arrays, with labels for calibration."""
    df = load_long(process, split, features)
    X, lengths, ids = to_arrays(df, max_len=max_len, pad_value=pad_value, align=align)
    y = None
    if split == "calibration":
        labels = load_labels(process)
        missing = set(ids) - set(labels.index)
        if missing:
            raise ValueError(f"No label for batches: {sorted(missing)[:5]}...")
        y = labels.loc[ids].to_numpy(np.int64).copy()
    return BatchDataset(X, lengths, ids, list(df.columns[2:]), y, align)


# --------------------------------------------------------------------------- #
# Normalisation (fit on calibration, apply to test)
# --------------------------------------------------------------------------- #
@dataclass
class Standardizer:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    def fit(self, ds: BatchDataset) -> "Standardizer":
        # (n_real_steps, n_features), ignores padding; float64 avoids float32 accumulation error
        real = ds.X[ds.mask].astype(np.float64)
        self.mean = real.mean(0)
        self.std = real.std(0)
        self.std[self.std < 1e-8] = 1.0          # constant channels stay constant
        return self

    def transform(self, ds: BatchDataset, pad_value: float = 0.0) -> BatchDataset:
        X = (ds.X - self.mean) / self.std
        X[~ds.mask] = pad_value
        return BatchDataset(X.astype(np.float32), ds.lengths, ds.batch_ids, ds.feature_names,
                            ds.y, ds.align)


if __name__ == "__main__":
    for p in ("A", "B"):
        for s in ("calibration", "test"):
            ds = load_dataset(p, s)
            extra = f", rejected={int(ds.y.sum())}/{len(ds)}" if ds.y is not None else ""
            print(f"process {p} {s:<11} X={ds.X.shape} "
                  f"len[min/max]={ds.lengths.min()}/{ds.lengths.max()}{extra}")
