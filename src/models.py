"""Anomaly-detection models for batch quality (one-class: trained on accepted batches only).

Every sklearn-style model here exposes ``fit(X)`` and ``score_samples(X)`` where
**higher = more normal** (sklearn convention). ``evaluation.evaluate`` flips the sign
so that higher = more anomalous.

Tabular models (one row per batch, see ``features.py``)::

    from models import get_models
    models = get_models()          # name -> unfitted pipeline

Sequence model (LSTM autoencoder on fixed-length resampled series)::

    from models import to_fixed_length, train_ae, recon_error
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# --------------------------------------------------------------------------- #
# Custom detectors
# --------------------------------------------------------------------------- #
class KMeansDetector(BaseEstimator):
    """Anomaly score = distance to the nearest cluster centre."""

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.km_ = KMeans(self.n_clusters, n_init=10, random_state=0).fit(X)
        return self

    def score_samples(self, X):
        return -self.km_.transform(X).min(axis=1)  # higher = more normal


class FastABOD(BaseEstimator):
    """Angle-based outlier detection on k nearest neighbours."""

    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.X_ = np.asarray(X)
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X_)
        return self

    def score_samples(self, X):
        X = np.asarray(X)
        _, idx = self.nn_.kneighbors(X)
        out = []
        for x, nb in zip(X, idx):
            D = self.X_[nb] - x
            n2 = (D ** 2).sum(axis=1) + 1e-12
            i, j = np.triu_indices(len(D), k=1)
            vals = (D[i] * D[j]).sum(axis=1) / (n2[i] * n2[j])
            out.append(vals.var())
        return np.array(out)  # higher = more normal


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def get_models() -> dict:
    """Unfitted tabular anomaly detectors, each with its own scaling inside the pipeline."""
    return {
        "Isolation Forest": make_pipeline(
            StandardScaler(), IsolationForest(n_estimators=300, random_state=0)),
        "LOF": make_pipeline(
            StandardScaler(), LocalOutlierFactor(n_neighbors=10, novelty=True)),
        "Robust covariance": make_pipeline(
            StandardScaler(), PCA(n_components=10), EllipticEnvelope(random_state=0)),
        "One-class SVM": make_pipeline(
            StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        "One-class SVM (SGD)": make_pipeline(
            StandardScaler(),
            Nystroem(gamma=0.01, n_components=100, random_state=0),
            SGDOneClassSVM(nu=0.05, random_state=0)),
        "K-means": make_pipeline(StandardScaler(), KMeansDetector(n_clusters=3)),
        "ABOD": make_pipeline(StandardScaler(), FastABOD(n_neighbors=10)),
    }


# --------------------------------------------------------------------------- #
# LSTM autoencoder
# --------------------------------------------------------------------------- #
def to_fixed_length(g: pd.DataFrame, features: list[str], L: int = 100) -> np.ndarray:
    """Resample one batch to ``L`` steps by averaging ``L`` equal blocks. Returns (L, n_features)."""
    return np.stack([blk.mean(axis=0) for blk in np.array_split(g[features].to_numpy(), L)])


class LSTMAE(nn.Module):
    """Sequence autoencoder: encode the whole series to one hidden state, decode it back."""

    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(z)
        return self.out(y)


def train_ae(Xtr: np.ndarray, epochs: int = 100, hidden: int = 32, lr: float = 1e-3, batch_size: int = 16):
    """Train an LSTM autoencoder on (n_batches, L, n_features) standardized sequences."""
    torch.manual_seed(0)
    m = LSTMAE(Xtr.shape[2], hidden=hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    for _ in range(epochs):
        for i in torch.randperm(len(Xt)).split(batch_size):
            loss = ((m(Xt[i]) - Xt[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def recon_error(m, A: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error per sequence (higher = more anomalous)."""
    with torch.no_grad():
        At = torch.tensor(A, dtype=torch.float32)
        return ((m(At) - At) ** 2).mean(dim=(1, 2)).numpy()
