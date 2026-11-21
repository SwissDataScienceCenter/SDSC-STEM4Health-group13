"""Hand-made features: one row per batch, for tabular models.

Everything here takes the long DataFrame from ``data_loader.load_long``
(``batch_id``, ``t``, <sensor columns>) and returns a DataFrame indexed by
``batch_id``. No models live here.

Feature groups:
  summary   level and spread of each sensor (mean, std, min, max, first, last, ...)
  trend     overall drift: slope over normalised time, last - first
  dynamics  how jumpy the signal is: mean |step change|, std of step changes
  segments  summary stats per phase, splitting each batch into equal fractions
            of its own length (captures *when* things happen, for any length)

Typical use::

    from data_loader import load_long, load_labels
    from features import build_features, drop_constant

    train = build_features(load_long("A", "calibration"))
    test  = build_features(load_long("A", "test"))
    train, test = drop_constant(train, test)
    y = load_labels("A").loc[train.index]

    # early classification: only use the first 30% of every batch
    train_early = build_features(load_long("A", "calibration"), prefix_frac=0.3)
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd

KEYS = ("batch_id", "t")


def _sensors(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEYS]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(list(KEYS), kind="stable")


# --------------------------------------------------------------------------- #
# Truncation (early classification)
# --------------------------------------------------------------------------- #
def truncate(df: pd.DataFrame, steps: Optional[int] = None, frac: Optional[float] = None) -> pd.DataFrame:
    """Keep only the start of every batch.

    ``steps``: the first N steps of each batch.
    ``frac``:  the first fraction of each batch's own length (at least 1 step).
    """
    if (steps is None) == (frac is None):
        raise ValueError("give exactly one of steps= or frac=")
    if steps is not None:
        return df[df["t"] < steps]
    n = df.groupby("batch_id")["t"].transform("size")
    return df[df["t"] < np.maximum(np.ceil(n * frac), 1)]


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
def summary_features(
    df: pd.DataFrame,
    stats: Iterable[str] = ("mean", "std", "min", "max", "first", "last"),
    quantiles: Iterable[float] = (),
) -> pd.DataFrame:
    """Per-sensor summary stats plus ``n_timesteps``. Columns: ``<sensor>_<stat>``."""
    sensors = _sensors(df)
    g = _sorted(df).groupby("batch_id", sort=True)
    out = g[sensors].agg(list(stats))
    out.columns = [f"{s}_{stat}" for s, stat in out.columns]
    for q in quantiles:
        qs = g[sensors].quantile(q)
        qs.columns = [f"{s}_q{int(q * 100):02d}" for s in sensors]
        out = out.join(qs)
    out.insert(0, "n_timesteps", g.size())
    return out


def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Least-squares slope against time scaled to [0, 1] and ``last - first`` per sensor.

    Time is scaled per batch, so slopes are comparable between short and long batches
    (slope = total change over the batch implied by the linear fit).
    """
    sensors = _sensors(df)
    d = _sorted(df)
    g = d.groupby("batch_id", sort=True)
    n = g["t"].transform("size").to_numpy()
    tn = d["t"].to_numpy() / np.maximum(n - 1, 1)
    tc = tn - pd.Series(tn, index=d.index).groupby(d["batch_id"]).transform("mean").to_numpy()
    x = d[sensors].to_numpy(np.float64)
    xc = x - g[sensors].transform("mean").to_numpy(np.float64)

    prod = pd.DataFrame(xc * tc[:, None], columns=sensors, index=d.index)
    cov = prod.groupby(d["batch_id"], sort=True).sum()
    var = pd.Series(tc ** 2, index=d.index).groupby(d["batch_id"], sort=True).sum()
    slope = cov.div(var.replace(0, np.nan), axis=0)
    slope.columns = [f"{s}_slope" for s in sensors]

    delta = g[sensors].last() - g[sensors].first()
    delta.columns = [f"{s}_delta" for s in sensors]
    return slope.join(delta)


def dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step-to-step behaviour per sensor: mean |change| and std of change."""
    sensors = _sensors(df)
    d = _sorted(df)
    diff = d.groupby("batch_id", sort=True)[sensors].diff()
    diff["batch_id"] = d["batch_id"]
    g = diff.groupby("batch_id", sort=True)[sensors]
    absmean = g.agg(lambda s: s.abs().mean())
    absmean.columns = [f"{s}_diff_absmean" for s in sensors]
    dstd = g.std()
    dstd.columns = [f"{s}_diff_std" for s in sensors]
    return absmean.join(dstd)


def segment_features(
    df: pd.DataFrame,
    n_segments: int = 4,
    stats: Iterable[str] = ("mean", "std"),
) -> pd.DataFrame:
    """Summary stats per phase. Segment k covers fraction [k/n, (k+1)/n) of each batch.

    Columns: ``<sensor>_seg<k>_<stat>``.
    """
    sensors = _sensors(df)
    d = _sorted(df)
    n = d.groupby("batch_id")["t"].transform("size").to_numpy()
    seg = np.minimum((d["t"].to_numpy() * n_segments) // n, n_segments - 1)
    out = d.assign(_seg=seg).groupby(["batch_id", "_seg"], sort=True)[sensors].agg(list(stats))
    out = out.unstack("_seg")                     # columns: (sensor, stat, seg)
    out.columns = [f"{s}_seg{k}_{stat}" for s, stat, k in out.columns]
    return out


# --------------------------------------------------------------------------- #
# Combine
# --------------------------------------------------------------------------- #
def build_features(
    df: pd.DataFrame,
    groups: Iterable[str] = ("summary", "trend", "dynamics", "segments"),
    prefix_steps: Optional[int] = None,
    prefix_frac: Optional[float] = None,
    n_segments: int = 4,
) -> pd.DataFrame:
    """Build the chosen feature groups and join them on ``batch_id``.

    ``prefix_steps`` / ``prefix_frac`` truncate every batch first (early classification).
    """
    if prefix_steps is not None or prefix_frac is not None:
        df = truncate(df, steps=prefix_steps, frac=prefix_frac)
    makers = {
        "summary": summary_features,
        "trend": trend_features,
        "dynamics": dynamics_features,
        "segments": lambda d: segment_features(d, n_segments=n_segments),
    }
    unknown = set(groups) - set(makers)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}; choose from {list(makers)}")
    parts = [makers[g](df) for g in groups]
    return pd.concat(parts, axis=1).sort_index()


def drop_constant(train: pd.DataFrame, *others: pd.DataFrame, tol: float = 1e-12):
    """Drop columns that are constant (or all-NaN) in ``train``; apply the same to ``others``.

    Process B has several sensors that never change (e.g. ``feed_substate_concentration``).
    """
    keep = train.columns[(train.std(skipna=True) > tol).to_numpy()]
    out = [train[keep]] + [o[keep] for o in others]
    return out[0] if not others else tuple(out)
