"""Early time series classification (eTSC) for BRPMM batches -- one module.

Hard requirement: NEVER release a bad batch. The model therefore never releases
anything; it can only ABORT a running batch or HOLD it for full QC. Positive = rejected.

  1. checkpoints  decision points in ELAPSED STEPS (a batch's final length is unknown live)
  2. prefixes     per-batch features from only the steps seen so far, one model per checkpoint
  3. scores       out-of-fold score matrix S (batch x checkpoint, higher = more anomalous),
                  optionally with exact per-feature contributions (PCA T² + Q monitor)
  4. zones        abort limit (above every good batch) / release limit (below every bad batch),
                  always fitted on other batches than they're tested on
  5. evaluation   error counts with 95% upper bounds, cost model, sweeps
  6. explanation  which sensors drove each flag, in which direction, and since when

Usage::

    df = load_long("A", "calibration"); y = load_labels_fixed("A"); L = batch_lengths(df)
    sensors = [c for c in df.columns if c not in ("batch_id", "t")]
    feats = build_prefix_table(df, make_checkpoints(df))
    S = cv_score_matrix(feats, y)                       # or: S, C, Z = cv_score_matrix(..., contributions=True)
    earliness_curve(S, y, L)                            # AUC vs time observed
    release_sweep(S, y, L); margin_sweep(S, y, L)       # safety / abort trade-offs
    dec = nested_limits(S, y, L, margin=4)[0]; cost_table(dec, y, L)
    SC = sensor_contributions(C, sensors); explain_batches(dec, SC, Z, sensors, y)

CLI smoke test:  python eTSC.py [A|B]
"""

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from data_loader import load_labels, load_long
from evaluation import BORDERLINE, summarize
from features import build_features, drop_constant
from models import pca_monitor

GROUPS = ("summary", "trend")          # beat summary+trend+dynamics+segments in our tests
LENGTH = "batch length so far"         # pseudo-sensor for the n_timesteps feature


# =========================================================================== #
# 1-2. Data, checkpoints, prefix features
# =========================================================================== #
def batch_lengths(df: pd.DataFrame) -> pd.Series:
    return df.groupby("batch_id").size().sort_index()


def load_labels_fixed(process: str) -> pd.Series:
    """Labels (1 = rejected) with the brief's borderline batches A_101-A_104 set to rejected."""
    y = load_labels(process).copy()
    y[y.index.isin(BORDERLINE)] = 1
    return y


def make_checkpoints(df: pd.DataFrame, n: int = 10, ref: Union[str, float] = "median",
                     mode: Literal["ref", "steps"] = "ref", min_steps: int = 10) -> list[int]:
    """Steps at 1/n, 2/n, ... of a reference length (default: median batch length),
    plus the longest batch so that every batch is eventually seen in full."""
    L = batch_lengths(df)
    L_max = int(L.max())
    base = L_max if mode == "steps" else (float(getattr(L, ref)()) if isinstance(ref, str) else float(ref))
    cps = [int(round(f * base)) for f in np.linspace(1 / n, 1, n)]
    return sorted({max(min_steps, c) for c in cps if c <= L_max} | {L_max})


def build_prefix_table(df: pd.DataFrame, checkpoints: list[int], groups=GROUPS,
                       n_segments: int = 4) -> dict[int, pd.DataFrame]:
    """``{checkpoint: one row per batch from steps t < checkpoint}``. Features are per batch
    and label-free, so building them once for all batches leaks nothing."""
    return {cp: build_features(df, groups=groups, prefix_steps=cp, n_segments=n_segments)
            for cp in checkpoints}


# =========================================================================== #
# 3. Per-checkpoint models and out-of-fold scores
# =========================================================================== #
@dataclass
class PrefixModel:
    model: object
    columns: pd.Index
    fill: pd.Series
    norm: tuple[float, float]           # median / MAD of training good-batch scores

    def prep(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.columns).fillna(self.fill).fillna(0.0)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Anomaly score normalised so one limit means the same at every checkpoint."""
        med, scale = self.norm
        return (-self.model.score_samples(self.prep(X)) - med) / scale


def fit_prefix_model(model, X: pd.DataFrame, y: pd.Series) -> PrefixModel:
    """One-class: fit on the good batches in X only."""
    Xg = drop_constant(X[y.loc[X.index].to_numpy() == 0])
    pm = PrefixModel(clone(model), Xg.columns, Xg.median(), (0.0, 1.0))
    Xf = pm.prep(Xg)
    pm.model.fit(Xf)
    s = -pm.model.score_samples(Xf)
    med = float(np.median(s))
    pm.norm = (med, float(np.median(np.abs(s - med)) * 1.4826 + 1e-9))
    return pm


def fit_prefix_models(feats: dict, y: pd.Series, model=None) -> dict:
    """Final per-checkpoint models on all calibration batches (to score the test set)."""
    model = model if model is not None else pca_monitor()
    return {cp: fit_prefix_model(model, X, y) for cp, X in feats.items()}


def score_prefixes(fitted: dict, feats: dict) -> pd.DataFrame:
    return pd.DataFrame({cp: pd.Series(fitted[cp].score(X), index=X.index) for cp, X in feats.items()})


def cv_score_matrix(feats: dict, y: pd.Series, model=None, n_splits: int = 5,
                    random_state: int = 0, contributions: bool = False):
    """Out-of-fold score for every (batch, checkpoint); a batch's whole row comes from
    models that never saw it. Default model: PCA T² + Q.

    ``contributions=True`` (needs a ``models.pca_monitor`` pipeline) also returns C and Z,
    ``{checkpoint: batch x feature}``: exact contributions to the score, and z-scores vs
    the fold's good batches (sign = above / below normal).
    """
    model = model if model is not None else pca_monitor()
    cps = list(feats)
    ids = feats[cps[0]].index
    yy = y.loc[ids]
    S = pd.DataFrame(np.nan, index=ids, columns=cps)
    C, Z = {cp: [] for cp in cps}, {cp: [] for cp in cps}
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=random_state).split(ids, yy):
        for j, cp in enumerate(cps):
            X = feats[cp]
            pm = fit_prefix_model(model, X.iloc[tr], yy.iloc[tr])
            S.iloc[te, j] = pm.score(X.iloc[te])
            if contributions:
                Xte = pm.prep(X.iloc[te])
                z = pm.model[0].transform(Xte)
                C[cp].append(pd.DataFrame(pm.model[-1].contributions(z), Xte.index, pm.columns))
                Z[cp].append(pd.DataFrame(z, Xte.index, pm.columns))
    if not contributions:
        return S
    stack = lambda d: {cp: pd.concat(v).fillna(0.0).loc[ids] for cp, v in d.items()}
    return S, stack(C), stack(Z)


# =========================================================================== #
# 4. Zones: abort / hold / release-eligible
# =========================================================================== #
def _final_index(S: pd.DataFrame, lengths: Optional[pd.Series]) -> np.ndarray:
    """Column where each batch must be decided: first checkpoint at or after its end."""
    last = np.full(len(S), S.shape[1] - 1)
    if lengths is None:
        return last
    done = np.asarray(S.columns, float)[None, :] >= lengths.loc[S.index].to_numpy()[:, None]
    return np.where(done.any(1), done.argmax(1), last)


def control_limits(S: pd.DataFrame, y: pd.Series, q: float = 1.0, margin: float = 0.0) -> pd.Series:
    """Abort limit per checkpoint: q-quantile (default max) of good-batch scores + margin."""
    return S[y.loc[S.index].to_numpy() == 0].quantile(q) + margin


def trigger_limits(S: pd.DataFrame, limits: pd.Series, lengths: Optional[pd.Series] = None) -> pd.DataFrame:
    """Abort at the first checkpoint over the limit (pred=1); else run to the end (pred=0)."""
    V, final = S.to_numpy(), _final_index(S, lengths)
    over = (V > limits.loc[S.columns].to_numpy()[None, :]) & (np.arange(V.shape[1])[None, :] <= final[:, None])
    idx = np.where(over.any(1), over.argmax(1), final)
    return pd.DataFrame({"t_decision": np.asarray(S.columns)[idx], "pred": over.any(1).astype(int)}, S.index)


def _folds(S, y, n_splits, seed):
    return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(S.index, y.loc[S.index])


def nested_limits(S: pd.DataFrame, y: pd.Series, lengths: pd.Series, q: float = 1.0, margin: float = 0.0,
                  n_splits: int = 5, seeds: Iterable[int] = (0, 1, 2)) -> list[pd.DataFrame]:
    """Abort decisions with limits fitted on the other folds; one DataFrame per CV seed."""
    yy = y.loc[S.index]
    return [pd.concat([trigger_limits(S.iloc[te], control_limits(S.iloc[tr], yy.iloc[tr], q, margin), lengths)
                       for tr, te in _folds(S, y, n_splits, s)]).loc[S.index] for s in seeds]


def nested_zones(S: pd.DataFrame, y: pd.Series, lengths: pd.Series, release_margin: float = 0.0,
                 abort_margin: float = 0.0, n_splits: int = 5, seeds: Iterable[int] = (0, 1, 2)) -> list[pd.DataFrame]:
    """Zone per batch: abort (over abort limit at any checkpoint), release (final score below
    the lowest bad batch's final score minus ``release_margin``), else hold."""
    yy = y.loc[S.index]
    F = pd.Series(S.to_numpy()[np.arange(len(S)), _final_index(S, lengths)], S.index)
    runs = []
    for seed in seeds:
        parts = []
        for tr, te in _folds(S, y, n_splits, seed):
            ab = trigger_limits(S.iloc[te], control_limits(S.iloc[tr], yy.iloc[tr], margin=abort_margin), lengths)
            rel_lim = F.iloc[tr][yy.iloc[tr].to_numpy() == 1].min() - release_margin
            zone = np.where(ab["pred"] == 1, "abort", np.where(F.iloc[te] < rel_lim, "release", "hold"))
            parts.append(pd.DataFrame({"t_decision": ab["t_decision"], "zone": zone}, ab.index))
        runs.append(pd.concat(parts).loc[S.index])
    return runs


# =========================================================================== #
# 5. Evaluation
# =========================================================================== #
def upper_bound(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a rate after k events in n (0 in n ~ 3/n)."""
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def earliness_curve(S: pd.DataFrame, y: pd.Series, lengths: Optional[pd.Series] = None, **kw) -> pd.DataFrame:
    """``evaluation.summarize`` per checkpoint (read ROC / PR AUC)."""
    yy, rows = y.loc[S.index], {}
    for cp in S.columns:
        rows[cp] = summarize(S.loc[yy == 0, cp], S.loc[yy == 1, cp], **kw)
        if lengths is not None:
            L = lengths.loc[S.index]
            rows[cp]["mean observed fraction"] = float(np.minimum(cp, L).div(L).mean())
    return pd.DataFrame(rows).T.rename_axis("checkpoint")


def limit_metrics(dec: pd.DataFrame, y: pd.Series, lengths: pd.Series) -> dict:
    """Abort-side metrics: good batches wrongly aborted, bad batches stopped, how early."""
    yt, yp = y.loc[dec.index].to_numpy(), dec["pred"].to_numpy()
    L = lengths.loc[dec.index].to_numpy()
    obs = np.minimum(dec["t_decision"].to_numpy(), L) / L
    tp, fp = (yt == 1) & (yp == 1), (yt == 0) & (yp == 1)
    n_good, n_bad = int((yt == 0).sum()), int((yt == 1).sum())
    return {"false alarms (good batches aborted)": int(fp.sum()),
            "false alarm rate, 95% upper bound": upper_bound(int(fp.sum()), n_good),
            "recall": tp.sum() / n_bad,
            "stopped at (mean fraction of batch)": float(obs[tp].mean()) if tp.any() else np.nan,
            "stopped at (median step)": float(np.median(dec["t_decision"].to_numpy()[tp])) if tp.any() else np.nan}


def zone_metrics(z: pd.DataFrame, y: pd.Series) -> dict:
    """Release-side metrics. The number that must be 0: BAD batches released."""
    yt, zone = y.loc[z.index].to_numpy(), z["zone"].to_numpy()
    n_good, n_bad = int((yt == 0).sum()), int((yt == 1).sum())
    bad_rel = int(((yt == 1) & (zone == "release")).sum())
    frac = lambda lab, zn: ((yt == lab) & (zone == zn)).sum() / (n_bad if lab else n_good)
    return {"BAD batches released": bad_rel,
            "bad-release rate, 95% upper bound": upper_bound(bad_rel, n_bad),
            "good batches released": frac(0, "release"), "good batches held for QC": frac(0, "hold"),
            "good batches aborted": int(((yt == 0) & (zone == "abort")).sum()),
            "bad batches aborted early": frac(1, "abort"), "bad batches held for QC": frac(1, "hold")}


def cost_table(dec: pd.DataFrame, y: pd.Series, lengths: pd.Series,
               value_ratios: Iterable[float] = (1, 3, 10, 30, 100)) -> pd.DataFrame:
    """Net benefit of aborting vs running everything to the end. Unit = run cost of a
    median-length batch (cost ~ run time). V = cost of losing one good batch in that unit.
    ``.attrs["break-even value"]``: above this V, wrong aborts cost more than they save."""
    yt, yp = y.loc[dec.index].to_numpy(), dec["pred"].to_numpy().astype(bool)
    L = lengths.loc[dec.index].to_numpy().astype(float)
    ref = float(np.median(L))
    saved = np.where(yp, (L - np.minimum(dec["t_decision"].to_numpy(), L)) / ref, 0.0).sum()
    n_fp = int(((yt == 0) & yp).sum())
    out = pd.DataFrame([{"value of a good batch (x run cost)": V, "run cost saved": saved,
                         "good batches lost": n_fp, "net benefit": saved - V * n_fp,
                         "net benefit, % of total run cost": 100 * (saved - V * n_fp) / (L.sum() / ref)}
                        for V in value_ratios]).set_index("value of a good batch (x run cost)")
    out.attrs["break-even value"] = saved / n_fp if n_fp else np.inf
    return out


def margin_sweep(S, y, lengths, margins=(0, 1, 2, 4, 8, 16, 32), seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Abort side: false aborts / recall / savings as the abort-limit margin grows."""
    rows = {}
    for m in margins:
        runs = nested_limits(S, y, lengths, margin=m, seeds=seeds)
        r = pd.DataFrame([limit_metrics(d, y, lengths) for d in runs]).mean()
        r["run cost saved (median-batch units)"] = np.mean(
            [cost_table(d, y, lengths, [0])["run cost saved"].iloc[0] for d in runs])
        rows[m] = r
    return pd.DataFrame(rows).T.rename_axis("margin")


def release_sweep(S, y, lengths, margins=(0, 0.5, 1, 2, 4, 8), abort_margin=0.0,
                  seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Release side: margin needed before no bad batch is released in any seed."""
    rows = {}
    for m in margins:
        met = pd.DataFrame([zone_metrics(z, y) for z in
                            nested_zones(S, y, lengths, m, abort_margin, seeds=seeds)])
        rows[m] = met.mean()
        rows[m]["BAD batches released (worst seed)"] = met["BAD batches released"].max()
    return pd.DataFrame(rows).T.rename_axis("release margin")


# =========================================================================== #
# 6. Explanation (needs cv_score_matrix(..., contributions=True))
# =========================================================================== #
def sensor_of(feature: str, sensors: list[str]) -> str:
    """Feature column (e.g. ``temperature_1_mean``) -> its sensor."""
    hits = [s for s in sensors if feature == s or feature.startswith(s + "_")]
    return max(hits, key=len) if hits else LENGTH


def sensor_contributions(C: dict, sensors: list[str]) -> dict:
    """``{checkpoint: batch x sensor}``: contributions summed per sensor."""
    cols = list(sensors) + [LENGTH]
    return {cp: c.T.groupby([sensor_of(f, sensors) for f in c.columns]).sum().T
            .reindex(columns=cols, fill_value=0.0) for cp, c in C.items()}


def normal_range(SC: dict, y: pd.Series, exclude: Optional[str] = None) -> dict:
    """Per checkpoint and sensor: largest contribution in any good batch other than ``exclude``."""
    return {cp: sc[(y.loc[sc.index].to_numpy() == 0) & (sc.index != exclude)].max() for cp, sc in SC.items()}


def explain_batch(b: str, t: int, SC: dict, Z: dict, y: pd.Series, sensors: list[str], top: int = 3) -> dict:
    """Top sensors behind batch ``b``'s score at checkpoint ``t``: share of the score, most
    extreme feature (sd from normal), first checkpoint outside every other good batch."""
    normal = normal_range(SC, y, exclude=b)
    sc, z = SC[t].loc[b], Z[t].loc[b]
    feat_sensor = np.array([sensor_of(f, sensors) for f in z.index])
    out = {"batch_id": b, "decided at step": t}
    for k, s in enumerate(sc.sort_values(ascending=False).index[:top], 1):
        fz = z[feat_sensor == s]
        out[f"#{k} sensor"] = s
        out[f"#{k} share"] = sc[s] / sc.sum() if sc.sum() else np.nan
        if len(fz):
            f = fz.abs().idxmax()
            out[f"#{k} evidence"] = f"{f} {abs(fz[f]):.1f} sd {'above' if fz[f] > 0 else 'below'} normal"
        out[f"#{k} abnormal since step"] = next(
            (cp for cp in SC if cp <= t and SC[cp].loc[b, s] > normal[cp][s]), None)
    return out


def explain_batches(dec: pd.DataFrame, SC: dict, Z: dict, sensors: list[str], y: pd.Series,
                    top: int = 3) -> pd.DataFrame:
    """One row per bad batch and per false alarm (missed ones explained at their end)."""
    yy, rows = y.loc[dec.index], []
    names = {(1, 1): "caught early", (1, 0): "FALSE ALARM", (0, 1): "missed (QC catches it)"}
    for b in dec.index:
        key = (int(dec.loc[b, "pred"]), int(yy[b]))
        if key in names:
            rows.append({"outcome": names[key], **explain_batch(b, dec.loc[b, "t_decision"], SC, Z, y, sensors, top)})
    return pd.DataFrame(rows).set_index("batch_id").sort_values(["outcome", "decided at step"])


def failure_modes(table: pd.DataFrame) -> pd.DataFrame:
    """How often each sensor is the main driver, per outcome."""
    out = table.groupby(["outcome", "#1 sensor"]).size().unstack(0, fill_value=0)
    return out.loc[out.sum(1).sort_values(ascending=False).index]


# =========================================================================== #
# Plots
# =========================================================================== #
def plot_earliness_curve(curve: pd.DataFrame, metrics=("ROC AUC", "PR AUC"), ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    for m in metrics:
        ax.plot(curve.index, curve[m].astype(float), marker="o", label=m)
    ax.set(xlabel="checkpoint (elapsed steps)", ylabel="metric", ylim=(0, 1.02),
           title="Early classification: performance vs time observed")
    ax.legend(loc="lower right")
    return ax


def plot_batch(b: str, SC: dict, y: pd.Series, dec: Optional[pd.DataFrame] = None, n_sensors: int = 8, ax=None):
    """Heatmap of sensor contribution / max in other good batches, per checkpoint (>1 = abnormal)."""
    import matplotlib.pyplot as plt
    normal = normal_range(SC, y, exclude=b)
    R = pd.DataFrame({cp: SC[cp].loc[b] / normal[cp].replace(0, np.nan) for cp in SC})
    R = R.loc[R.max(1).sort_values(ascending=False).index[:n_sensors]]
    ax = ax or plt.subplots(figsize=(9, 0.45 * len(R) + 1.5))[1]
    im = ax.imshow(np.log2(R.clip(lower=1e-3).to_numpy(float)), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(range(len(R)), R.index)
    ax.set_xticks(range(R.shape[1]), R.columns, rotation=45)
    ax.set_xlabel("checkpoint (elapsed steps)")
    if dec is not None:
        ax.axvline(list(SC).index(dec.loc[b, "t_decision"]), color="k", lw=2, ls="--")
    plt.colorbar(im, ax=ax, label="log2(contribution / max in other good batches)")
    ax.set_title(f"{b}: which sensors are outside the normal range, and when")
    return ax


# =========================================================================== #
# Smoke test
# =========================================================================== #
if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    P = sys.argv[1] if len(sys.argv) > 1 else "A"
    df = load_long(P, "calibration")
    y, L = load_labels_fixed(P), batch_lengths(df)
    sensors = [c for c in df.columns if c not in ("batch_id", "t")]
    feats = build_prefix_table(df, make_checkpoints(df))
    S, C, Z = cv_score_matrix(feats, y, contributions=True)
    print(earliness_curve(S, y, L)[["ROC AUC", "PR AUC", "mean observed fraction"]].astype(float).round(3))
    print(release_sweep(S, y, L).round(3))
    print(margin_sweep(S, y, L).round(3))
    dec = nested_limits(S, y, L, margin={"A": 4, "B": 32}[P], seeds=(0,))[0]
    print(explain_batches(dec, sensor_contributions(C, sensors), Z, sensors, y).iloc[:, :6])
