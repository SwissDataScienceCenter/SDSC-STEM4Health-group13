"""Evaluation of one-class anomaly detectors on labelled calibration batches.

Models only ever learn from accepted batches. Each accepted batch is scored by a
model that never saw it; rejected batches are scored by every fold's model and the
scores are averaged. All scores returned here: **higher = more anomalous**.

Typical use::

    from models import get_models
    from evaluation import evaluate, summarize

    acc = labels == "accepted"
    acc_s, rej_s = evaluate(get_models()["LOF"], X, acc)
    summarize(acc_s, rej_s)
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold

from models import recon_error, to_fixed_length, train_ae

# Process A batches the brief describes as borderline REJECTS. Note: calibration_labels.csv has
# A_101 and A_103 as "accepted" (the notebook overrides them); A_102 is only in the test data.
BORDERLINE = ["A_101", "A_102", "A_103", "A_104"]


def evaluate(model, X: pd.DataFrame, acc_mask) -> tuple[pd.Series, pd.Series]:
    """Leave-one-accepted-batch-out scores for an sklearn-style detector.

    Returns ``(acc_scores, rej_scores)`` indexed by batch_id.
    """
    X_acc, X_rej = X[acc_mask], X[~acc_mask]
    acc_s = pd.Series(index=X_acc.index, dtype=float)
    rej_s = []
    for b in X_acc.index:
        m = clone(model).fit(X_acc.drop(index=b))
        acc_s[b] = -m.score_samples(X_acc.loc[[b]])[0]
        rej_s.append(-m.score_samples(X_rej))
    rej_s = pd.Series(np.mean(rej_s, axis=0), index=X_rej.index)
    return acc_s, rej_s


def evaluate_lstm_ae(
    df: pd.DataFrame,
    labels: pd.Series,
    features: list[str],
    L: int = 100,
    n_splits: int = 5,
    epochs: int = 100,
    verbose: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """K-fold (over accepted batches) scores from an LSTM autoencoder's reconstruction error.

    ``df``: long data with ``batch_id`` and the ``features`` columns.
    ``labels``: ``accepted`` / ``rejected`` indexed by batch_id; defines batch order.
    Standardization stats are computed on each fold's training batches only.
    """
    ids = labels.index
    S = np.stack([to_fixed_length(df[df["batch_id"] == b], features, L) for b in ids])
    lab = labels.to_numpy()
    acc_idx, rej_idx = np.where(lab == "accepted")[0], np.where(lab == "rejected")[0]

    acc_s = pd.Series(index=ids[acc_idx], dtype=float)
    rej_list = []
    for k, (tr, te) in enumerate(KFold(n_splits, shuffle=True, random_state=0).split(acc_idx)):
        if verbose:
            print("fold", k + 1)
        mu = S[acc_idx[tr]].mean(axis=(0, 1))
        sd = S[acc_idx[tr]].std(axis=(0, 1)) + 1e-8
        m = train_ae((S[acc_idx[tr]] - mu) / sd, epochs=epochs)
        acc_s.iloc[te] = recon_error(m, (S[acc_idx[te]] - mu) / sd)
        rej_list.append(recon_error(m, (S[rej_idx] - mu) / sd))

    rej_s = pd.Series(np.mean(rej_list, axis=0), index=ids[rej_idx])
    return acc_s, rej_s


def summarize(acc_s: pd.Series, rej_s: pd.Series, q: float = 0.95, borderline=BORDERLINE) -> dict:
    """Metrics for one model. Threshold = ``q`` quantile of accepted scores
    (so roughly ``1 - q`` false alarms). Positive class = rejected.
    """
    scores = pd.concat([acc_s, rej_s])
    y = scores.index.isin(rej_s.index).astype(int)
    thr = acc_s.quantile(q)
    pred = (scores > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    border = rej_s[rej_s.index.isin(borderline)]

    return {
        # threshold-free
        "ROC AUC": roc_auc_score(y, scores),
        "PR AUC": average_precision_score(y, scores),
        # at the chosen threshold
        "accuracy": accuracy_score(y, pred),
        "balanced accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall (sensitivity)": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "F1": f1_score(y, pred, zero_division=0),
        "F2 (recall-weighted)": fbeta_score(y, pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "false alarm rate": fp / (fp + tn),
        "miss rate": fn / (fn + tp),
        # raw counts
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "borderline caught": f"{(border > thr).sum()}/{len(border)}",
    }


def results_table(all_scores: dict, **kw) -> pd.DataFrame:
    """``{name: (acc_s, rej_s)}`` -> one row of ``summarize`` metrics per model, best ROC AUC first."""
    res = {name: summarize(a, r, **kw) for name, (a, r) in all_scores.items()}
    return pd.DataFrame(res).T.sort_values("ROC AUC", ascending=False)
