"""Discrimination and calibration metrics over a set of Predictions.

Within-task AUROC: over tasks that have both resolved and unresolved runs, the fraction of
same-task (pass, fail) pairs where the pass run gets the higher score (ties count 1/2). This
equals the per-task AUROC weighted by n_pass * n_fail.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.schema import EvalReport

N_BINS = 15   # equal-mass bins, as in the proposal's evaluation plan


def pooled_auroc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def within_task_auroc(df: pd.DataFrame) -> tuple[Optional[float], int, int]:
    """df has columns instance_id, label, p_success. -> (auroc, n_mixed_tasks, n_pairs)"""
    wins = 0.0
    pairs = tasks = 0
    for _, g in df.groupby("instance_id"):
        pos = g.loc[g["label"], "p_success"].to_numpy()
        neg = g.loc[~g["label"], "p_success"].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        pairs += diff.size
        tasks += 1
    return (wins / pairs if pairs else None), tasks, pairs


def _equal_mass_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index per prediction, with bins holding (near) equal numbers of runs."""
    ranks = np.argsort(np.argsort(p, kind="stable"), kind="stable")
    return (ranks * n_bins // len(p)).astype(int)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error over equal-mass bins."""
    bins = _equal_mass_bins(p, n_bins)
    err = 0.0
    for b in np.unique(bins):
        m = bins == b
        err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    bins = _equal_mass_bins(p, n_bins)
    return (pd.DataFrame({"bin": bins, "y": y, "p": p}).groupby("bin")
            .agg(mean_p=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index())


def evaluate(preds: pd.DataFrame, method: str, eval_set: str = "all") -> EvalReport:
    """preds: rows of schema.Prediction for one method."""
    y = preds["label"].to_numpy(bool)
    p = preds["p_success"].to_numpy(float)
    wt, n_tasks, n_pairs = within_task_auroc(preds)
    return EvalReport(
        method=method, eval_set=eval_set, n_runs=len(preds),
        pooled_auroc=pooled_auroc(y, p), within_task_auroc=wt,
        n_mixed_tasks=n_tasks, n_pairs=n_pairs,
        brier=float(brier_score_loss(y, p)), ece=ece(y, p), base_rate=float(y.mean()),
    )
