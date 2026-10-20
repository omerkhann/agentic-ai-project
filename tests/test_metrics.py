import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import ece, evaluate, within_task_auroc


def _preds(rows):
    return pd.DataFrame(rows, columns=["instance_id", "label", "p_success"])


def test_within_task_auroc_counts_only_same_task_pairs():
    df = _preds([
        ("a", True, 0.9), ("a", False, 0.1), ("a", False, 0.95),   # 2 pairs, 1 win
        ("b", True, 0.5), ("b", False, 0.5),                       # 1 pair, tie
        ("c", False, 0.99),                                        # not mixed
    ])
    auc, tasks, pairs = within_task_auroc(df)
    assert (tasks, pairs) == (2, 3)
    assert auc == pytest.approx(1.5 / 3)


def test_within_task_auroc_none_without_mixed_tasks():
    assert within_task_auroc(_preds([("a", True, 0.3), ("b", False, 0.2)])) == (None, 0, 0)


def test_ece_zero_when_perfectly_calibrated():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.25] * 4 + [0.75] * 4)
    assert ece(y, p, n_bins=2) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    y = np.zeros(10)
    p = np.full(10, 0.5)
    assert ece(y, p) == pytest.approx(0.5)


def test_evaluate_report_fields():
    df = _preds([("a", True, 0.8), ("a", False, 0.2), ("b", False, 0.1), ("b", True, 0.7)])
    r = evaluate(df, "m")
    assert r.pooled_auroc == 1.0 and r.within_task_auroc == 1.0
    assert r.base_rate == 0.5 and r.n_pairs == 2
