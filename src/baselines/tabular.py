"""Cross-validated tabular baselines. Each returns out-of-fold schema.Prediction rows.

  length_only     LogReg on log(n_steps), log(total_tokens)        (B.10 baseline 2)
  prompt_only     LogReg on issue features + repo one-hot           (B.10 baseline 3, task difficulty)
  telemetry_lr    LogReg on all RunFeatures (log1p, standardised)   (proposed, Phase 1 subset)
  telemetry_gbm   HistGradientBoosting on all RunFeatures           (proposed, Phase 1 subset)

All use the fold column from data/eval_splits.parquet, so every method sees the same splits.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.schema import RunFeatures, TaskFeatures

_log1p = FunctionTransformer(lambda x: np.log1p(np.asarray(x, dtype=float)), feature_names_out="one-to-one")


def _logreg(cols: list[str], categorical: list[str] = ()) -> Callable[[], Pipeline]:
    def make():
        num = make_pipeline(_log1p, StandardScaler())
        ct = ColumnTransformer([("num", num, cols)] +
                               ([("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical))] if categorical else []))
        return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000))
    return make


def _gbm(cols: list[str]) -> Callable[[], Pipeline]:
    def make():
        ct = ColumnTransformer([("num", "passthrough", cols)])
        return make_pipeline(ct, HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=config.SEED))
    return make


TELEMETRY = RunFeatures.feature_names()
METHODS: dict[str, tuple[Callable[[], Pipeline], list[str]]] = {
    "length_only": (_logreg(["n_steps", "total_tokens"]), ["n_steps", "total_tokens"]),
    "prompt_only": (_logreg(TaskFeatures.numeric_names(), ["repo"]), TaskFeatures.numeric_names() + ["repo"]),
    "telemetry_lr": (_logreg(TELEMETRY), TELEMETRY),
    "telemetry_gbm": (_gbm(TELEMETRY), TELEMETRY),
}


def cv_predict(data: pd.DataFrame, method: str) -> pd.DataFrame:
    """data: features merged with splits (uid, instance_id, resolved, fold, ...)."""
    make, cols = METHODS[method]
    X = data[cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = data["resolved"].to_numpy(bool)
    p = np.zeros(len(data))
    for k in sorted(data["fold"].unique()):
        test = (data["fold"] == k).to_numpy()
        model = make().fit(X[~test], y[~test])
        p[test] = model.predict_proba(X[test])[:, 1]
    return pd.DataFrame({"uid": data["uid"], "instance_id": data["instance_id"], "method": method,
                         "fold": data["fold"], "p_success": p, "label": y})


def coefficients(data: pd.DataFrame, method: str = "telemetry_lr") -> pd.Series:
    """Standardised LogReg coefficients fit on all runs, for the explainability readout (Gap 3)."""
    make, cols = METHODS[method]
    X = data[cols].astype({c: int for c in cols if data[c].dtype == bool})
    model = make().fit(X, data["resolved"].to_numpy(bool))
    names = model[0].get_feature_names_out()
    return pd.Series(model[-1].coef_[0], index=[n.split("__", 1)[1] for n in names]).sort_values(key=abs, ascending=False)
