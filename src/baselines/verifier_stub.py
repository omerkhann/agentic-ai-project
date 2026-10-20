"""SWE-Gym 32B verifier baseline (proposal B.10, baseline 1). Phase 2.

The verifier (SWE-Gym/OpenHands-32B-Verifier, Qwen2.5-Coder-32B) does not fit in 6 GB of VRAM,
so it will run on Kaggle in Phase 2. This module fixes the interface now so the evaluation code
does not change when it arrives: any verifier maps a Trace to p(resolved), and its scores are
loaded as ordinary Prediction rows.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from src import config
from src.schema import Trace

SCORES_PATH = config.DATA_DIR / "verifier_scores.parquet"   # uid, p_success (produced on Kaggle)


class Verifier(Protocol):
    name: str

    def score(self, trace: Trace) -> float:
        """p(resolved | issue, trajectory, final patch)."""


class SWEGymVerifierStub:
    name = "swegym_verifier_32b"

    def score(self, trace: Trace) -> float:
        raise NotImplementedError("32B verifier runs on Kaggle in Phase 2; see load_scores()")


def load_scores(splits: pd.DataFrame) -> pd.DataFrame | None:
    """Prediction rows from precomputed verifier scores, or None until they exist."""
    if not SCORES_PATH.exists():
        return None
    s = pd.read_parquet(SCORES_PATH).merge(splits, on="uid")
    return pd.DataFrame({"uid": s["uid"], "instance_id": s["instance_id"], "method": SWEGymVerifierStub.name,
                         "fold": s["fold"], "p_success": s["p_success"], "label": s["resolved"].astype(bool)})
