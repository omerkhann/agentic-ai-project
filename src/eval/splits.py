"""Fold assignment (GroupKFold by instance_id) and the fixed LLM-judge sample.

    python -m src.eval.splits [--refresh]

Output: data/eval_splits.parquet with columns uid, instance_id, resolved, fold, judge_sample.
The judge sample is LLM_JUDGE_SAMPLE/2 mixed tasks, each with one resolved and one unresolved
run, so it is label-balanced and every task contributes exactly one (pass, fail) pair.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config
from src.data.loader import load_meta

JUDGE_SAMPLE_JSON = config.RESULTS_DIR / "llm_judge_sample.json"


def assign_folds(meta: pd.DataFrame) -> np.ndarray:
    # Shuffle groups with a fixed seed so folds don't follow shard order.
    tasks = meta["instance_id"].unique()
    rng = np.random.default_rng(config.SEED)
    order = {t: i for i, t in enumerate(rng.permutation(tasks))}
    groups = meta["instance_id"].map(order).to_numpy()
    folds = np.full(len(meta), -1)
    for k, (_, test) in enumerate(GroupKFold(n_splits=config.N_FOLDS).split(meta, groups=groups)):
        folds[test] = k
    return folds


def judge_sample(meta: pd.DataFrame) -> list[str]:
    per_task = meta.groupby("instance_id")["resolved"].agg(["size", "sum"])
    mixed = sorted(per_task[(per_task["sum"] > 0) & (per_task["sum"] < per_task["size"])].index)
    rng = np.random.default_rng(config.SEED)
    tasks = rng.choice(mixed, size=config.LLM_JUDGE_SAMPLE // 2, replace=False)
    uids = []
    for t in sorted(tasks):
        runs = meta[meta["instance_id"] == t]
        for label in (True, False):
            pool = sorted(runs.loc[runs["resolved"] == label, "uid"])
            uids.append(pool[rng.integers(len(pool))])
    return uids


def build_splits(refresh: bool = False) -> pd.DataFrame:
    if config.SPLITS_PATH.exists() and not refresh:
        return pd.read_parquet(config.SPLITS_PATH)
    meta = load_meta()
    df = meta[["uid", "instance_id", "resolved"]].copy()
    df["fold"] = assign_folds(meta)
    sample = judge_sample(meta)
    df["judge_sample"] = df["uid"].isin(sample)
    df.to_parquet(config.SPLITS_PATH, index=False)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    JUDGE_SAMPLE_JSON.write_text(json.dumps({"seed": config.SEED, "uids": sample}, indent=1))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    df = build_splits(ap.parse_args().refresh)
    print(df.groupby("fold").agg(runs=("uid", "size"), tasks=("instance_id", "nunique"),
                                 resolved_rate=("resolved", "mean")).round(3).to_string())
    print(f"judge sample: {df['judge_sample'].sum()} runs, resolved rate "
          f"{df.loc[df['judge_sample'], 'resolved'].mean():.2f}")


if __name__ == "__main__":
    main()
