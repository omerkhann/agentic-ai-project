"""Download the SWE-Gym trajectory shards and build a light metadata table.

Only the cheap columns (ids, label, patch) are materialised. The heavy `messages`
column is streamed later by `src.data.parser`.

    python -m src.data.loader            # download (if needed), print stats, cache meta
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src import config


def download_shards(refresh: bool = False) -> list[Path]:
    """Fetch the 3 parquet shards into data/raw (no-op if already present)."""
    from huggingface_hub import hf_hub_download

    paths = []
    for shard in config.HF_SHARDS:
        local = config.RAW_DIR / shard
        if refresh or not local.exists():
            hf_hub_download(
                config.HF_DATASET, shard, repo_type="dataset",
                local_dir=config.RAW_DIR, token=os.getenv("HF_TOKEN") or None,
            )
        paths.append(local)
    return paths


def shard_paths() -> list[Path]:
    return [config.RAW_DIR / s for s in config.HF_SHARDS]


def load_meta(refresh: bool = False) -> pd.DataFrame:
    """One row per run: uid, instance_id, run_id, resolved, git_patch, shard, row, policy_model, max_iter."""
    if config.META_PATH.exists() and not refresh:
        return pd.read_parquet(config.META_PATH)

    frames = []
    for shard_idx, path in enumerate(download_shards(refresh)):
        pf = pq.ParquetFile(path)
        # Nested-column projection: only git_patch out of the test_result struct.
        tbl = pf.read(columns=["instance_id", "run_id", "resolved", "test_result.git_patch"])
        df = pd.DataFrame({
            "instance_id": tbl.column("instance_id").to_pylist(),
            "run_id": tbl.column("run_id").to_pylist(),
            "resolved": tbl.column("resolved").to_pylist(),
            "git_patch": [r["git_patch"] if r else None for r in tbl.column("test_result").to_pylist()],
        })
        df["shard"] = shard_idx
        df["row"] = range(len(df))
        frames.append(df)

    meta = pd.concat(frames, ignore_index=True)
    meta["resolved"] = meta["resolved"].astype(bool)
    # run_id is a sampling-config string shared by many runs (and not unique per
    # instance), so the unique run key is the physical location in the shards.
    meta.insert(0, "uid", [f"s{s}r{r}" for s, r in zip(meta["shard"], meta["row"])])
    cfg = meta["run_id"].str.extract(r"^(?P<policy_model>.+?)_maxiter_(?P<max_iter>\d+)")
    meta["policy_model"] = cfg["policy_model"]
    meta["max_iter"] = cfg["max_iter"].astype(int)
    config.META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(config.META_PATH, index=False)
    return meta


def dataset_stats(meta: pd.DataFrame) -> dict:
    per_task = meta.groupby("instance_id")["resolved"].agg(["size", "sum"])
    mixed = per_task[(per_task["sum"] > 0) & (per_task["sum"] < per_task["size"])]
    runs = per_task["size"]
    return {
        "rows": int(len(meta)),
        "unique_run_configs": int(meta["run_id"].nunique()),
        "runs_by_policy": meta["policy_model"].value_counts().to_dict(),
        "resolved_rate_by_policy": meta.groupby("policy_model")["resolved"].mean().round(4).to_dict(),
        "unique_instance_ids": int(len(per_task)),
        "resolved_rate": round(float(meta["resolved"].mean()), 4),
        "runs_per_instance": {
            "mean": round(float(runs.mean()), 2), "median": float(runs.median()),
            "min": int(runs.min()), "max": int(runs.max()),
        },
        "instances_all_pass": int((per_task["sum"] == per_task["size"]).sum()),
        "instances_all_fail": int((per_task["sum"] == 0).sum()),
        "mixed_instances": int(len(mixed)),
        "runs_in_mixed_instances": int(mixed["size"].sum()),
        "pass_fail_pairs_in_mixed": int((mixed["sum"] * (mixed["size"] - mixed["sum"])).sum()),
    }


def print_stats(stats: dict) -> None:
    print("== SWE-Gym OpenHands-Sampled-Trajectories ==")
    for k, v in stats.items():
        print(f"  {k:28s} {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    meta = load_meta(args.refresh)
    print_stats(dataset_stats(meta))
    print(f"cached -> {config.META_PATH}")


if __name__ == "__main__":
    main()
