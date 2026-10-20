"""End-to-end Phase 1 baseline: data -> features -> splits -> baselines -> metrics -> results/.

    python -m src.run_baseline                 # tabular baselines only
    python -m src.run_baseline --with-llm      # + zero-shot LLM judge on the 50-run sample
    python -m src.run_baseline --refresh       # rebuild cached data/ artifacts

Each stage reuses its cached output in data/ unless --refresh is given.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src import config
from src.baselines import tabular, verifier_stub
from src.data.loader import dataset_stats, load_meta
from src.eval import report
from src.eval.metrics import evaluate
from src.eval.splits import build_splits
from src.features import build_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="rebuild meta, features and splits")
    ap.add_argument("--with-llm", action="store_true", help="also run the LLM judge (needs a provider, see .env)")
    ap.add_argument("--provider", help="override LLM_PROVIDER")
    args = ap.parse_args()
    t0 = time.time()

    print("[1/5] data")
    meta = load_meta(args.refresh)
    stats = dataset_stats(meta)
    print(f"  {stats['rows']} runs, {stats['unique_instance_ids']} tasks, resolved rate {stats['resolved_rate']:.3f}")

    print("[2/5] features")
    feats = build_features(args.refresh)
    print("[3/5] splits")
    splits = build_splits(args.refresh)
    data = feats.merge(splits[["uid", "fold", "judge_sample"]], on="uid", validate="1:1")

    print("[4/5] baselines")
    preds: dict[str, pd.DataFrame] = {}
    for method in tabular.METHODS:
        t = time.time()
        preds[method] = tabular.cv_predict(data, method)
        print(f"  {method}: {time.time() - t:.1f}s")
    verifier = verifier_stub.load_scores(splits)
    if verifier is not None:
        preds[verifier_stub.SWEGymVerifierStub.name] = verifier

    judge = None
    if args.with_llm:
        from src.baselines.llm_judge import run_judge
        judge = run_judge(splits, args.provider)

    print("[5/5] evaluation")
    reports = [evaluate(p, m) for m, p in preds.items()]
    sample = set(splits.loc[splits["judge_sample"], "uid"])
    reports += [evaluate(p[p["uid"].isin(sample)], m, "judge_sample") for m, p in preds.items()]
    extra = {"dataset": stats, "telemetry_lr_coefficients": tabular.coefficients(data).round(4).to_dict()}
    if judge is not None:
        reports.append(evaluate(judge, "llm_judge", "judge_sample"))
        extra["llm_judge"] = {"provider": judge["provider"].iloc[0], "model": judge["model"].iloc[0],
                              "parse_failures": int((~judge["parse_ok"]).sum()),
                              "mean_latency_s": round(float(judge["latency_s"].mean()), 2)}
    report.write_metrics(reports, extra)
    report.reliability_plot(preds)
    pd.concat(list(preds.values()) + ([judge[["uid", "instance_id", "method", "fold", "p_success", "label"]]]
                                      if judge is not None else [])).to_parquet(
        config.DATA_DIR / "predictions.parquet", index=False)

    print((config.RESULTS_DIR / "metrics.md").read_text())
    print(f"done in {time.time() - t0:.0f}s -> {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
