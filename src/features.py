"""Telemetry feature extraction: Trace -> RunFeatures, cached to data/features.parquet.

    python -m src.features [--refresh]
"""
from __future__ import annotations

import argparse
import re
import time

import pandas as pd

from src import config
from src.data.loader import load_meta
from src.data.parser import iter_traces
from src.schema import ActionType, Role, RunFeatures, Trace

# Commands that execute tests (a test runner, or a test file run directly).
_TEST_RE = re.compile(
    r"\b(pytest|py\.test|unittest|nosetests|tox|nox)\b"
    r"|manage\.py\s+test|\bmake\s+test"
    r"|\btest_\w*\.py\b|\b\w*_test\.py\b"
)
_AGENT_ACTIONS = {ActionType.bash, ActionType.view, ActionType.create, ActionType.edit,
                  ActionType.finish, ActionType.message, ActionType.other_tool}


def patch_stats(patch: str) -> tuple[int, int]:
    """-> (changed lines, files touched) of a unified git diff."""
    lines = files = 0
    for ln in patch.splitlines():
        if ln.startswith("diff --git"):
            files += 1
        elif (ln.startswith("+") and not ln.startswith("+++")) or (ln.startswith("-") and not ln.startswith("---")):
            lines += 1
    return lines, files


def extract(trace: Trace) -> RunFeatures:
    actions = [s for s in trace.steps if s.role == Role.assistant and s.action_type in _AGENT_ACTIONS]

    seen_sigs: set[str] = set()
    repeated = 0
    viewed: set[str] = set()
    edits_blind = 0
    ran_tests = False
    for s in actions:
        if s.args_sig:
            repeated += s.args_sig in seen_sigs
            seen_sigs.add(s.args_sig)
        if s.action_type in (ActionType.view, ActionType.create) and s.path:
            viewed.add(s.path)
        elif s.action_type == ActionType.edit and s.path and s.path not in viewed:
            edits_blind += 1
        if s.action_type == ActionType.bash and s.command and _TEST_RE.search(s.command):
            ran_tests = True

    p_lines, p_files = patch_stats(trace.final_patch)
    return RunFeatures(
        uid=trace.uid,
        instance_id=trace.instance_id,
        resolved=trace.resolved,
        n_steps=len(actions),
        total_tokens=sum(s.tokens + s.obs_tokens for s in trace.steps),
        n_errors=sum(s.is_error for s in trace.steps),
        n_repeated_commands=repeated,
        n_distinct_tools=len({s.tool_name for s in actions if s.tool_name}),
        n_edits_without_view=edits_blind,
        ran_tests=ran_tests,
        patch_lines=p_lines,
        patch_files=p_files,
        patch_empty=not trace.final_patch.strip(),
        finished=any(s.action_type == ActionType.finish for s in actions),
    )


def build_features(refresh: bool = False) -> pd.DataFrame:
    if config.FEATURES_PATH.exists() and not refresh:
        return pd.read_parquet(config.FEATURES_PATH)
    meta = load_meta()
    t0 = time.time()
    rows = []
    for i, tr in enumerate(iter_traces(meta), 1):
        rows.append(extract(tr).model_dump())
        if i % 1000 == 0:
            print(f"  features: {i}/{len(meta)} runs ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df = df.merge(meta[["uid", "run_id", "policy_model", "max_iter"]], on="uid", how="left")
    df.to_parquet(config.FEATURES_PATH, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    df = build_features(ap.parse_args().refresh)
    feats = RunFeatures.feature_names()
    print(df.groupby("resolved")[feats].mean().T.round(2).to_string())
    print(f"cached -> {config.FEATURES_PATH}")


if __name__ == "__main__":
    main()
