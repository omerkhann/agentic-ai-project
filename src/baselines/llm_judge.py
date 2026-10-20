"""Zero-shot LLM-as-judge baseline (proposal B.10, baseline 4).

The judge sees the issue, a condensed action log rendered from the parsed Trace (tool, command or
path, exit code, error flag) and the final patch, all truncated to a fixed character budget. It
answers with JSON validated into schema.JudgeVerdict. No hidden-test output is ever shown.

Runs only on the fixed 50-run judge sample. Replies are cached in data/llm_judge_cache.jsonl,
keyed by (provider, model, uid), so re-runs are free.

    python -m src.baselines.llm_judge [--provider ollama|gemini|groq|openrouter]
"""
from __future__ import annotations

import argparse
import json
import re
import time

import pandas as pd
from pydantic import ValidationError

from src import config
from src.data.loader import load_meta
from src.data.parser import iter_traces
from src.llm.provider import LLMProvider, get_provider
from src.schema import ActionType, JudgeVerdict, Role, Trace

CACHE_PATH = config.DATA_DIR / "llm_judge_cache.jsonl"
ISSUE_CHARS, CMD_CHARS, PATCH_CHARS = 3000, 160, 4000
MAX_ATTEMPTS = 2

SYSTEM = (
    "You review runs of an autonomous software-engineering agent. Given a GitHub issue, the agent's "
    "action log and its final patch, judge whether the patch resolves the issue, i.e. whether the "
    "repository's hidden unit tests for this issue will pass.\n"
    'Reply with JSON only: {"verdict": "PASS" or "FAIL", "confidence": integer 0-100 (how sure you are '
    'of the verdict), "rationale": one or two sentences}.'
)

_ISSUE_RE = re.compile(r"<pr_description>(.*?)(?:</pr_description>|$)", re.S)
_JSON_RE = re.compile(r"\{.*\}", re.S)


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split()) if n <= CMD_CHARS else s
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def render(trace: Trace) -> str:
    m = _ISSUE_RE.search(trace.task)
    issue = (m.group(1) if m else trace.task).strip()
    lines = []
    n = 0
    for s in trace.steps:
        if s.role != Role.assistant:
            continue
        n += 1
        if s.action_type == ActionType.bash:
            desc = f"bash `{_clip(s.command or '', CMD_CHARS)}`"
            if s.exit_code is not None:
                desc += f" -> exit {s.exit_code}"
        elif s.action_type in (ActionType.view, ActionType.create, ActionType.edit):
            desc = f"{s.action_type.value} {s.path or '?'}"
        else:
            desc = s.action_type.value
        lines.append(f"{n:>3}. {desc}{'  [ERROR]' if s.is_error else ''}")
    patch = trace.final_patch.strip() or "(empty patch: the agent changed nothing)"
    return (f"## Issue\n{_clip(issue, ISSUE_CHARS)}\n\n## Agent action log ({n} actions)\n" + "\n".join(lines)
            + f"\n\n## Final patch\n{_clip(patch, PATCH_CHARS)}")


def parse_verdict(text: str) -> JudgeVerdict:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("no JSON object in reply")
    data = json.loads(m.group(0))
    if isinstance(data.get("verdict"), str):
        data["verdict"] = data["verdict"].strip().upper()
    return JudgeVerdict.model_validate(data)


def judge(trace: Trace, llm: LLMProvider) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(trace)}]
    t0 = time.time()
    err = None
    for _ in range(MAX_ATTEMPTS):
        try:
            v = parse_verdict(llm.complete(messages, json_mode=True))
            return {"p_success": v.p_success, "verdict": v.verdict, "confidence": v.confidence,
                    "rationale": v.rationale, "parse_ok": True, "latency_s": time.time() - t0}
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            err = str(e)[:200]
    # Unparseable after retries: fall back to an uninformative 0.5 and record it.
    return {"p_success": 0.5, "verdict": None, "confidence": None, "rationale": f"parse error: {err}",
            "parse_ok": False, "latency_s": time.time() - t0}


def _load_cache() -> dict[tuple, dict]:
    if not CACHE_PATH.exists():
        return {}
    rows = [json.loads(l) for l in CACHE_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["provider"], r["model"], r["uid"]): r for r in rows}


def run_judge(splits: pd.DataFrame, provider: str | None = None) -> pd.DataFrame:
    """-> Prediction rows (+ verdict/latency columns) for the judge sample."""
    llm = get_provider(provider)
    sample = splits[splits["judge_sample"]].set_index("uid")
    cache = _load_cache()
    todo = {u for u in sample.index if (llm.name, llm.model, u) not in cache}
    if todo:
        print(f"  llm_judge: {len(todo)} runs to score with {llm.name}:{llm.model}")
        with CACHE_PATH.open("a", encoding="utf-8") as f:
            for i, tr in enumerate(iter_traces(load_meta(), uids=todo), 1):
                row = {"provider": llm.name, "model": llm.model, "uid": tr.uid} | judge(tr, llm)
                f.write(json.dumps(row) + "\n")
                f.flush()
                cache[(llm.name, llm.model, tr.uid)] = row
                print(f"    {i}/{len(todo)} {tr.uid}: p={row['p_success']:.2f} ({row['latency_s']:.1f}s)")
    rows = [cache[(llm.name, llm.model, u)] for u in sample.index]
    out = pd.DataFrame(rows)
    out["instance_id"] = sample.loc[out["uid"], "instance_id"].to_numpy()
    out["label"] = sample.loc[out["uid"], "resolved"].to_numpy(bool)
    out["fold"] = sample.loc[out["uid"], "fold"].to_numpy()
    out["method"] = "llm_judge"
    return out


def main() -> None:
    from src.eval.splits import build_splits
    from src.eval.metrics import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--provider")
    preds = run_judge(build_splits(), ap.parse_args().provider)
    print(evaluate(preds, "llm_judge", "judge_sample").model_dump_json(indent=1))
    print(f"parse failures: {(~preds['parse_ok']).sum()}, mean latency {preds['latency_s'].mean():.1f}s")


if __name__ == "__main__":
    main()
