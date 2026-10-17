"""Parse OpenHands chat-format trajectories into `schema.Trace`.

Message format (OpenAI-style):
  system / user           -> prompt
  assistant               -> optional text + tool_calls[{id, function{name, arguments(JSON)}}]
  tool (tool_call_id)     -> observation text for that call

`iter_traces` streams parquet record batches, so the full `messages` column is never
materialised. Peak memory is about one row group.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Iterator, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.data.loader import shard_paths
from src.schema import ActionType, Role, Step, Trace

_EXIT_RE = re.compile(r"\[Command finished with exit code (-?\d+)\]")
_RUNTIME_ERR = ("[Error occurred in processing last action]", "Action execution failed")
_EDIT_CMDS = {"str_replace", "insert", "undo_edit"}
TASK_CHARS = 4000


def estimate_tokens(text: Optional[str]) -> int:
    """Cheap tokenizer-free estimate (~4 chars/token for English and code)."""
    return len(text) // 4 if text else 0


def _parse_args(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def _classify(tool: str, args: dict) -> tuple[ActionType, Optional[str], Optional[str]]:
    """-> (action_type, command, path)"""
    if tool == "execute_bash":
        return ActionType.bash, (args.get("command") or "").strip(), None
    if tool == "str_replace_editor":
        cmd = args.get("command")
        path = args.get("path")
        if cmd == "view":
            return ActionType.view, cmd, path
        if cmd == "create":
            return ActionType.create, cmd, path
        if cmd in _EDIT_CMDS:
            return ActionType.edit, cmd, path
        return ActionType.other_tool, cmd, path
    if tool == "finish":
        return ActionType.finish, None, None
    return ActionType.other_tool, None, None


def _observation_status(tool: Optional[str], content: str) -> tuple[Optional[int], bool]:
    """-> (exit_code, is_error) from an observation's text."""
    if any(m in content for m in _RUNTIME_ERR):
        return None, True
    if tool == "execute_bash":
        codes = _EXIT_RE.findall(content)
        if codes:
            code = int(codes[-1])
            return code, code != 0
        return None, False
    # editor errors are reported as "OBSERVATION:\nERROR:\n..."
    return None, content.lstrip().removeprefix("OBSERVATION:").lstrip().startswith("ERROR:")


def parse_messages(messages: list[dict], *, uid: str, instance_id: str, run_id: str,
                   resolved: Optional[bool] = None, final_patch: str = "") -> Trace:
    steps: list[Step] = []
    by_call_id: dict[str, int] = {}   # tool_call_id -> position in `steps`
    task = ""

    def add(**kw) -> int:
        steps.append(Step(index=len(steps), **kw))
        return len(steps) - 1

    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role in ("system", "user"):
            if role == "user" and not task:
                task = content[:TASK_CHARS]
            add(role=Role(role), action_type=ActionType.prompt, tokens=estimate_tokens(content))
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if not calls:
                add(role=Role.assistant, action_type=ActionType.message, tokens=estimate_tokens(content))
                continue
            # Assistant "thought" text is charged to the first call of the turn.
            for i, tc in enumerate(calls):
                fn = tc.get("function") or {}
                tool = fn.get("name")
                raw_args = fn.get("arguments")
                atype, cmd, path = _classify(tool, _parse_args(raw_args))
                sig = hashlib.md5(f"{tool}|{raw_args or ''}".encode()).hexdigest()[:12]
                pos = add(role=Role.assistant, action_type=atype, tool_name=tool, command=cmd, path=path, args_sig=sig,
                          tokens=estimate_tokens(raw_args) + (estimate_tokens(content) if i == 0 else 0))
                if tc.get("id"):
                    by_call_id[tc["id"]] = pos
        elif role == "tool":
            pos = by_call_id.get(m.get("tool_call_id"))
            exit_code, is_error = _observation_status(m.get("name"), content)
            if pos is None:   # orphan observation: keep as its own step
                add(role=Role.tool, action_type=ActionType.observation, tool_name=m.get("name"),
                    exit_code=exit_code, is_error=is_error, obs_tokens=estimate_tokens(content))
            else:
                s = steps[pos]
                s.exit_code, s.is_error = exit_code, is_error
                s.obs_tokens = estimate_tokens(content)

    return Trace(uid=uid, instance_id=instance_id, run_id=run_id, resolved=resolved,
                 task=task, steps=steps, final_patch=final_patch or "")


def iter_traces(meta: pd.DataFrame, batch_size: int = 64, limit: Optional[int] = None,
                uids: Optional[set[str]] = None) -> Iterator[Trace]:
    """Stream Traces shard by shard. `meta` comes from `loader.load_meta()` (used for uid/patch)."""
    meta_idx = meta.set_index(["shard", "row"])
    n = 0
    for shard_idx, path in enumerate(shard_paths()):
        pf = pq.ParquetFile(path)
        row = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=["messages"]):
            for msgs in batch.column("messages").to_pylist():
                r = meta_idx.loc[(shard_idx, row)]
                row += 1
                uid = r["uid"]
                if uids is not None and uid not in uids:
                    continue
                yield parse_messages(msgs, uid=uid, instance_id=r["instance_id"], run_id=r["run_id"],
                                     resolved=bool(r["resolved"]),
                                     final_patch=r["git_patch"] if isinstance(r["git_patch"], str) else "")
                n += 1
                if limit is not None and n >= limit:
                    return
