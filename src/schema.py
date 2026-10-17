"""Structured contracts passed between pipeline stages.

Trace / Step   : normalised telemetry for one agent run (parsed from OpenHands messages)
RunFeatures    : fixed-width feature vector extracted from a Trace
JudgeVerdict   : validated structured output of the LLM-as-judge
Prediction     : one method's probability for one run (out-of-fold)
EvalReport     : metrics for one method
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ActionType(str, Enum):
    # agent actions (assistant turns)
    bash = "bash"
    view = "view"
    create = "create"
    edit = "edit"          # str_replace / insert / undo_edit
    finish = "finish"
    message = "message"    # assistant text with no tool call
    other_tool = "other_tool"
    # environment side
    observation = "observation"
    prompt = "prompt"      # system / user messages


class Step(BaseModel):
    """One event in a trajectory. Each assistant tool call becomes its own Step, and its
    observation is attached to the same Step (exit_code / is_error / obs_tokens)."""
    index: int = Field(ge=0)
    role: Role
    action_type: ActionType
    tool_name: Optional[str] = None
    command: Optional[str] = None          # bash command, or editor sub-command
    path: Optional[str] = None             # editor target path
    exit_code: Optional[int] = None        # bash only, parsed from observation
    is_error: bool = False
    tokens: int = Field(0, ge=0)           # estimated tokens of the action text (chars/4)
    obs_tokens: int = Field(0, ge=0)       # estimated tokens of the observation


class Trace(BaseModel):
    uid: str
    instance_id: str
    run_id: str
    resolved: Optional[bool] = None        # label; None at inference time
    task: str = ""                         # first user message (issue text), truncated
    steps: list[Step]
    final_patch: str = ""

    @computed_field
    @property
    def n_actions(self) -> int:
        return sum(s.role == Role.assistant for s in self.steps)


class RunFeatures(BaseModel):
    uid: str
    instance_id: str
    resolved: Optional[bool] = None
    n_steps: int                  # agent actions (tool calls + plain messages)
    total_tokens: int             # estimated, actions + observations + prompts
    n_errors: int                 # non-zero exit / editor ERROR / runtime failure
    n_repeated_commands: int      # actions identical to an earlier one in the same run
    n_distinct_tools: int
    n_edits_without_view: int     # edits to a path never viewed/created before
    ran_tests: bool               # any bash command invoking a test runner
    patch_lines: int              # changed (+/-) lines in final git patch
    patch_files: int
    patch_empty: bool
    finished: bool                # agent called `finish` (vs. hitting iteration cap)

    @classmethod
    def feature_names(cls) -> list[str]:
        return [f for f in cls.model_fields if f not in {"uid", "instance_id", "resolved"}]


class JudgeVerdict(BaseModel):
    p_success: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class Prediction(BaseModel):
    uid: str
    instance_id: str
    method: str
    fold: int
    p_success: float = Field(ge=0.0, le=1.0)
    label: bool


class EvalReport(BaseModel):
    method: str
    n_runs: int
    pooled_auroc: Optional[float]
    within_task_auroc: Optional[float]
    n_mixed_tasks: int
    n_pairs: int
    brier: float
    ece: float
    base_rate: float
