import json

from src.data.parser import parse_messages
from src.schema import ActionType


def _call(cid, name, **args):
    return {"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}


MSGS = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "Fix the bug in foo.py"},
    {"role": "assistant", "content": "look", "tool_calls": [_call("a", "str_replace_editor", command="view", path="/w/foo.py")]},
    {"role": "tool", "name": "str_replace_editor", "tool_call_id": "a", "content": "OBSERVATION:\nHere's the result"},
    {"role": "assistant", "content": "", "tool_calls": [_call("b", "execute_bash", command="pytest -q")]},
    {"role": "tool", "name": "execute_bash", "tool_call_id": "b", "content": "OBSERVATION:\n1 failed\n[Command finished with exit code 1]"},
    {"role": "assistant", "content": "", "tool_calls": [_call("c", "str_replace_editor", command="str_replace", path="/w/bar.py", old_str="x", new_str="y")]},
    {"role": "tool", "name": "str_replace_editor", "tool_call_id": "c", "content": "OBSERVATION:\nERROR:\nNo replacement was performed"},
    {"role": "assistant", "content": "done", "tool_calls": [_call("d", "finish")]},
]


def test_parse_basic():
    t = parse_messages(MSGS, uid="u", instance_id="i", run_id="r", resolved=False)
    acts = [s.action_type for s in t.steps]
    assert acts == [ActionType.prompt, ActionType.prompt, ActionType.view, ActionType.bash,
                    ActionType.edit, ActionType.finish]
    bash = t.steps[3]
    assert bash.command == "pytest -q" and bash.exit_code == 1 and bash.is_error
    assert t.steps[4].is_error and t.steps[4].path == "/w/bar.py"
    assert not t.steps[2].is_error
    assert t.n_actions == 4
    assert t.task.startswith("Fix the bug")


def test_runtime_failure_is_error():
    msgs = MSGS[:4] + [
        {"role": "assistant", "content": "", "tool_calls": [_call("e", "execute_bash", command="ls")]},
        {"role": "tool", "name": "execute_bash", "tool_call_id": "e",
         "content": "OBSERVATION:\nAction execution failed: 504\n[Error occurred in processing last action]"},
    ]
    t = parse_messages(msgs, uid="u", instance_id="i", run_id="r")
    assert t.steps[-1].is_error and t.steps[-1].exit_code is None
