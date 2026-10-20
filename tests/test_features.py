from src.data.parser import parse_messages
from src.features import extract, extract_task, patch_stats
from tests.test_parser import MSGS, _call

PATCH = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
 y = 3
"""


def test_patch_stats():
    assert patch_stats(PATCH) == (2, 1)
    assert patch_stats("") == (0, 0)


def test_extract():
    msgs = MSGS[:-1] + [
        {"role": "assistant", "content": "", "tool_calls": [_call("x", "execute_bash", command="pytest -q")]},
        {"role": "tool", "name": "execute_bash", "tool_call_id": "x", "content": "[Command finished with exit code 0]"},
        MSGS[-1],
    ]
    f = extract(parse_messages(msgs, uid="u", instance_id="i", run_id="r", resolved=True, final_patch=PATCH))
    assert f.n_steps == 5
    assert f.n_errors == 2                 # failing pytest + editor ERROR
    assert f.n_repeated_commands == 1      # second identical `pytest -q`
    assert f.n_distinct_tools == 3
    assert f.n_edits_without_view == 1     # bar.py edited, only foo.py viewed
    assert f.ran_tests and f.finished and not f.patch_empty
    assert f.patch_lines == 2 and f.patch_files == 1


def test_extract_task():
    task = ("<uploaded_files>/workspace/x</uploaded_files>\n<pr_description>\nBug\n```python\nf()\n```\n"
            "Traceback (most recent call last):\n  ...\n</pr_description>\nPlease fix.")
    tr = parse_messages([{"role": "user", "content": task}], uid="u", instance_id="getmoto__moto-5321", run_id="r")
    t = extract_task(tr)
    assert t.repo == "getmoto__moto"
    assert t.issue_code_blocks == 1 and t.issue_has_traceback
    assert 0 < t.issue_tokens < len(task) // 4
