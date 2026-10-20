import pytest

from src.baselines.llm_judge import judge, parse_verdict, render
from src.data.parser import parse_messages
from src.llm.provider import LLMProvider
from tests.test_parser import MSGS


class FakeLLM(LLMProvider):
    name, model = "fake", "fake-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def complete(self, messages, json_mode=False):
        self.seen.append(messages)
        return self.replies.pop(0)


def _trace():
    return parse_messages(MSGS, uid="u", instance_id="i", run_id="r", resolved=False, final_patch="")


def test_parse_verdict_maps_confidence_to_p_success():
    assert parse_verdict('{"verdict": "PASS", "confidence": 80}').p_success == pytest.approx(0.8)
    assert parse_verdict('Here: {"verdict": "fail", "confidence": 90, "rationale": "x"}').p_success == pytest.approx(0.1)


def test_render_has_no_label_and_marks_empty_patch():
    text = render(_trace())
    assert "## Agent action log" in text and "empty patch" in text
    assert "resolved" not in text.lower()


def test_judge_retries_then_succeeds():
    llm = FakeLLM(["not json", '{"verdict": "FAIL", "confidence": 70, "rationale": "no patch"}'])
    out = judge(_trace(), llm)
    assert out["parse_ok"] and out["p_success"] == pytest.approx(0.3)
    assert len(llm.seen) == 2


def test_judge_falls_back_to_half_when_unparseable():
    out = judge(_trace(), FakeLLM(["nope", '{"verdict": "MAYBE", "confidence": 5}']))
    assert not out["parse_ok"] and out["p_success"] == 0.5
