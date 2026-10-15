import json
from types import SimpleNamespace

import pytest

from providers import vertex_ai

LOOP = '{"candidates": [' + ", ".join(
    ['{"quote": "Lytic lesion in the L3 vertebral body.", "confidence": "high"}'] * 400
)


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    # google-genai is optional here: the adapter only needs a config type and
    # an error class, so the config is recorded as plain keyword arguments.
    monkeypatch.setattr(
        vertex_ai, "genai_types", SimpleNamespace(GenerateContentConfig=dict)
    )
    monkeypatch.setattr(
        vertex_ai,
        "genai_errors",
        SimpleNamespace(APIError=type("APIError", (Exception,), {})),
    )


def _response(text, finish_reason="STOP", output_tokens=0, thinking_tokens=None):
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        text=text,
        usage_metadata=SimpleNamespace(
            candidates_token_count=output_tokens,
            thoughts_token_count=thinking_tokens,
        ),
    )


def _call(responses, model="gemini-2.5-flash-lite"):
    models = FakeModels(responses)
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": '{"notes": []}'},
    ]
    text, error = vertex_ai.call_with_retry(
        SimpleNamespace(models=models), model, messages
    )
    return text, error, models.calls


def test_json_completed_before_the_cap_is_returned_without_a_retry():
    answer = {"candidates": [{"quote": "Lytic lesion in the L3 vertebral body."}]}
    text, error, calls = _call(
        [_response(json.dumps(answer) + "\n" * 5000, "MAX_TOKENS", 16384)]
    )

    assert error is None
    assert json.loads(text) == answer
    assert len(calls) == 1


def test_truncated_response_is_regenerated_with_more_room_and_temperature():
    text, error, calls = _call(
        [_response(LOOP, "MAX_TOKENS", 16384), _response('{"candidates": []}')]
    )

    assert (text, error) == ('{"candidates": []}', None)
    assert [
        (call.config.get("temperature"), call.config["max_output_tokens"])
        for call in calls
    ] == [(0, 16384), (0.5, 32768)]
    assert vertex_ai._TRUNCATION_RETRY_MESSAGE not in calls[0].contents
    assert calls[1].contents.endswith(vertex_ai._TRUNCATION_RETRY_MESSAGE)


def test_persistent_truncation_keeps_its_prefix_and_reports_a_loop():
    text, error, calls = _call([_response(LOOP, "MAX_TOKENS", 16384)] * 3)

    assert text is None
    assert error.startswith(
        "truncated_response: finish_reason=MAX_TOKENS cause=repetition"
    )
    assert error.endswith("attempts=3")
    assert len(calls) == 3


def test_truncation_spent_on_thinking_is_reported_as_thinking():
    partial = '{"candidates": [{"quote": "Lytic lesion'
    _, error, _ = _call(
        [_response(partial, "MAX_TOKENS", 40, thinking_tokens=30000)] * 3
    )

    assert "cause=thinking" in error


def test_gemini_3_keeps_its_default_temperature():
    _, _, calls = _call(
        [_response('{"candidates": []}')], model="gemini-3.1-flash-lite"
    )

    assert "temperature" not in calls[0].config
    assert calls[0].config["max_output_tokens"] == 16384


def test_models_before_gemini_2_5_keep_the_8k_output_cap():
    _, error, calls = _call(
        [_response(LOOP, "MAX_TOKENS", 8192)] * 3, model="gemini-2.0-flash-001"
    )

    assert error.startswith("truncated_response")
    assert {call.config["max_output_tokens"] for call in calls} == {8192}
