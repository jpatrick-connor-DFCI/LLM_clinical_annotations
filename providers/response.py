"""Provider-independent LLM response parsing."""

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_code_fence(text):
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def _first_json_value(text):
    """Decode the first complete JSON value in text, ignoring anything the
    model appended afterward (e.g. a duplicated/retried second object that
    otherwise triggers json.JSONDecodeError: Extra data)."""
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise json.JSONDecodeError("no JSON value found", text, 0)
    value, _ = json.JSONDecoder().raw_decode(text, start)
    return value


def parse_json_response(response_text):
    if response_text is None:
        return None
    text = response_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate = _strip_code_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        return _first_json_value(candidate)
    except json.JSONDecodeError:
        pass

    # Last resort: Gemini occasionally emits a trailing comma before a
    # closing brace/bracket, which json.loads rejects outright.
    deduped = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    return _first_json_value(deduped)
