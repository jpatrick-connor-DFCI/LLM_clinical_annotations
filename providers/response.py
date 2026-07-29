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


def _escape_stray_quotes(text):
    """Escape unescaped double-quotes that appear inside JSON string values.

    Gemini occasionally emits a verbatim clinical quote that itself contains
    embedded quotation marks (e.g. a patient statement) without escaping them,
    which breaks the string it's nested in ("Expecting ',' delimiter" /
    "Expecting property name..."). Walk the text tracking string state; a
    quote encountered inside a string is only treated as the real terminator
    if it's immediately followed (modulo whitespace) by a JSON structural
    character or end of text — otherwise it's a literal quote and gets escaped.
    """
    out = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                out.append(ch)
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                terminates = j >= n or text[j] in ",:}]"
                if terminates:
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        if ch == '"':
            in_string = True
        i += 1
    return "".join(out)


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

    # Gemini occasionally emits a trailing comma before a closing
    # brace/bracket, which json.loads rejects outright.
    deduped = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    try:
        return _first_json_value(deduped)
    except json.JSONDecodeError:
        pass

    # Last resort: repair unescaped quotes nested inside string values (e.g. a
    # verbatim clinical quote containing its own quotation marks).
    repaired = _escape_stray_quotes(deduped)
    return _first_json_value(repaired)
