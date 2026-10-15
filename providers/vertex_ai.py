"""Google Vertex AI (Gemini) provider adapter, via the google-genai SDK."""

import json
import os
import re
import time
import zlib

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    GENAI_IMPORT_ERROR = None
except ImportError as error:
    genai = None
    genai_errors = None
    genai_types = None
    GENAI_IMPORT_ERROR = error


name = "vertex_ai"

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "gusevlabllm")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
default_model = os.environ.get("VERTEX_MODEL", "gemini-2.0-flash-001")

# Gemini 2.5+ counts thinking against max_output_tokens (up to 8K on Flash and
# Flash-Lite), so leave another 8K for JSON. Models before 2.5 cap output at 8K.
_MAX_OUTPUT_TOKENS = 16384
_LEGACY_MAX_OUTPUT_TOKENS = 8192
# A response cut off at the cap is regenerated with more room, in case thinking
# or a genuinely long answer used the budget, and at a higher temperature,
# because at temperature 0 Gemini can repeat one item until it hits the cap.
_TRUNCATION_RETRY_MAX_OUTPUT_TOKENS = 32768
_TRUNCATION_RETRY_TEMPERATURES = (0.5, 1.0)
_TRUNCATION_RETRY_MESSAGE = (
    "Your previous reply reached the output token limit before its JSON was "
    "complete, usually because it repeated itself. Reply again with one complete "
    "JSON object in the required format. List each item at most once, never "
    "repeat text, and keep every quote as short as the instructions allow."
)


def build_client():
    if GENAI_IMPORT_ERROR is not None:
        raise ImportError(
            "Vertex AI note extraction requires google-genai in the active environment."
        ) from GENAI_IMPORT_ERROR
    return genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
    )


def _gemini_version(model_name):
    match = re.search(r"gemini-(\d+)(?:\.(\d+))?", model_name.lower())
    return (int(match.group(1)), int(match.group(2) or 0)) if match else None


def _generation_attempts(model_name):
    """(temperature, max_output_tokens) for the first call, then each truncation retry.

    A temperature of None leaves Gemini 3+ at its default of 1.0, because Google
    warns that lower values can make those models loop.
    """
    version = _gemini_version(model_name) or (0, 0)
    if version < (2, 5):
        return [(0, _LEGACY_MAX_OUTPUT_TOKENS)] + [
            (temperature, _LEGACY_MAX_OUTPUT_TOKENS)
            for temperature in _TRUNCATION_RETRY_TEMPERATURES
        ]
    if version >= (3, 0):
        return [(None, _MAX_OUTPUT_TOKENS)] + [
            (None, _TRUNCATION_RETRY_MAX_OUTPUT_TOKENS)
        ] * len(_TRUNCATION_RETRY_TEMPERATURES)
    return [(0, _MAX_OUTPUT_TOKENS)] + [
        (temperature, _TRUNCATION_RETRY_MAX_OUTPUT_TOKENS)
        for temperature in _TRUNCATION_RETRY_TEMPERATURES
    ]


def _complete_json(text):
    """Return the JSON value a truncated response finished before the cap, if any.

    Gemini sometimes completes its answer, then pads with whitespace or starts
    repeating the object until max_output_tokens. The first value is complete.
    """
    stripped = (text or "").lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    return json.dumps(value, ensure_ascii=False)


def _is_repetitive(text):
    """True when the end of the text compresses like a loop, not like prose."""
    tail = text[-8000:].encode("utf-8")
    return len(zlib.compress(tail)) < 0.15 * len(tail)


def _describe_truncation(response, text, max_output_tokens):
    usage = getattr(response, "usage_metadata", None)
    output_tokens = getattr(usage, "candidates_token_count", None) or 0
    thinking_tokens = getattr(usage, "thoughts_token_count", None) or 0
    if not text or thinking_tokens >= max_output_tokens // 2:
        cause = "thinking"
    elif _is_repetitive(text):
        cause = "repetition"
    else:
        cause = "length"
    return f"cause={cause} output_tokens={output_tokens} thinking_tokens={thinking_tokens}"


def _generate(client, model_name, contents, config, max_retries):
    """Make one generate_content call, retrying transport errors with backoff."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return response, None
        except genai_errors.APIError as error:
            body = str(error)
            status_code = getattr(error, "code", None)
            if attempt < max_retries - 1:
                if status_code == 429:
                    time.sleep(2 ** attempt * 5)
                elif status_code in (408, 504):
                    time.sleep(2 ** attempt * 3)
                else:
                    time.sleep(2 ** attempt)
                continue
            if status_code == 429:
                return None, f"resource_exhausted: {body[:200]}"
            if status_code in (408, 504):
                return None, f"deadline_exceeded: {body[:200]}"
            return None, f"api_error: {body[:200]}"
        except Exception as error:  # noqa: BLE001
            return None, f"unexpected: {type(error).__name__}: {str(error)[:200]}"
    return None, "max_retries_exceeded"


def call_with_retry(client, model_name, messages, max_retries=3):
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] == "user"]
    system_instruction = "\n\n".join(system_parts) if system_parts else None

    attempts = _generation_attempts(model_name)
    truncation = None
    for attempt, (temperature, max_output_tokens) in enumerate(attempts):
        config = {
            "response_mime_type": "application/json",
            "system_instruction": system_instruction,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            config["temperature"] = temperature
        parts = (user_parts + [_TRUNCATION_RETRY_MESSAGE]) if attempt else user_parts
        response, error = _generate(
            client,
            model_name,
            "\n\n".join(parts),
            genai_types.GenerateContentConfig(**config),
            max_retries,
        )
        if error:
            return None, error
        try:
            finish_reason = response.candidates[0].finish_reason
            finish_name = getattr(finish_reason, "value", str(finish_reason))
            if finish_name in ("SAFETY", "RECITATION", "BLOCKLIST"):
                return None, "content_filter_response"
            text = response.text
            if finish_name == "MAX_TOKENS":
                complete = _complete_json(text)
                if complete is not None:
                    return complete, None
                truncation = _describe_truncation(response, text, max_output_tokens)
                continue
            if text is None:
                return None, f"empty_response: finish_reason={finish_name}"
            return text.strip(), None
        except Exception as error:  # noqa: BLE001
            return None, f"unexpected: {type(error).__name__}: {str(error)[:200]}"
    return None, (
        f"truncated_response: finish_reason=MAX_TOKENS {truncation} "
        f"attempts={len(attempts)}"
    )
