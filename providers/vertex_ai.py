"""Google Vertex AI (Gemini) provider adapter, via the google-genai SDK."""

import os
import time

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


def call_with_retry(client, model_name, messages, max_retries=3):
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] == "user"]
    system_instruction = "\n\n".join(system_parts) if system_parts else None

    generation_config = genai_types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        system_instruction=system_instruction,
        # Gemini Flash's real ceiling is 8192; request it explicitly so a
        # truncated response fails fast via finish_reason=MAX_TOKENS below
        # rather than silently defaulting to a smaller cap.
        max_output_tokens=8192,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="\n\n".join(user_parts),
                config=generation_config,
            )
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            finish_name = getattr(finish_reason, "value", str(finish_reason))
            if finish_name in ("SAFETY", "RECITATION", "BLOCKLIST"):
                return None, "content_filter_response"
            if finish_name == "MAX_TOKENS":
                return None, "truncated_response: finish_reason=MAX_TOKENS"
            if response.text is None:
                return None, f"empty_response: finish_reason={finish_name}"
            text = response.text.strip()
            return text, None
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
