"""DFCI Azure OpenAI provider adapter."""

import os
import random
import re
import time

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import APIError, APITimeoutError, AzureOpenAI, RateLimitError

    AZURE_IMPORT_ERROR = None
except ImportError as error:
    DefaultAzureCredential = None
    get_bearer_token_provider = None
    AzureOpenAI = None
    APIError = Exception
    APITimeoutError = Exception
    RateLimitError = Exception
    AZURE_IMPORT_ERROR = error


name = "dfci_gpt"

DEFAULT_AZURE_OPENAI_ENDPOINT = os.environ.get(
    "CAIA_AZURE_OPENAI_ENDPOINT",
    "https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
)
DEFAULT_AZURE_OPENAI_API_VERSION = os.environ.get(
    "CAIA_AZURE_OPENAI_API_VERSION", "2024-04-01-preview"
)
default_model = os.environ.get("CAIA_AZURE_OPENAI_MODEL", "gpt-4o")


def build_client():
    if AZURE_IMPORT_ERROR is not None:
        raise ImportError(
            "Azure note extraction requires azure-identity and openai in the active environment."
        ) from AZURE_IMPORT_ERROR
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        api_version=DEFAULT_AZURE_OPENAI_API_VERSION,
        azure_endpoint=DEFAULT_AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
    )


def _retry_after_seconds(error):
    """Extract the server-requested wait (seconds) from a RateLimit/APIError, if any.

    Azure OpenAI 429s carry the wait in the `Retry-After` response header (seconds)
    or in the error body ("retry after N seconds"). Honoring it avoids retrying
    straight back into the same throttle window."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    match = re.search(r"retry after (\d+(?:\.\d+)?)\s*second", str(error), flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _backoff_sleep(attempt, base, error=None, cap=120):
    """Sleep with exponential backoff + jitter, preferring the server's Retry-After."""
    server_wait = _retry_after_seconds(error) if error is not None else None
    if server_wait is not None:
        delay = server_wait
    else:
        delay = base * (2 ** attempt)
    delay = min(delay, cap) + random.uniform(0, base)
    time.sleep(delay)


def call_with_retry(client, model_name, messages, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages, temperature=0,
            )
            if response.choices[0].finish_reason == "content_filter":
                return None, "content_filter_response"
            content = response.choices[0].message.content
            if content is None:
                return None, f"empty_response: finish_reason={response.choices[0].finish_reason}"
            return content.strip(), None
        except RateLimitError as error:
            last_error = f"rate_limit: {str(error)[:200]}"
            if attempt < max_retries - 1:
                _backoff_sleep(attempt, base=5, error=error)
        except APITimeoutError as error:
            last_error = f"timeout: {str(error)[:200]}"
            if attempt < max_retries - 1:
                _backoff_sleep(attempt, base=3, error=error)
        except APIError as error:
            body = str(error)
            if "content_filter" in body.lower() or "content_management" in body.lower():
                return None, "content_filter_input"
            last_error = f"api_error: {body[:200]}"
            if attempt < max_retries - 1:
                _backoff_sleep(attempt, base=2, error=error)
                continue
            return None, last_error
        except Exception as error:  # noqa: BLE001
            return None, f"unexpected: {type(error).__name__}: {str(error)[:200]}"
    return None, last_error or "max_retries_exceeded"
