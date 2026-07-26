"""Provider adapter protocol implemented by each LLM backend."""

from typing import Protocol


class Provider(Protocol):
    name: str
    default_model: str

    def build_client(self): ...

    def call_with_retry(self, client, model_name, messages, max_retries=3):
        """Return (response_text, error_string). Errors are returned, never raised.

        `messages` is OpenAI-style [{"role": ..., "content": ...}].
        Adapters translate as needed.
        """
        ...
