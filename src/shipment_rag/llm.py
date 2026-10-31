"""Gemini client (google-genai SDK) with retries on transient failures.

The old `google-generativeai` package used in the Colab notebook is deprecated;
`google-genai` is its supported successor.
"""

import logging
from typing import Protocol

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """The model could not produce an answer (quota, auth, outage, empty reply)."""


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


def _is_transient(exc: BaseException) -> bool:
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return exc.code in (408, 429, 500, 502, 503, 504)
    return isinstance(exc, (ConnectionError, TimeoutError))


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_output_tokens: int = 512,
        max_retries: int = 4,
        timeout_s: float = 30.0,
    ):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._call = retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(initial=2, max=30),
            before_sleep=lambda rs: log.warning(
                "llm transient error, retrying", extra={"attempt": rs.attempt_number}
            ),
            reraise=True,
        )(self._generate_once)

    def _generate_once(self, system: str, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        return (resp.text or "").strip()

    def generate(self, system: str, prompt: str) -> str:
        try:
            text = self._call(system, prompt)
        except Exception as exc:
            log.exception("llm call failed")
            raise LLMUnavailableError(str(exc)) from exc
        if not text:
            raise LLMUnavailableError("empty response from model")
        return text
