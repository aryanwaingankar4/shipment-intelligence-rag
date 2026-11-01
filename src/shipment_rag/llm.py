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


class ModelNotFoundError(LLMUnavailableError):
    """GEMINI_MODEL doesn't exist or isn't available to this API key."""

    def __init__(self, model: str):
        super().__init__(
            f"Gemini model '{model}' was not found (HTTP 404). It may be retired or not "
            "enabled for this API key. Set GEMINI_MODEL to an available model; "
            "`shipment-rag models` lists them."
        )


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


def _is_transient(exc: BaseException) -> bool:
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return exc.code in (408, 429, 500, 502, 503, 504)
    return isinstance(exc, (ConnectionError, TimeoutError))


def _is_not_found(exc: BaseException) -> bool:
    from google.genai import errors

    return isinstance(exc, errors.APIError) and exc.code == 404


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        max_retries: int = 4,
        timeout_s: float = 30.0,
        thinking_level: str = "low",
        thinking_budget: int | None = None,
    ):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )
        self.model = model
        self.temperature = temperature
        # On thinking models, thought tokens count against this budget too.
        self.max_output_tokens = max_output_tokens
        # Gemini 3.x uses thinking_level; 2.5 models need thinking_budget instead.
        if thinking_budget is not None:
            self._thinking = types.ThinkingConfig(thinking_budget=thinking_budget)
        elif thinking_level:
            self._thinking = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel[thinking_level.upper()]
            )
        else:
            self._thinking = None
        self._call = retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(initial=2, max=30),
            before_sleep=lambda rs: log.warning(
                "llm transient error, retrying", extra={"attempt": rs.attempt_number}
            ),
            reraise=True,
        )(self._generate_once)

    def check_model(self) -> None:
        """Fail fast at startup if GEMINI_MODEL doesn't exist (costs no generation quota)."""
        try:
            self._client.models.get(model=self.model)
        except Exception as exc:
            if _is_not_found(exc):
                raise ModelNotFoundError(self.model) from exc
            raise

    def list_models(self) -> list[str]:
        return sorted(
            m.name.removeprefix("models/")
            for m in self._client.models.list()
            if "generateContent" in (m.supported_actions or [])
        )

    def _generate_once(self, system: str, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                thinking_config=self._thinking,
            ),
        )
        self._warn_if_truncated(resp)
        return (resp.text or "").strip()

    def _warn_if_truncated(self, resp) -> None:
        candidates = resp.candidates or []
        reason = candidates[0].finish_reason if candidates else None
        if reason == self._types.FinishReason.MAX_TOKENS:
            usage = resp.usage_metadata
            log.warning(
                "llm answer truncated: hit max_output_tokens",
                extra={
                    "model": self.model,
                    "max_output_tokens": self.max_output_tokens,
                    "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
                    "answer_tokens": getattr(usage, "candidates_token_count", None),
                    "hint": "raise LLM_MAX_OUTPUT_TOKENS or lower LLM_THINKING_LEVEL",
                },
            )

    def generate(self, system: str, prompt: str) -> str:
        try:
            text = self._call(system, prompt)
        except Exception as exc:
            if _is_not_found(exc):
                log.error("llm model not found", extra={"model": self.model})
                raise ModelNotFoundError(self.model) from exc
            log.exception("llm call failed")
            raise LLMUnavailableError(str(exc)) from exc
        if not text:
            raise LLMUnavailableError("empty response from model")
        return text
