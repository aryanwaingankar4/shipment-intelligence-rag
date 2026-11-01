"""Gemini client (google-genai SDK) with retries on transient failures.

The old `google-generativeai` package used in the Colab notebook is deprecated;
`google-genai` is its supported successor.
"""

import logging
import time
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


class QuotaExhaustedError(LLMUnavailableError):
    """A daily (or other long-window) quota is used up; retrying now is pointless."""


def _retry_delay_s(exc) -> float | None:
    """Server-suggested retry delay from a 429's RetryInfo detail, in seconds."""
    details = ((getattr(exc, "details", None) or {}).get("error") or {}).get("details") or []
    for d in details:
        delay = str(d.get("retryDelay", ""))
        if delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def _quota_desc(exc) -> str:
    """'quotaId (limit N)' from a 429's QuotaFailure detail, if present."""
    details = ((getattr(exc, "details", None) or {}).get("error") or {}).get("details") or []
    for d in details:
        for v in d.get("violations", []) or []:
            if v.get("quotaId"):
                return f"{v['quotaId']} (limit {v.get('quotaValue', '?')})"
    return ""


def _is_long_quota(exc: BaseException) -> bool:
    """429 for a per-day quota, or one whose suggested retry is minutes away."""
    from google.genai import errors

    if not (isinstance(exc, errors.APIError) and exc.code == 429):
        return False
    delay = _retry_delay_s(exc)
    return "PerDay" in str(getattr(exc, "details", "")) or (delay is not None and delay > 120)


class ModelNotFoundError(LLMUnavailableError):
    """GEMINI_MODEL doesn't exist or isn't available to this API key."""

    def __init__(self, model: str):
        super().__init__(
            f"Gemini model '{model}' was not found (HTTP 404). It may be retired or not "
            "enabled for this API key. Set GEMINI_MODEL to an available model; "
            "`shipment-rag models` lists them."
        )


class LLM(Protocol):
    def generate(self, system: str, prompt: str, json_schema: dict | None = None) -> str: ...


def _is_transient(exc: BaseException) -> bool:
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        if _is_long_quota(exc):  # daily quota: retrying with backoff only wastes time
            return False
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
        self._thinking_desc = (
            f"budget={thinking_budget}" if thinking_budget is not None
            else f"level={thinking_level or 'default'}"
        )
        # running totals, for quota reporting
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0,
                      "seconds": 0.0, "max_seconds": 0.0}
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

    def cache_identity(self) -> dict:
        """Everything that changes the output for a given prompt (used as a cache key)."""
        return {"model": self.model, "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens, "thinking": self._thinking_desc}

    def _generate_once(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        json_cfg = (
            {"response_mime_type": "application/json", "response_json_schema": json_schema}
            if json_schema else {}
        )
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                thinking_config=self._thinking,
                **json_cfg,
            ),
        )
        self._record_usage(resp, time.perf_counter() - t0)
        self._warn_if_truncated(resp)
        return (resp.text or "").strip()

    def _record_usage(self, resp, seconds: float) -> None:
        u = resp.usage_metadata
        self.usage["calls"] += 1
        self.usage["seconds"] = round(self.usage["seconds"] + seconds, 1)
        self.usage["max_seconds"] = round(max(self.usage["max_seconds"], seconds), 1)
        self.usage["prompt_tokens"] += getattr(u, "prompt_token_count", 0) or 0
        self.usage["output_tokens"] += getattr(u, "candidates_token_count", 0) or 0
        self.usage["thought_tokens"] += getattr(u, "thoughts_token_count", 0) or 0

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

    def generate(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        """json_schema: ask for JSON matching this schema (structured output)."""
        try:
            text = self._call(system, prompt, json_schema)
        except Exception as exc:
            if _is_not_found(exc):
                log.error("llm model not found", extra={"model": self.model})
                raise ModelNotFoundError(self.model) from exc
            if _is_long_quota(exc):
                delay = _retry_delay_s(exc)
                hours = f"; resets in ~{delay / 3600:.1f} h" if delay else ""
                quota = _quota_desc(exc)
                hours += f"; quota {quota}" if quota else ""
                log.error("llm quota exhausted", extra={"model": self.model})
                raise QuotaExhaustedError(
                    f"Gemini quota exhausted for {self.model} (HTTP 429){hours}"
                ) from exc
            log.exception("llm call failed")
            raise LLMUnavailableError(str(exc)) from exc
        if not text:
            raise LLMUnavailableError("empty response from model")
        return text
