"""LLM clients: Gemini (google-genai SDK, retries on transient failures) for answers, and a
local Ollama client that eval-gen can use as its judge.

The old `google-generativeai` package used in the Colab notebook is deprecated;
`google-genai` is its supported successor.
"""

import json
import logging
import time
from collections.abc import Iterator
from typing import Protocol

import httpx
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
    """The configured model doesn't exist or isn't available."""

    def __init__(self, model: str, message: str | None = None):
        super().__init__(
            message or f"Gemini model '{model}' was not found (HTTP 404). It may be retired or "
            "not enabled for this API key. Set GEMINI_MODEL to an available model; "
            "`shipment-rag models` lists them."
        )


class LLM(Protocol):
    def generate(self, system: str, prompt: str, json_schema: dict | None = None) -> str: ...

    def generate_stream(self, system: str, prompt: str) -> Iterator[str]: ...


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
        self._retry = retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(initial=2, max=30),
            before_sleep=lambda rs: log.warning(
                "llm transient error, retrying", extra={"attempt": rs.attempt_number}
            ),
            reraise=True,
        )
        self._call = self._retry(self._generate_once)

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

    def _config(self, system: str, json_schema: dict | None = None):
        json_cfg = (
            {"response_mime_type": "application/json", "response_json_schema": json_schema}
            if json_schema else {}
        )
        return self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=self._thinking,
            **json_cfg,
        )

    def _generate_once(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model, contents=prompt, config=self._config(system, json_schema)
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
            raise self._unavailable(exc) from exc
        if not text:
            raise LLMUnavailableError("empty response from model")
        return text

    def _open_stream(self, system: str, prompt: str):
        """Start a stream and read its first chunk, which is when a failed request shows up."""
        stream = iter(self._client.models.generate_content_stream(
            model=self.model, contents=prompt, config=self._config(system)
        ))
        return stream, next(stream, None)

    def generate_stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield the answer as it is written. Retries only before the first chunk: after that
        the reader has seen part of the answer, so a failure has to surface as a failure."""
        t0 = time.perf_counter()
        last, produced = None, False
        try:
            stream, chunk = self._retry(self._open_stream)(system, prompt)
            while chunk is not None:
                last = chunk
                if chunk.text:
                    produced = True
                    yield chunk.text
                chunk = next(stream, None)
        except Exception as exc:
            raise self._unavailable(exc) from exc
        if last is not None:
            self._record_usage(last, time.perf_counter() - t0)
            self._warn_if_truncated(last)
        if not produced:
            raise LLMUnavailableError("empty response from model")

    def _unavailable(self, exc: Exception) -> LLMUnavailableError:
        """The error a caller should see for a failed Gemini call."""
        if _is_not_found(exc):
            log.error("llm model not found", extra={"model": self.model})
            return ModelNotFoundError(self.model)
        if _is_long_quota(exc):
            delay = _retry_delay_s(exc)
            hours = f"; resets in ~{delay / 3600:.1f} h" if delay else ""
            quota = _quota_desc(exc)
            hours += f"; quota {quota}" if quota else ""
            log.error("llm quota exhausted", extra={"model": self.model})
            return QuotaExhaustedError(
                f"Gemini quota exhausted for {self.model} (HTTP 429){hours}"
            )
        log.exception("llm call failed")
        return LLMUnavailableError(str(exc))


class OllamaLLM:
    """A local model served by Ollama: no quota and no dependence on a hosted API.

    Fails loudly rather than returning a damaged reply: a cut-off answer or a prompt that did
    not fit the context window raises, so it is never cached or scored as a verdict.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        num_ctx: int = 4096,
        seed: int = 42,
        timeout_s: float = 600.0,  # CPU inference, and the first call also loads the model
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.num_ctx = num_ctx
        self.seed = seed
        self.digest = ""  # set by check_model: a re-pulled tag must not reuse old verdicts
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0,
                      "seconds": 0.0, "max_seconds": 0.0}
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def _not_pulled(self) -> ModelNotFoundError:
        return ModelNotFoundError(
            self.model, f"Ollama has no model '{self.model}'. Run: ollama pull {self.model}"
        )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            return self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama not reachable at {self.base_url} ({type(exc).__name__}). "
                "Is it running?"
            ) from exc

    def check_model(self) -> None:
        """Fail fast if the server is down or the model has not been pulled."""
        resp = self._request("GET", "/api/tags")
        if resp.status_code >= 400:
            raise LLMUnavailableError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        for m in resp.json().get("models", []):
            if m.get("name") == self.model:
                self.digest = m.get("digest", "")[:12]
                return
        raise self._not_pulled()

    def cache_identity(self) -> dict:
        return {"provider": "ollama", "model": self.model, "digest": self.digest,
                "temperature": self.temperature, "max_output_tokens": self.max_output_tokens,
                "num_ctx": self.num_ctx, "seed": self.seed}

    def _body(self, system: str, prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_predict": self.max_output_tokens, "num_ctx": self.num_ctx},
        }

    def generate(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        body = self._body(system, prompt, stream=False)
        if json_schema:
            body["format"] = json_schema
        t0 = time.perf_counter()
        resp = self._request("POST", "/api/chat", json=body)
        if resp.status_code == 404:
            raise self._not_pulled()
        if resp.status_code >= 400:
            raise LLMUnavailableError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        self._check_finished(data, time.perf_counter() - t0)
        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMUnavailableError("empty response from model")
        return text

    def generate_stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield the reply as Ollama writes it. A reply that turns out to be cut off raises
        at the end, so it is never reported as a complete answer."""
        t0 = time.perf_counter()
        produced = False
        try:
            with self._client.stream(
                "POST", "/api/chat", json=self._body(system, prompt, stream=True)
            ) as resp:
                if resp.status_code == 404:
                    raise self._not_pulled()
                if resp.status_code >= 400:
                    resp.read()
                    raise LLMUnavailableError(
                        f"Ollama HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise LLMUnavailableError(f"Ollama error: {str(data['error'])[:200]}")
                    piece = (data.get("message") or {}).get("content") or ""
                    if piece:
                        produced = True
                        yield piece
                    if data.get("done"):
                        self._check_finished(data, time.perf_counter() - t0)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama not reachable at {self.base_url} ({type(exc).__name__}). "
                "Is it running?"
            ) from exc
        if not produced:
            raise LLMUnavailableError("empty response from model")

    def _check_finished(self, data: dict, seconds: float) -> None:
        """Record usage from Ollama's final message and reject a damaged reply."""
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        self.usage["calls"] += 1
        self.usage["seconds"] = round(self.usage["seconds"] + seconds, 1)
        self.usage["max_seconds"] = round(max(self.usage["max_seconds"], seconds), 1)
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["output_tokens"] += data.get("eval_count", 0) or 0

        if data.get("done_reason") == "length":
            raise LLMUnavailableError(
                f"Ollama reply hit the output limit ({self.max_output_tokens} tokens)"
            )
        if prompt_tokens > self.num_ctx - self.max_output_tokens:
            raise LLMUnavailableError(
                f"prompt of {prompt_tokens} tokens may not fit num_ctx={self.num_ctx}"
            )
