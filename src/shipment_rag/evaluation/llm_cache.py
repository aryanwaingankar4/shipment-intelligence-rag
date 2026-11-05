"""Quota-aware LLM wrapper for evaluation runs.

- Disk cache: the key is a hash of (namespace, model settings, system prompt, prompt,
  schema). Re-running an eval with unchanged inputs makes zero API calls, and a run that
  stops on a quota error resumes from where it got to.
- Rate limiter: spaces real (uncached) calls at least 60/rpm seconds apart.
Failures are never cached. After a daily-quota error, further *uncached* calls fail at once
(no API request) while cached responses are still served, so a run keeps scoring
everything it already has.
"""

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from shipment_rag.llm import QuotaExhaustedError


class RateLimiter:
    def __init__(self, rpm: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None and self.interval:
            remaining = self.interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


class CachedLLM:
    def __init__(self, llm, cache_dir: Path, namespace: str,
                 rate_limiter: RateLimiter | None = None, enabled: bool = True):
        self.llm = llm
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace
        self.rate_limiter = rate_limiter
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.exhausted: Exception | None = None  # set after a daily-quota error

    def _key(self, system: str, prompt: str, json_schema: dict | None) -> str:
        identity = getattr(self.llm, "cache_identity", lambda: {"model": type(self.llm).__name__})()
        blob = json.dumps([self.namespace, identity, system, prompt, json_schema],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        key = self._key(system, prompt, json_schema)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if self.enabled and path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        if self.exhausted is not None:
            raise self.exhausted  # no point asking: the daily quota is used up
        if self.rate_limiter:
            self.rate_limiter.wait()
        try:
            text = self.llm.generate(system, prompt, json_schema=json_schema) if json_schema \
                else self.llm.generate(system, prompt)
        except QuotaExhaustedError as exc:
            self.exhausted = exc
            raise
        self.misses += 1
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            # atomic write so an interrupted run never leaves a half-written entry
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"text": text}, f, ensure_ascii=False)
            os.replace(tmp, path)
        return text

    def __getattr__(self, name):
        # check_model, usage, model, ... come from the wrapped client
        return getattr(self.llm, name)
