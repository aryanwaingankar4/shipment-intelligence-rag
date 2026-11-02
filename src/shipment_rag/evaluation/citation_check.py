"""Does each cited passage support the sentence citing it? Checked one citation at a time.

Code finds each sentence and its [n] markers and hands the checker only that sentence and the
passage(s) it cites, so the checker never has to work out which passage is number n (the step
a small local judge got wrong when it saw all the passages at once).

Two interchangeable checkers: an LLM asked one question, or a small NLI cross-encoder.
Known limits:
- A claim that stretches what a passage says can pass: calibration case C4 ("breaking the storage
  rules carries a fine of Rs. 5,000", citing a penalty that is only for a late shipment log) scores
  0.979 entailment. The judge's correctness and faithfulness scores catch that answer instead.
- A sentence that needs earlier sentences for context ("It was rejected [1]") is judged alone.
- A sentence citing several passages is checked against them together, so one irrelevant marker
  among good ones is not caught.
"""

import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

_MARKER = re.compile(r"\[(\d+)\]")
_MARKERS_AFTER_PUNCT = re.compile(r"([.!?])((?:\s*\[\d+\])+)")
_SENTENCE_END = re.compile(
    r"(?<!\bRs\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bNo\.)(?<!\bvs\.)(?<!\bPvt\.)"
    r"(?<=[.!?])\s+(?=[A-Z(\"])"
)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

SUPPORT_SYSTEM = (
    "You check one citation. Use only the passage you are given; never use outside knowledge."
)
SUPPORT_PROMPT = """Passage:
{passage}

Statement:
{claim}

Does this passage itself state the facts in the statement?
- evidence: copy the words of the passage that state those facts, or "" if there are none.
- supported: true only if the passage states all the facts in the statement. Different wording
  is fine. A fact that is missing from the passage, or different in it, means false."""
SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {"evidence": {"type": "string"}, "supported": {"type": "boolean"}},
    "required": ["evidence", "supported"],
}


@dataclass
class CitedClaim:
    markers: list[int]
    claim: str  # the sentence, without its [n] markers


@dataclass
class CitationCheck:
    markers: list[int]
    claim: str
    supported: bool
    detail: str = ""


class SupportChecker(Protocol):
    name: str

    def supports(self, passage: str, claim: str) -> tuple[bool, str]: ...


def cited_claims(answer: str) -> list[CitedClaim]:
    """Each sentence that carries [n] markers. 'Fact. [1]' counts the same as 'Fact [1].'"""
    text = _MARKERS_AFTER_PUNCT.sub(lambda m: f"{m.group(2)}{m.group(1)}", answer)
    claims = []
    for line in text.splitlines():
        for sentence in _SENTENCE_END.split(_LIST_PREFIX.sub("", line)):
            markers = [int(n) for n in _MARKER.findall(sentence)]
            if markers:
                claim = re.sub(r"\s+", " ", _MARKER.sub("", sentence))
                claim = re.sub(r"\s+([.,;:!?)])", r"\1", claim).strip()
                claims.append(CitedClaim(list(dict.fromkeys(markers)), claim))
    return claims


def check_citations(checker: SupportChecker, answer: str,
                    passages: list[str]) -> list[CitationCheck]:
    checks = []
    for c in cited_claims(answer):
        if not all(1 <= n <= len(passages) for n in c.markers):
            checks.append(CitationCheck(c.markers, c.claim, False, "no such passage"))
            continue
        cited = "\n\n".join(passages[n - 1] for n in c.markers)
        supported, detail = checker.supports(cited, c.claim)
        checks.append(CitationCheck(c.markers, c.claim, supported, detail))
    return checks


def citation_label(checks: list[CitationCheck]) -> str:
    if not checks:
        return "no_citations"
    return "all_supported" if all(c.supported for c in checks) else "some_unsupported"


class LLMSupportChecker:
    def __init__(self, llm):
        self.llm = llm
        self.name = f"LLM `{llm.model}`"

    def supports(self, passage: str, claim: str) -> tuple[bool, str]:
        raw = self.llm.generate(
            SUPPORT_SYSTEM, SUPPORT_PROMPT.format(passage=passage, claim=claim),
            json_schema=SUPPORT_SCHEMA,
        )
        data = json.loads(raw)
        return bool(data["supported"]), str(data["evidence"])


class NLISupportChecker:
    """Supported means P(entailment) for (passage, claim) is at least min_entailment."""

    def __init__(self, model_name: str, min_entailment: float = 0.9):
        self.model_name = model_name
        self.min_entailment = min_entailment
        self.name = f"NLI `{model_name}` (entailment >= {min_entailment})"

    @cached_property
    def _model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, device="cpu")

    @cached_property
    def _labels(self) -> dict[int, str]:
        return {int(i): label.lower() for i, label in self._model.config.id2label.items()}

    def supports(self, passage: str, claim: str) -> tuple[bool, str]:
        logits = [float(s) for s in self._model.predict([(passage, claim)])[0]]
        total = sum(math.exp(s) for s in logits)
        probs = {self._labels[i]: math.exp(s) / total for i, s in enumerate(logits)}
        entailment = probs["entailment"]
        return entailment >= self.min_entailment, f"entailment {entailment:.3f}"
