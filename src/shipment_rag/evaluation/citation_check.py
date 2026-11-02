"""Does each cited passage support the sentence citing it? Checked one citation at a time.

Code finds each sentence and its [n] markers and hands the checker only that sentence and the
passage(s) it cites, so the checker never has to work out which passage is number n (the step
a small local judge got wrong when it saw all the passages at once).

Two interchangeable checkers: an LLM asked one question, or a small NLI cross-encoder.
A retrieved chunk often holds several unrelated rules, which makes NLI answer "neutral" even when
one of them states the claim exactly, so NLI scores the whole passage and each line of it.
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
    claims, intro = [], ""
    for line in text.splitlines():
        if not line.strip():
            continue
        if _LIST_PREFIX.match(line):  # a bullet alone ("Tax Invoice") is not a statement
            line = f"{intro} {_LIST_PREFIX.sub('', line)}".strip()
        else:
            intro = ""
            if line.rstrip().endswith(":") and not _MARKER.search(line):
                intro = line.strip().rstrip(":")
                continue
        for sentence in _SENTENCE_END.split(line):
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


def _premises(passage: str) -> list[str]:
    """The whole passage, then each line with its section header kept for context."""
    premises = [passage]
    for block in passage.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        header = lines[0] if lines and lines[0].startswith("===") else ""
        premises += [f"{header}\n{line}" if header else line for line in lines if line != header]
    return list(dict.fromkeys(premises))


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
    """Supported means P(entailment) reaches min_entailment for the whole passage or any line."""

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
        premises = _premises(passage)
        best = max(self._entailment(row) for row in self._model.predict(
            [(premise, claim) for premise in premises]))
        return best >= self.min_entailment, f"entailment {best:.3f}"

    def _entailment(self, logits) -> float:
        values = [float(s) for s in logits]
        total = sum(math.exp(s) for s in values)
        return {self._labels[i]: math.exp(s) / total for i, s in enumerate(values)}["entailment"]
