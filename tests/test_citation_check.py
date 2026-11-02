"""One-citation-at-a-time checks with stub checkers (no model downloads, no network)."""

import json

from shipment_rag.config import Settings
from shipment_rag.evaluation.citation_check import (
    SUPPORT_SCHEMA,
    LLMSupportChecker,
    NLISupportChecker,
    check_citations,
    citation_label,
    cited_claims,
)
from shipment_rag.factory import build_citation_checker


class Stub:
    name = "stub"

    def __init__(self, answer=True):
        self.answer, self.calls = answer, []

    def supports(self, passage, claim):
        self.calls.append((passage, claim))
        return self.answer, "stub"


def test_each_sentence_keeps_its_own_markers():
    claims = cited_claims("Quarantine at once [1]. The log is due in 24 hours [2].")
    assert [(c.markers, c.claim) for c in claims] == [
        ([1], "Quarantine at once."), ([2], "The log is due in 24 hours.")
    ]


def test_marker_after_the_full_stop_belongs_to_that_sentence():
    claims = cited_claims("First fact. [1] Second fact. [2]")
    assert [(c.markers, c.claim) for c in claims] == [([1], "First fact."), ([2], "Second fact.")]


def test_abbreviations_and_decimals_do_not_split_a_sentence():
    [c] = cited_claims("It reached 11.2°C and cost Rs. 5,000 at Medco Pvt. Ltd [1].")
    assert c.claim == "It reached 11.2°C and cost Rs. 5,000 at Medco Pvt. Ltd."


def test_bullets_are_split_and_unmarked_lines_are_ignored():
    claims = cited_claims("The rules are:\n- Log it [1]\n* Tell QA [2]\n1. Hold it [1]\nDone.")
    assert [(c.markers, c.claim) for c in claims] == [
        ([1], "Log it"), ([2], "Tell QA"), ([1], "Hold it")
    ]


def test_checker_sees_only_the_cited_passage():
    stub = Stub()
    checks = check_citations(stub, "Imran is the contact [2].", ["about Anil", "about Imran"])
    assert stub.calls == [("about Imran", "Imran is the contact.")]
    assert citation_label(checks) == "all_supported"


def test_several_markers_are_checked_against_those_passages_together():
    stub = Stub()
    check_citations(stub, "A and B [1][3].", ["pa", "pb", "pc"])
    assert stub.calls == [("pa\n\npc", "A and B.")]


def test_marker_for_a_missing_passage_fails_without_asking_the_checker():
    stub = Stub()
    [c] = check_citations(stub, "Fact [5].", ["only one"])
    assert not c.supported and c.detail == "no such passage" and stub.calls == []


def test_labels():
    assert citation_label(check_citations(Stub(), "No markers here.", ["p"])) == "no_citations"
    assert citation_label(check_citations(Stub(False), "Fact [1].", ["p"])) == "some_unsupported"


def test_one_bad_sentence_makes_the_answer_some_unsupported():
    class SecondFails(Stub):
        def supports(self, passage, claim):
            return "Second" not in claim, ""

    checks = check_citations(SecondFails(), "First [1]. Second [1].", ["p"])
    assert [c.supported for c in checks] == [True, False]
    assert citation_label(checks) == "some_unsupported"


def test_llm_checker_asks_about_one_passage_with_a_schema():
    class FakeLLM:
        model = "fake"

        def generate(self, system, prompt, json_schema=None):
            self.prompt, self.schema = prompt, json_schema
            return json.dumps({"evidence": "36 hours", "supported": True})

    llm = FakeLLM()
    assert LLMSupportChecker(llm).supports("SLA 36 hours", "It takes 36 hours.") == (
        True, "36 hours")
    assert "SLA 36 hours" in llm.prompt and "It takes 36 hours." in llm.prompt
    assert llm.schema == SUPPORT_SCHEMA


def test_nli_checker_needs_the_minimum_entailment_in_the_models_label_order():
    class FakeConfig:
        id2label = {0: "contradiction", 1: "ENTAILMENT", 2: "neutral"}

    class FakeModel:
        config = FakeConfig()

        def __init__(self, scores):
            self.scores = scores

        def predict(self, pairs):
            return [self.scores]

    def checker(scores):
        c = NLISupportChecker("fake")
        c.__dict__["_model"] = FakeModel(scores)  # no download
        return c

    supported, detail = checker([0.1, 5.0, 0.2]).supports("p", "c")
    assert supported is True and detail.startswith("entailment 0.9")
    assert checker([5.0, 0.1, 0.2]).supports("p", "c")[0] is False
    assert checker([0.1, 0.2, 5.0]).supports("p", "c")[0] is False
    # top label is entailment but the model is unsure (calibration case C8): not supported
    assert checker([-1.0, 1.0, 0.9]).supports("p", "c")[0] is False


def test_factory_builds_the_configured_checker():
    llm = type("L", (), {"model": "m"})()
    nli = build_citation_checker(Settings(_env_file=None, citation_checker="nli"), llm)
    assert isinstance(nli, NLISupportChecker)
    assert nli.model_name == "cross-encoder/nli-deberta-v3-small"
    assert nli.min_entailment == 0.9
    assert Settings(_env_file=None).citation_checker == "nli"  # the default
    assert isinstance(build_citation_checker(Settings(_env_file=None, citation_checker="llm"), llm),
                      LLMSupportChecker)
