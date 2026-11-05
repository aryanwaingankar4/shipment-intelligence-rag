"""Generation eval with a fake pipeline and a scripted judge (no network)."""

import json

import pytest

from shipment_rag.evaluation import generation_eval as ge
from shipment_rag.evaluation.llm_cache import CachedLLM, RateLimiter
from shipment_rag.llm import LLMUnavailableError
from shipment_rag.models import RAGAnswer, RetrievedChunk


def chunk(source, text="passage"):
    return RetrievedChunk(text=text, source=source, section="", embed_score=0.5,
                          rerank_score=1.0)


def item(id_, type_="answerable", facts=(("36 hours",),), sources=("vendor_list.txt",)):
    return ge.GoldenItem(id=id_, type=type_, question=f"question {id_}",
                         reference_answer="ref", key_facts=[list(f) for f in facts],
                         expected_sources=list(sources))


class FakePipeline:
    def __init__(self, answers):
        self.answers = answers

    def answer(self, question):
        a = self.answers[question]
        if isinstance(a, Exception):
            raise a
        return a


class ScriptedJudge:
    def __init__(self, verdict=None, exc=None):
        self.verdict = verdict or {"is_refusal": False, "correctness": 2,
                                   "unsupported_claims": [], "faithful": True,
                                   "citations": "all_supported", "reason": "ok"}
        self.exc, self.calls = exc, []

    def generate(self, system, prompt, json_schema=None):
        self.calls.append((system, prompt, json_schema))
        if self.exc:
            raise self.exc
        return json.dumps(self.verdict)


class AlwaysSupported:
    name = "always"

    def supports(self, passage, claim):
        return True, ""


CHECKER = AlwaysSupported()


def answered(text, context):
    return RAGAnswer(text, context, refused=False, latency_ms=5, context=context,
                     llm_called=True)


def test_answer_is_judged_with_numbered_passages_and_deterministic_checks():
    ctx = [chunk("shipment_log.txt"), chunk("vendor_list.txt", "SLA: 36 hours")]
    pipe = FakePipeline({"question G1": answered("It is 36 hours [2].", ctx)})
    judge = ScriptedJudge()
    [r] = ge.run(pipe, judge, CHECKER, [item("G1")])
    _, prompt, schema = judge.calls[0]
    assert "[2] (source: vendor_list.txt)\nSLA: 36 hours" in prompt
    assert schema == ge.JUDGE_SCHEMA
    assert r.cited == [2] and r.citations_valid and r.cited_expected_source
    assert r.key_fact_hits == 1 and r.verdict.correctness == 2


def test_retrieval_refusal_is_scored_by_rule_without_a_judge_call():
    refused = RAGAnswer("I don't have enough information...", [], refused=True, latency_ms=1)
    pipe = FakePipeline({"question R": refused, "question T": refused, "question A": refused})
    judge = ScriptedJudge()
    results = ge.run(pipe, judge, CHECKER, [item("R", "refusal"), item("T", "trap"), item("A")])
    assert judge.calls == []
    assert [r.verdict.correctness for r in results] == [2, 1, 0]
    assert all(r.retrieval_refused and r.verdict.by_rule for r in results)


def test_invalid_citation_index_and_wrong_source_are_detected():
    ctx = [chunk("shipment_log.txt")]
    pipe = FakePipeline({"question G": answered("Answer [1] and [3].", ctx)})
    [r] = ge.run(pipe, ScriptedJudge(), CHECKER, [item("G")])
    assert not r.citations_valid
    assert not r.cited_expected_source  # only shipment_log was cited; vendor_list expected


def test_errors_are_recorded_and_run_stops_after_repeated_failures():
    items = [item(f"E{i}") for i in range(5)]
    pipe = FakePipeline({f"question E{i}": LLMUnavailableError("429") for i in range(5)})
    results = ge.run(pipe, ScriptedJudge(), CHECKER, items, max_consecutive_errors=3)
    assert [bool(r.error) for r in results] == [True] * 5
    assert "not run" in results[3].error and "not run" in results[4].error
    assert ge.summarize(results)["n_scored"] == 0


@pytest.mark.parametrize("answer,expected", [
    ("Recorded −22 °C min and -18°C max", 2),   # unicode minus, space before °
    ("Only -22°C was recorded", 1),
])
def test_key_fact_matching_normalizes_symbols(answer, expected):
    assert ge.key_fact_hits(answer, [["-22°c"], ["-18°c"]]) == expected


def test_summary_counts():
    def res(type_, correctness, refusal=False, faithful=True, cited=(1,), by_rule=False):
        r = ge.GenResult(item("X", type_), answer="a", sources=["vendor_list.txt"],
                         cited=list(cited), retrieval_refused=by_rule)
        r.verdict = ge.Verdict(refusal, correctness, [], faithful,
                               "all_supported" if cited else "no_citations", "", by_rule)
        return r

    s = ge.summarize([
        res("answerable", 2), res("answerable", 1, faithful=False),
        res("answerable", 0, True, cited=()),
        res("refusal", 2, True, cited=(), by_rule=True), res("refusal", 0),
        res("trap", 2), res("trap", 0),
    ])
    assert s["answer_correct"] == (1, 3) and s["answer_partial"] == (1, 3)
    assert s["wrongly_refused"] == (1, 3)
    assert s["faithful"] == (4, 5)  # generated (non-refusal) answers only
    assert s["has_citation"] == (2, 2)
    assert s["refused_correctly"] == (1, 2) and s["refused_at_retrieval"] == (1, 2)
    assert s["trap_handled"] == (1, 2) and s["trap_invented"] == (1, 2)
    assert "| fully correct (judge: 2/2) | 1/3 (33%) |" in ge.summary_table(s)


def test_calibration_compares_judge_to_planted_flaws():
    case = {"id": "C1", "flaw": "wrong number", "question": "q", "reference_answer": "36 h",
            "passages": [{"source": "v.txt", "text": "36 hours"}], "answer": "48 hours [1]",
            "expect": {"correctness_max": 0, "faithful": False}}
    good = ScriptedJudge({"is_refusal": False, "correctness": 0, "unsupported_claims": ["48"],
                          "faithful": False, "citations": "some_unsupported", "reason": ""})
    [c] = ge.run_calibration(good, CHECKER, [case])
    assert all(ok for _, ok in c["checks"])
    lenient = ScriptedJudge()  # says everything is fine
    [c] = ge.run_calibration(lenient, CHECKER, [case])
    assert not any(ok for _, ok in c["checks"])
    assert "0/2" in ge.calibration_table([c])


# ---------------- cache + rate limiter ----------------

class CountingLLM:
    model = "fake-model"

    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def cache_identity(self):
        return {"model": self.model, "temperature": 0.1}

    def generate(self, system, prompt, json_schema=None):
        self.calls += 1
        if self.fail:
            raise LLMUnavailableError("quota")
        return f"answer to {prompt}"


def test_cache_hits_skip_the_api_and_failures_are_not_cached(tmp_path):
    llm = CountingLLM()
    c = CachedLLM(llm, tmp_path, "answer")
    assert c.generate("s", "p") == c.generate("s", "p") == "answer to p"
    assert (llm.calls, c.hits, c.misses) == (1, 1, 1)
    assert CachedLLM(llm, tmp_path, "judge").generate("s", "p") and llm.calls == 2  # namespace
    llm.model = "other-model"
    assert c.generate("s", "p") and llm.calls == 3  # model settings are part of the key

    broken = CachedLLM(CountingLLM(fail=True), tmp_path / "b", "answer")
    with pytest.raises(LLMUnavailableError):
        broken.generate("s", "p")
    assert not any((tmp_path / "b").rglob("*.json"))


def test_cache_passes_json_schema_and_exposes_wrapped_attributes(tmp_path):
    seen = {}

    class SchemaLLM(CountingLLM):
        def generate(self, system, prompt, json_schema=None):
            seen["schema"] = json_schema
            return "{}"

    c = CachedLLM(SchemaLLM(), tmp_path, "judge")
    c.generate("s", "p", json_schema={"type": "object"})
    assert seen["schema"] == {"type": "object"} and c.model == "fake-model"


def test_rate_limiter_spaces_calls():
    now, slept = [0.0], []
    rl = RateLimiter(rpm=6, clock=lambda: now[0], sleep=lambda s: (slept.append(s),
                                                                    now.__setitem__(0, now[0] + s)))
    rl.wait()          # first call: no wait
    now[0] += 4.0
    rl.wait()          # 10 s interval, 4 s elapsed -> sleep 6
    assert slept == [pytest.approx(6.0)]


def test_daily_quota_skips_items_but_keeps_scoring_what_needs_no_new_call():
    from shipment_rag.llm import QuotaExhaustedError

    refused = RAGAnswer("I don't have enough information...", [], refused=True, latency_ms=1)
    pipe = FakePipeline({"question Q0": QuotaExhaustedError("daily"),
                         "question Q1": QuotaExhaustedError("daily"),
                         "question Q2": refused,  # refused at retrieval: no LLM needed
                         "question Q3": QuotaExhaustedError("daily")})
    items = [item("Q0"), item("Q1"), item("Q2", "refusal"), item("Q3")]
    results = ge.run(pipe, ScriptedJudge(), CHECKER, items, max_consecutive_errors=2)
    assert [r.verdict is not None for r in results] == [False, False, True, False]
    assert all("not run" in r.error for r in (results[0], results[1], results[3]))

    case = {"id": "C", "flaw": "x", "question": "q", "reference_answer": "r",
            "passages": [], "answer": "a", "expect": {"faithful": True}}
    judge = ScriptedJudge(exc=QuotaExhaustedError("daily"))
    cal = ge.run_calibration(judge, CHECKER, [case, dict(case, id="D")])
    assert len(judge.calls) == 1 and "not run" in cal[1]["error"]
    assert "(0/2 cases scored)" in ge.calibration_table(cal)
    assert not ge.calibration_passed(cal)


def test_calibration_gate_requires_every_check_and_every_case():
    ok = {"id": "A", "flaw": "x", "checks": [("c", True)]}
    assert ge.calibration_passed([ok])
    assert not ge.calibration_passed([ok, {"id": "B", "flaw": "y", "checks": [("c", False)]}])
    assert not ge.calibration_passed([ok, {"id": "C", "flaw": "z", "checks": [], "error": "x"}])
    assert not ge.calibration_passed([])


def test_cache_short_circuits_after_daily_quota_but_still_serves_hits(tmp_path):
    from shipment_rag.llm import QuotaExhaustedError

    llm = CountingLLM()
    c = CachedLLM(llm, tmp_path, "answer")
    c.generate("s", "cached")
    llm.generate = lambda *a, **k: (_ for _ in ()).throw(QuotaExhaustedError("daily"))
    with pytest.raises(QuotaExhaustedError):
        c.generate("s", "new 1")
    calls_before = llm.calls
    with pytest.raises(QuotaExhaustedError):
        c.generate("s", "new 2")  # no API attempt
    assert llm.calls == calls_before
    assert c.generate("s", "cached") == "answer to cached"  # cache still served


def test_cli_output_survives_a_redirected_windows_console(monkeypatch):
    import io
    import sys

    from shipment_rag.cli import utf8_output

    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    utf8_output()
    print("correctness ≤ 0")
    sys.stdout.flush()
    assert "≤" in raw.getvalue().decode("utf-8")


def test_calibration_checks_each_cited_sentence():
    class SecondFails(AlwaysSupported):
        def supports(self, passage, claim):
            return "Second" not in claim, ""

    case = {"id": "C8", "flaw": "one wrong citation", "question": "q", "reference_answer": "r",
            "passages": [{"source": "v.txt", "text": "p"}], "answer": "First [1]. Second [1].",
            "expect": {"citation_claims": [True, False]}}
    [c] = ge.run_calibration(ScriptedJudge(), SecondFails(), [case])
    assert c["checks"] == [("citation 1 supported = True", True),
                           ("citation 2 supported = False", True)]
    assert c["verdict"].citations == "some_unsupported"
    [c] = ge.run_calibration(ScriptedJudge(), CHECKER, [case])
    assert c["checks"][1] == ("citation 2 supported = False", False)
    assert not ge.calibration_passed([c])
    # a split that finds the wrong number of sentences must fail, not pass silently
    [c] = ge.run_calibration(ScriptedJudge(), CHECKER, [dict(case, answer="Only one [1].")])
    assert c["checks"] == [("2 cited sentences found", False)]


def test_known_limit_is_shown_but_does_not_fail_the_gate():
    case = {"id": "C4", "flaw": "stretched claim", "question": "q", "reference_answer": "r",
            "passages": [{"source": "v.txt", "text": "p"}], "answer": "A stretched claim [1].",
            "expect": {"correctness_min": 2, "known_limit_citation_claims": [False]}}
    [c] = ge.run_calibration(ScriptedJudge(), CHECKER, [case])
    assert c["known_limits"] == [("citation 1 supported = False", False)]
    assert ge.calibration_passed([c])
    assert "known limit, not gated: citation 1 supported = False ❌" in ge.calibration_table([c])

