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


def answered(text, context):
    return RAGAnswer(text, context, refused=False, latency_ms=5, context=context,
                     llm_called=True)


def test_answer_is_judged_with_numbered_passages_and_deterministic_checks():
    ctx = [chunk("shipment_log.txt"), chunk("vendor_list.txt", "SLA: 36 hours")]
    pipe = FakePipeline({"question G1": answered("It is 36 hours [2].", ctx)})
    judge = ScriptedJudge()
    [r] = ge.run(pipe, judge, [item("G1")])
    _, prompt, schema = judge.calls[0]
    assert "[2] (source: vendor_list.txt)\nSLA: 36 hours" in prompt
    assert schema == ge.JUDGE_SCHEMA
    assert r.cited == [2] and r.citations_valid and r.cited_expected_source
    assert r.key_fact_hits == 1 and r.verdict.correctness == 2


def test_retrieval_refusal_is_scored_by_rule_without_a_judge_call():
    refused = RAGAnswer("I don't have enough information...", [], refused=True, latency_ms=1)
    pipe = FakePipeline({"question R": refused, "question T": refused, "question A": refused})
    judge = ScriptedJudge()
    results = ge.run(pipe, judge, [item("R", "refusal"), item("T", "trap"), item("A")])
    assert judge.calls == []
    assert [r.verdict.correctness for r in results] == [2, 1, 0]
    assert all(r.retrieval_refused and r.verdict.by_rule for r in results)


def test_invalid_citation_index_and_wrong_source_are_detected():
    ctx = [chunk("shipment_log.txt")]
    pipe = FakePipeline({"question G": answered("Answer [1] and [3].", ctx)})
    [r] = ge.run(pipe, ScriptedJudge(), [item("G")])
    assert not r.citations_valid
    assert not r.cited_expected_source  # only shipment_log was cited; vendor_list expected


def test_errors_are_recorded_and_run_stops_after_repeated_failures():
    items = [item(f"E{i}") for i in range(5)]
    pipe = FakePipeline({f"question E{i}": LLMUnavailableError("429") for i in range(5)})
    results = ge.run(pipe, ScriptedJudge(), items, max_consecutive_errors=3)
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
    [c] = ge.run_calibration(good, [case])
    assert all(ok for _, ok in c["checks"])
    lenient = ScriptedJudge()  # says everything is fine
    [c] = ge.run_calibration(lenient, [case])
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


def test_daily_quota_stops_the_run_and_calibration_immediately():
    from shipment_rag.llm import QuotaExhaustedError

    items = [item(f"Q{i}") for i in range(4)]
    pipe = FakePipeline({f"question Q{i}": QuotaExhaustedError("daily") for i in range(4)})
    results = ge.run(pipe, ScriptedJudge(), items, max_consecutive_errors=3)
    assert "QuotaExhaustedError" in results[0].error
    assert all("daily quota exhausted" in r.error for r in results[1:])

    case = {"id": "C", "flaw": "x", "question": "q", "reference_answer": "r",
            "passages": [], "answer": "a", "expect": {"faithful": True}}
    judge = ScriptedJudge(exc=QuotaExhaustedError("daily"))
    cal = ge.run_calibration(judge, [case, dict(case, id="D")])
    assert len(judge.calls) == 1 and "not run" in cal[1]["error"]
    assert "(0/2 cases scored)" in ge.calibration_table(cal)
