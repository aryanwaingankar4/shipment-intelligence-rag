"""Shipment table, router and their wiring into the pipeline and API. No models, no network."""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shipment_rag.api.main import create_app
from shipment_rag.config import Settings
from shipment_rag.evaluation.generation_eval import load_golden
from shipment_rag.factory import build_structured_router
from shipment_rag.pipeline import REFUSAL, RAGPipeline
from shipment_rag.structured.router import ROUTE_TABLE, StructuredRouter
from shipment_rag.structured.store import ShipmentStore, parse_shipment_log
from tests.conftest import FakeLLM, FakeRetriever

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "docs" / "shipment_log.txt"
EVAL = json.loads((ROOT / "data" / "eval" / "structured_eval.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store():
    return ShipmentStore.from_file(LOG)


@pytest.fixture(scope="module")
def router(store):
    return StructuredRouter(store)


# ---------------- parsing ----------------

def test_table_matches_the_logs_own_summary_line(store):
    """The log states its own totals, which is an independent check on the parser."""
    assert store.count() == 10
    assert {g["status"]: g["shipments"] for g in store.by_status()} == {
        "DELIVERED": 7, "QUARANTINED": 1, "REJECTED": 1, "DELAYED": 1}
    assert [r["shipment_id"] for r in store.rows({"breach": 1})] == ["SH-2024-003", "SH-2024-007"]


def test_every_record_in_the_log_is_parsed():
    text = LOG.read_text(encoding="utf-8")
    assert len(parse_shipment_log(text)) == text.count("Shipment ID:")


def test_parsed_fields(store):
    frozen = store.get("SH-2024-007")
    assert (frozen["product"], frozen["quantity"], frozen["unit"]) == ("TL-FROZ-7700", 40, "packs")
    assert (frozen["temp_min"], frozen["temp_max"]) == (-19.0, -8.0)
    assert frozen["breach"] == 1 and frozen["breach_detail"] == "rose above -10°C"
    assert "CDSCO notified" in frozen["action"]
    delayed = store.get("SH-2024-005")
    assert (delayed["status"], delayed["status_note"]) == ("DELAYED", "truck breakdown")
    assert (delayed["delay_days"], delayed["accepted_by"]) == (3, "Sandeep Arora")
    ambient = store.get("SH-2024-002")
    assert (ambient["temp_min"], ambient["temp_max"], ambient["temp_ambient"]) == (22.0, 22.0, 1)
    assert store.get("SH-2024-008")["destination"] == "District Health Office Nagpur"


def test_malformed_record_is_an_error_not_a_missing_row():
    bad = "=== SHIPMENT SH-2024-099 ===\nShipment ID: SH-2024-099 | Product: X\n"
    with pytest.raises(ValueError, match="SH-2024-099"):
        parse_shipment_log(bad)


def test_summary_section_is_not_a_shipment():
    assert "SUMMARY" not in {s.shipment_id for s in parse_shipment_log(LOG.read_text("utf-8"))}


# ---------------- safety ----------------

@pytest.mark.parametrize("sql", [
    "DROP TABLE shipments",
    "DELETE FROM shipments",
    "UPDATE shipments SET status = 'DELIVERED'",
    "INSERT INTO shipments (shipment_id) VALUES ('x')",
    "CREATE TABLE t (a)",
    "ATTACH DATABASE 'evil.db' AS evil",
    "PRAGMA query_only = OFF",
])
def test_connection_refuses_anything_but_reading(store, sql):
    with pytest.raises(sqlite3.Error):
        store._conn.execute(sql)
    assert store.count() == 10


@pytest.mark.parametrize("name", ["destination", "accepted_by", "status", "product"])
def test_values_are_bound_never_part_of_the_sql(store, name):
    evil = "x'; DROP TABLE shipments;--"
    assert store.rows({name: evil}) == []
    assert store.count({name: "' OR '1'='1"}) == 0
    assert store.count() == 10


def test_wildcards_in_a_value_match_literally(store):
    assert store.count({"destination": "%"}) == 0
    assert store.count({"destination": "_"}) == 0


def test_a_filter_outside_the_allowed_list_is_rejected(store):
    with pytest.raises(ValueError, match="unknown filter"):
        store.rows({"status = 'x' OR 1=1 --": "y"})


def test_a_bad_template_name_cannot_run_anything_but_a_template(router):
    from shipment_rag.structured.router import StructuredQuery

    with pytest.raises(AttributeError):
        router.run(StructuredQuery("__class__", {}))


# ---------------- routing + answers ----------------

@pytest.mark.parametrize("case", [c for c in EVAL if c["route"]], ids=lambda c: c["question"][:45])
def test_table_answers(router, case):
    result = router.answer(case["question"])
    assert result is not None, "should have been answered from the table"
    assert result.query.template == case["route"]
    for text in case["contains"]:
        assert text in result.answer
    assert result.refused is case.get("refused", False)


@pytest.mark.parametrize("case", [c for c in EVAL if not c["route"]],
                         ids=lambda c: c["question"][:45])
def test_questions_left_for_the_documents(router, case):
    assert router.route(case["question"]) is None


def test_eval_set_ground_truth_comes_from_independent_counts(store):
    """A few expected values recomputed from the raw rows, not from the query templates."""
    rows = store.rows()
    assert sum(r["quantity"] for r in rows if r["product"] == "TL-COLD-9001") == 650
    assert sum(r["quantity"] for r in rows if r["product"] == "TL-DRY-4402") == 1500
    assert max(r["temp_max"] for r in rows) == 27.0
    assert min(r["temp_min"] for r in rows) == -24.0


def test_golden_questions_the_table_claims_are_exactly_these(router):
    """The golden set is mostly narrative: the table must only take field lookups from it."""
    golden = load_golden(ROOT / "data" / "eval" / "golden_qa.json")
    claimed = {g.id for g in golden if router.route(g.question)}
    assert claimed == {"G08", "G09", "G21"}


def test_retrieval_questions_the_table_claims_are_only_lookups_and_counts(router):
    claimed = []
    for name in ["retrieval_eval.json", "user_paraphrases.json"]:
        raw = json.loads((ROOT / "data" / "eval" / name).read_text(encoding="utf-8"))
        cases = raw["cases"] if isinstance(raw, dict) else raw
        claimed += [c["question"] for c in cases if router.route(c["question"])]
    for question in claimed:
        assert router.answer(question).answer


# ---------------- pipeline + API ----------------

class ExplodingRetriever(FakeRetriever):
    def __init__(self):
        super().__init__([])

    def retrieve(self, query):
        raise AssertionError("the table should have answered without retrieval")


def test_table_answer_skips_retrieval_and_the_llm(router):
    llm = FakeLLM()
    result = RAGPipeline(ExplodingRetriever(), llm, structured=router).answer(
        "How many shipments were delivered?")
    assert result.route == ROUTE_TABLE and llm.calls == [] and result.llm_called is False
    assert result.answer.startswith("7 shipments") and not result.refused
    assert {c.source for c in result.sources} == {"shipment_log.txt"}
    assert result.structured_query["template"] == "count"
    assert result.structured_query["params"] == {"filters": {"status": "DELIVERED"}}
    assert len(result.structured_query["rows"]) == 7


def test_narrative_question_takes_the_document_path(router, chunk):
    llm = FakeLLM()
    result = RAGPipeline(FakeRetriever([chunk]), llm, structured=router).answer(
        "Why was SH-2024-003 quarantined?")
    assert result.route == "documents" and result.structured_query is None and len(llm.calls) == 1


def test_a_broken_table_falls_back_to_the_documents(chunk):
    class Broken:
        def answer(self, question):
            raise sqlite3.OperationalError("boom")

    result = RAGPipeline(FakeRetriever([chunk]), FakeLLM(), structured=Broken()).answer("anything")
    assert result.route == "documents" and result.answer


def test_no_router_means_documents_only(chunk):
    result = RAGPipeline(FakeRetriever([chunk]), FakeLLM()).answer("How many shipments?")
    assert result.route == "documents"


def test_refusal_on_the_document_path_still_reports_its_route(router):
    result = RAGPipeline(FakeRetriever([]), FakeLLM(), structured=router).answer(
        "What's the capital of France?")
    assert result.refused and result.answer == REFUSAL and result.route == "documents"


def client_for(pipeline):
    return TestClient(create_app(Settings(_env_file=None, log_json=False), pipeline=pipeline))


def test_api_shows_which_path_answered(router, chunk):
    pipeline = RAGPipeline(FakeRetriever([chunk]), FakeLLM(), structured=router)
    with client_for(pipeline) as c:
        table = c.post("/v1/query", json={"question": "Which shipments were rejected?"}).json()
        docs = c.post("/v1/query", json={"question": "TL-COLD-9001 storage?"}).json()
    assert table["route"] == "shipment_table" and "SH-2024-007" in table["answer"]
    assert table["structured_query"]["template"] == "list"
    assert table["structured_query"]["rows"][0]["shipment_id"] == "SH-2024-007"
    assert table["sources"][0] == {
        "source": "shipment_log.txt", "section": "SHIPMENT SH-2024-007", "rerank_score": None,
        "snippet": "SH-2024-007: TL-FROZ-7700, REJECTED, Pune to SciLab Research Kolkata"}
    assert docs["route"] == "documents" and docs["structured_query"] is None


def test_api_unknown_shipment_is_a_refusal_from_the_table(router, chunk):
    pipeline = RAGPipeline(FakeRetriever([chunk]), FakeLLM(), structured=router)
    with client_for(pipeline) as c:
        body = c.post("/v1/query", json={"question": "What is the status of SH-2024-011?"}).json()
    assert body["refused"] is True and body["sources"] == []
    assert body["route"] == "shipment_table" and "no shipment SH-2024-011" in body["answer"]


# ---------------- factory ----------------

def test_factory_builds_a_router_from_the_docs_folder():
    router = build_structured_router(Settings(_env_file=None, docs_path=ROOT / "data" / "docs"))
    assert router is not None and router.store.count() == 10


def test_factory_can_turn_the_table_off():
    settings = Settings(_env_file=None, docs_path=ROOT / "data" / "docs", structured_queries=False)
    assert build_structured_router(settings) is None


def test_factory_survives_a_missing_log(tmp_path):
    assert build_structured_router(Settings(_env_file=None, docs_path=tmp_path)) is None
