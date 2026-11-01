import pytest
from fastapi.testclient import TestClient

from shipment_rag.api.main import create_app
from shipment_rag.config import Settings
from shipment_rag.llm import LLMUnavailableError
from shipment_rag.pipeline import RAGPipeline
from tests.conftest import FakeLLM, FakeRetriever


def client_for(pipeline, **settings):
    app = create_app(Settings(_env_file=None, log_json=False, **settings), pipeline=pipeline)
    return TestClient(app)


@pytest.fixture
def pipeline(chunk):
    return RAGPipeline(FakeRetriever([chunk]), FakeLLM())


def test_health_and_ready(pipeline):
    with client_for(pipeline) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/ready").json() == {"status": "ready", "chunks": 1}


def test_ready_fails_on_empty_index():
    with client_for(RAGPipeline(FakeRetriever([]), FakeLLM())) as c:
        assert c.get("/ready").status_code == 503


def test_query_returns_answer_sources_and_request_id(pipeline):
    with client_for(pipeline) as c:
        r = c.post("/v1/query", json={"question": "TL-COLD-9001 storage?"},
                   headers={"x-request-id": "abc123"})
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] == "abc123" and r.headers["x-request-id"] == "abc123"
    assert body["sources"][0]["source"] == "product_manual.txt"
    assert body["refused"] is False


def test_query_validation(pipeline):
    with client_for(pipeline) as c:
        assert c.post("/v1/query", json={"question": ""}).status_code == 422
        assert c.post("/v1/query", json={"question": "x" * 1001}).status_code == 422


def test_api_key_enforced_when_configured(pipeline):
    with client_for(pipeline, api_keys="k1,k2") as c:
        assert c.post("/v1/query", json={"question": "q"}).status_code == 401
        assert c.post("/v1/query", json={"question": "q"},
                      headers={"x-api-key": "nope"}).status_code == 401
        assert c.post("/v1/query", json={"question": "q"},
                      headers={"x-api-key": "k2"}).status_code == 200


def test_llm_outage_maps_to_503(chunk):
    p = RAGPipeline(FakeRetriever([chunk]), FakeLLM(exc=LLMUnavailableError("quota")))
    with client_for(p) as c:
        r = c.post("/v1/query", json={"question": "q"})
    assert r.status_code == 503
    assert "quota" not in r.text  # internal error details are not leaked
