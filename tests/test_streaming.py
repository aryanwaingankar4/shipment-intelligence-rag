"""Streaming: the LLM clients, the pipeline and the SSE endpoint. No models, no network."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors

from shipment_rag.api.main import create_app
from shipment_rag.config import Settings
from shipment_rag.llm import (
    GeminiLLM,
    LLMUnavailableError,
    ModelNotFoundError,
    OllamaLLM,
    QuotaExhaustedError,
)
from shipment_rag.pipeline import REFUSAL, RAGPipeline
from shipment_rag.structured.router import StructuredRouter
from shipment_rag.structured.store import ShipmentStore
from tests.conftest import FakeLLM, FakeRetriever

ROOT = Path(__file__).resolve().parent.parent


# ---------------- Gemini ----------------

def gemini_chunk(text):
    return SimpleNamespace(
        text=text, candidates=[],
        usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=2,
                                       thoughts_token_count=0))


def server_error(code=503):
    return errors.ServerError(code, {"error": {"code": code, "message": "busy",
                                               "status": "UNAVAILABLE"}}, None)


class FakeGeminiModels:
    """Each call plays one script: chunks to yield, with exceptions raised where they appear."""

    def __init__(self, *scripts):
        self.scripts, self.calls = list(scripts), 0

    def generate_content_stream(self, **kwargs):
        script = self.scripts[self.calls]
        self.calls += 1

        def stream():
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield gemini_chunk(item)

        return stream()


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)

    def make(*scripts):
        llm = GeminiLLM(api_key="not-a-real-key", model="m", max_retries=3)
        models = FakeGeminiModels(*scripts)
        llm._client = SimpleNamespace(models=models)
        return llm, models

    return make


def test_gemini_stream_yields_pieces_and_records_usage(gemini):
    llm, _ = gemini(["Stored ", "at ", "2-8°C."])
    assert list(llm.generate_stream("sys", "prompt")) == ["Stored ", "at ", "2-8°C."]
    assert llm.usage["calls"] == 1 and llm.usage["prompt_tokens"] == 5


def test_gemini_stream_skips_chunks_with_no_text(gemini):
    llm, _ = gemini([None, "answer", ""])
    assert list(llm.generate_stream("sys", "prompt")) == ["answer"]


def test_gemini_stream_retries_a_failure_before_the_first_chunk(gemini):
    llm, models = gemini([server_error()], ["answer"])
    assert list(llm.generate_stream("sys", "prompt")) == ["answer"]
    assert models.calls == 2


def test_gemini_stream_does_not_retry_once_text_has_been_sent(gemini):
    """A retry would restart the answer in front of someone who has already read part of it."""
    llm, models = gemini(["first ", server_error()], ["never used"])
    seen = []
    with pytest.raises(LLMUnavailableError):
        for piece in llm.generate_stream("sys", "prompt"):
            seen.append(piece)
    assert seen == ["first "] and models.calls == 1


def test_gemini_stream_gives_up_after_its_retries(gemini):
    llm, models = gemini([server_error()], [server_error()], [server_error()])
    with pytest.raises(LLMUnavailableError):
        list(llm.generate_stream("sys", "prompt"))
    assert models.calls == 3


def test_gemini_stream_with_no_text_is_an_error(gemini):
    llm, _ = gemini([None])
    with pytest.raises(LLMUnavailableError, match="empty"):
        list(llm.generate_stream("sys", "prompt"))


def test_gemini_stream_maps_daily_quota_and_does_not_retry(gemini):
    quota = errors.ClientError(429, {"error": {"code": 429, "details": [{"violations": [
        {"quotaId": "GenerateRequestsPerDay", "quotaValue": "20"}]}]}}, None)
    llm, models = gemini([quota], ["never used"])
    with pytest.raises(QuotaExhaustedError):
        list(llm.generate_stream("sys", "prompt"))
    assert models.calls == 1


def test_a_503_on_the_startup_check_is_a_clean_unavailable_error(gemini):
    """eval-gen stops cleanly on this error type, so a raw SDK error would be a traceback."""
    llm, _ = gemini()

    def fails(model):
        raise server_error()

    llm._client.models.get = fails
    with pytest.raises(LLMUnavailableError, match="ServerError"):
        llm.check_model()


def test_gemini_stream_maps_a_missing_model(gemini):
    missing = errors.ClientError(404, {"error": {"code": 404, "message": "nope"}}, None)
    llm, _ = gemini([missing])
    with pytest.raises(ModelNotFoundError):
        list(llm.generate_stream("sys", "prompt"))


# ---------------- Ollama ----------------

def ndjson(*items):
    return "".join(json.dumps(i) + "\n" for i in items)


def ollama_with(handler):
    return OllamaLLM("m", transport=httpx.MockTransport(handler), max_output_tokens=50, num_ctx=500)


def ollama_lines(*items):
    def handler(request):
        return httpx.Response(200, text=ndjson(*items))

    return ollama_with(handler)


def piece(text):
    return {"message": {"content": text}, "done": False}


def test_ollama_stream_yields_pieces_and_asks_to_stream():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, text=ndjson(
            piece("Stored "), piece("cold."),
            {"message": {"content": ""}, "done": True, "done_reason": "stop",
             "prompt_eval_count": 12, "eval_count": 3}))

    llm = ollama_with(handler)
    assert list(llm.generate_stream("sys", "prompt")) == ["Stored ", "cold."]
    assert seen["stream"] is True and "format" not in seen
    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert llm.usage["prompt_tokens"] == 12 and llm.usage["output_tokens"] == 3


def test_ollama_stream_cut_off_by_the_limit_raises_after_its_text():
    llm = ollama_lines(piece("partial"), {"done": True, "done_reason": "length"})
    seen = []
    with pytest.raises(LLMUnavailableError, match="output limit"):
        for text in llm.generate_stream("sys", "prompt"):
            seen.append(text)
    assert seen == ["partial"]


def test_ollama_stream_reports_an_error_line():
    llm = ollama_lines(piece("partial"), {"error": "model crashed"})
    with pytest.raises(LLMUnavailableError, match="model crashed"):
        list(llm.generate_stream("sys", "prompt"))


def test_ollama_stream_with_no_text_is_an_error():
    llm = ollama_lines({"done": True, "done_reason": "stop"})
    with pytest.raises(LLMUnavailableError, match="empty"):
        list(llm.generate_stream("sys", "prompt"))


def test_ollama_stream_model_not_pulled():
    llm = ollama_with(lambda request: httpx.Response(404, text="not found"))
    with pytest.raises(ModelNotFoundError, match="ollama pull"):
        list(llm.generate_stream("sys", "prompt"))


def test_ollama_stream_http_error():
    llm = ollama_with(lambda request: httpx.Response(500, text="broken"))
    with pytest.raises(LLMUnavailableError, match="500"):
        list(llm.generate_stream("sys", "prompt"))


def test_ollama_stream_server_down():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError, match="not reachable"):
        list(ollama_with(handler).generate_stream("sys", "prompt"))


# ---------------- pipeline ----------------

def kinds(events):
    return [k for k, _ in events]


def test_stream_order_and_text(chunk):
    events = list(RAGPipeline(FakeRetriever([chunk]), FakeLLM()).answer_stream("q"))
    assert kinds(events)[0] == "start" and kinds(events)[-1] == "done"
    assert set(kinds(events)[1:-1]) == {"delta"} and len(events) > 3
    text = "".join(d for k, d in events if k == "delta")
    assert text == events[-1][1].answer == "TL-COLD-9001 is stored at 2°C to 8°C [1]."


def test_stream_gives_the_same_result_as_the_plain_answer(chunk):
    pipeline = RAGPipeline(FakeRetriever([chunk]), FakeLLM())
    plain = pipeline.answer("q")
    streamed = list(pipeline.answer_stream("q"))[-1][1]
    for field in ["answer", "refused", "sources", "context", "llm_called", "route"]:
        assert getattr(streamed, field) == getattr(plain, field)


def test_stream_uses_the_same_prompt_as_the_plain_answer(chunk):
    llm = FakeLLM()
    pipeline = RAGPipeline(FakeRetriever([chunk]), llm)
    pipeline.answer("q", "be brief")
    list(pipeline.answer_stream("q", "be brief"))
    assert llm.calls[0] == llm.calls[1]


def test_stream_refusal_at_retrieval_calls_no_model():
    llm = FakeLLM()
    events = list(RAGPipeline(FakeRetriever([]), llm).answer_stream("capital of France?"))
    assert kinds(events) == ["start", "delta", "done"]
    assert events[1][1] == REFUSAL and events[2][1].refused and llm.calls == []


def test_stream_refusal_by_the_model_is_known_only_at_the_end(chunk):
    events = list(RAGPipeline(FakeRetriever([chunk]), FakeLLM(reply=REFUSAL)).answer_stream("q"))
    done = events[-1][1]
    assert done.refused is True and done.sources == [] and done.context == [chunk]


def test_stream_answers_from_the_table_without_a_model(chunk):
    table = StructuredRouter(ShipmentStore.from_file(ROOT / "data" / "docs" / "shipment_log.txt"))
    llm = FakeLLM()
    events = list(RAGPipeline(FakeRetriever([chunk]), llm, structured=table).answer_stream(
        "How many shipments were delivered?"))
    assert kinds(events) == ["start", "delta", "done"]
    assert events[0][1]["route"] == "shipment_table" and llm.calls == []
    assert events[1][1].startswith("7 shipments")


class FailsPartway(FakeLLM):
    def generate_stream(self, system, prompt):
        yield "first "
        raise LLMUnavailableError("gone")


def test_stream_failure_partway_raises_and_never_says_done(chunk, caplog):
    stream = RAGPipeline(FakeRetriever([chunk]), FailsPartway()).answer_stream("q", request_id="r")
    seen = []
    with caplog.at_level(logging.ERROR, "shipment_rag.audit"), pytest.raises(LLMUnavailableError):
        for event in stream:
            seen.append(event)
    assert kinds(seen) == ["start", "delta"]
    assert "rag_query_failed" in caplog.text


def test_reader_leaving_partway_is_logged(chunk, caplog):
    stream = RAGPipeline(FakeRetriever([chunk]), FakeLLM()).answer_stream("q", request_id="r")
    with caplog.at_level(logging.INFO, "shipment_rag.audit"):
        next(stream), next(stream)
        stream.close()
    assert "rag_query_abandoned" in caplog.text


def test_completed_stream_reaches_the_audit_log(chunk, caplog):
    with caplog.at_level(logging.INFO, "shipment_rag.audit"):
        list(RAGPipeline(FakeRetriever([chunk]), FakeLLM()).answer_stream("q", request_id="r"))
    assert any(r.message == "rag_query" for r in caplog.records)


# ---------------- SSE endpoint ----------------

def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        assert len(lines) == 2 and lines[0].startswith("event: ") and lines[1].startswith("data: ")
        events.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return events


def client_for(pipeline, **settings):
    app = create_app(Settings(_env_file=None, log_json=False, **settings), pipeline=pipeline)
    return TestClient(app)


def post_stream(client, **body):
    return client.post("/v1/query/stream", json={"question": "q", **body})


def test_sse_format_and_headers(chunk):
    with client_for(RAGPipeline(FakeRetriever([chunk]), FakeLLM())) as c:
        r = post_stream(c)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = parse_sse(r.text)
    assert events[0][0] == "start" and events[-1][0] == "done"
    assert events[0][1]["route"] == "documents"
    assert events[0][1]["request_id"] == r.headers["x-request-id"]
    assert "".join(d["text"] for k, d in events if k == "delta") == events[-1][1]["answer"]


def test_sse_done_matches_the_plain_endpoint(chunk):
    with client_for(RAGPipeline(FakeRetriever([chunk]), FakeLLM())) as c:
        plain = c.post("/v1/query", json={"question": "q"}).json()
        done = parse_sse(post_stream(c).text)[-1][1]
    for ignored in ("latency_ms", "request_id"):
        plain.pop(ignored), done.pop(ignored)
    assert done == plain


def test_sse_a_newline_in_the_answer_cannot_break_the_framing(chunk):
    llm = FakeLLM(reply="line one\n\nevent: done\ndata: {}")
    with client_for(RAGPipeline(FakeRetriever([chunk]), llm)) as c:
        events = parse_sse(post_stream(c).text)
    assert [k for k, _ in events].count("done") == 1
    assert events[-1][1]["answer"] == llm.reply


def test_sse_failure_is_an_error_event_with_no_internal_detail(chunk):
    llm = FakeLLM(exc=LLMUnavailableError("quota key=secret-looking"))
    with client_for(RAGPipeline(FakeRetriever([chunk]), llm)) as c:
        r = post_stream(c)
    assert r.status_code == 200 and "secret-looking" not in r.text
    events = parse_sse(r.text)
    assert [k for k, _ in events] == ["start", "error"]
    assert events[-1][1]["detail"] == "The assistant is temporarily unavailable."


def test_sse_failure_partway_never_sends_done(chunk):
    with client_for(RAGPipeline(FakeRetriever([chunk]), FailsPartway())) as c:
        assert [k for k, _ in parse_sse(post_stream(c).text)] == ["start", "delta", "error"]


def test_sse_unexpected_bug_is_an_error_event(chunk):
    llm = FakeLLM(exc=ValueError("bug with private detail"))
    with client_for(RAGPipeline(FakeRetriever([chunk]), llm)) as c:
        r = post_stream(c)
    assert "private detail" not in r.text and parse_sse(r.text)[-1][0] == "error"


def test_sse_needs_an_api_key_when_configured(chunk):
    with client_for(RAGPipeline(FakeRetriever([chunk]), FakeLLM()), api_keys="k1") as c:
        assert post_stream(c).status_code == 401
        assert c.post("/v1/query/stream", json={"question": "q"},
                      headers={"x-api-key": "k1"}).status_code == 200


def test_sse_validates_its_input(chunk):
    with client_for(RAGPipeline(FakeRetriever([chunk]), FakeLLM())) as c:
        assert c.post("/v1/query/stream", json={"question": ""}).status_code == 422
        assert c.post("/v1/query/stream", json={"question": "x" * 1001}).status_code == 422
