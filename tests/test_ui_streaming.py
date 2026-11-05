"""The chat page against a stubbed API: nothing listens on a port and no browser is involved."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
requests = pytest.importorskip("requests")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")


def sse(*events):
    lines = []
    for name, data in events:
        lines += [f"event: {name}", f"data: {json.dumps(data, ensure_ascii=False)}", ""]
    return lines


class FakeResponse:
    def __init__(self, lines, status_code=200, then_raise=None):
        self.lines, self.status_code, self.then_raise = lines, status_code, then_raise
        self.encoding = None

    def iter_lines(self, decode_unicode=False):
        yield from self.lines
        if self.then_raise:
            raise self.then_raise


DONE = {"answer": "Stored at 2-8°C [1].", "refused": False, "route": "documents",
        "sources": [{"source": "product_manual.txt"}],
        "rewritten_question": "What is the storage condition for TL-COLD-9001?"}


def run_chat(monkeypatch, response, question="And its storage?", messages=()):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(requests, "post", fake_post)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["authenticated"] = True
    at.session_state["page"] = "chat"
    at.session_state["messages"] = list(messages)
    at.run()
    at.chat_input[0].set_value(question).run()
    return at, calls


def page_text(at):
    return "\n".join(m.value for m in at.markdown)


def test_streamed_answer_is_shown_with_its_labels(monkeypatch):
    response = FakeResponse(sse(
        ("start", {"route": "documents", "rewritten_question": DONE["rewritten_question"]}),
        ("delta", {"text": "Stored at "}), ("delta", {"text": "2-8°C [1]."}), ("done", DONE)))
    at, calls = run_chat(monkeypatch, response)
    text = page_text(at)
    assert not at.exception
    assert "Stored at 2-8°C [1]." in text
    assert "Understood as: What is the storage condition for TL-COLD-9001?" in text
    assert "Answered from: company documents" in text
    assert "Sources: product_manual.txt" in text
    url, kwargs = calls[0]
    assert url.endswith("/v1/query/stream") and kwargs["stream"] is True


def test_earlier_messages_are_sent_and_the_new_question_is_not_among_them(monkeypatch):
    earlier = [{"role": "user", "content": "What is TL-COLD-9001?"},
               {"role": "assistant", "content": "CryoVial [1].", "sources": [],
                "route": "documents"}]
    response = FakeResponse(sse(("delta", {"text": "x"}), ("done", {**DONE, "answer": "x"})))
    _, calls = run_chat(monkeypatch, response, messages=earlier)
    body = calls[0][1]["json"]
    assert body["question"] == "And its storage?"
    assert body["history"] == [{"role": "user", "content": "What is TL-COLD-9001?"},
                               {"role": "assistant", "content": "CryoVial [1]."}]


def test_history_is_limited_so_the_api_never_rejects_it(monkeypatch):
    earlier = [{"role": "user", "content": "x" * 9000} for _ in range(10)]
    response = FakeResponse(sse(("delta", {"text": "x"}), ("done", {**DONE, "answer": "x"})))
    _, calls = run_chat(monkeypatch, response, messages=earlier)
    history = calls[0][1]["json"]["history"]
    assert len(history) == 6 and all(len(m["content"]) == 2000 for m in history)


def test_model_output_is_escaped_while_streaming_and_after(monkeypatch):
    evil = "<img src=x onerror=alert(1)>"
    response = FakeResponse(sse(("delta", {"text": evil}), ("done", {**DONE, "answer": evil})))
    at, _ = run_chat(monkeypatch, response)
    text = page_text(at)
    assert "<img" not in text and "&lt;img" in text


@pytest.mark.parametrize("response", [
    FakeResponse(sse(("start", {}), ("delta", {"text": "half an ans"}),
                     ("error", {"detail": "x"}))),
    FakeResponse(sse(("start", {}), ("delta", {"text": "half an ans"}))),
    FakeResponse(sse(("delta", {"text": "half an ans"})),
                 then_raise=requests.ConnectionError("dropped")),
    FakeResponse([], status_code=503),
    requests.ConnectionError("refused"),
], ids=["error-event", "ended-without-done", "connection-dropped", "http-503", "no-connection"])
def test_a_failed_answer_shows_the_error_and_no_partial_text(monkeypatch, response):
    at, _ = run_chat(monkeypatch, response)
    text = page_text(at)
    assert not at.exception
    assert "currently" in text and "unavailable" in text
    assert "half an ans" not in text
    assert all(m["role"] == "user" for m in at.session_state["messages"])
