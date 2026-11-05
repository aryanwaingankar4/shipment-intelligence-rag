"""Ollama judge client against a fake server (httpx.MockTransport, no network or Ollama)."""

import json

import httpx
import pytest

from shipment_rag.config import Settings
from shipment_rag.factory import build_judge_llm
from shipment_rag.llm import LLMUnavailableError, ModelNotFoundError, OllamaLLM

MODEL = "qwen2.5:7b"


def client(handler, **kwargs) -> OllamaLLM:
    return OllamaLLM(MODEL, transport=httpx.MockTransport(handler), **kwargs)


def chat_reply(content='{"ok": true}', **extra):
    body = {"message": {"content": content}, "done_reason": "stop",
            "prompt_eval_count": 100, "eval_count": 20, **extra}
    return lambda request: httpx.Response(200, json=body)


def test_generate_sends_schema_and_deterministic_options():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), path=request.url.path)
        return chat_reply()(request)

    llm = client(handler)
    schema = {"type": "object"}
    assert llm.generate("sys", "prompt", json_schema=schema) == '{"ok": true}'
    assert seen["path"] == "/api/chat" and seen["model"] == MODEL and seen["stream"] is False
    assert seen["format"] == schema
    assert seen["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "prompt"}]
    assert seen["options"]["temperature"] == 0.0 and seen["options"]["seed"] == 42
    assert llm.usage["calls"] == 1 and llm.usage["prompt_tokens"] == 100


def test_no_schema_means_no_format_field():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return chat_reply("plain")(request)

    assert client(handler).generate("s", "p") == "plain"
    assert "format" not in seen


@pytest.mark.parametrize("reply, match", [
    (chat_reply(done_reason="length"), "output limit"),
    (chat_reply(prompt_eval_count=3500), "may not fit"),
    (chat_reply("   "), "empty"),
    (lambda request: httpx.Response(500, text="boom"), "HTTP 500"),
])
def test_damaged_replies_raise_so_they_are_never_cached(reply, match):
    with pytest.raises(LLMUnavailableError, match=match):
        client(reply).generate("s", "p")


def test_model_not_pulled_says_how_to_fix_it():
    with pytest.raises(ModelNotFoundError, match=f"ollama pull {MODEL}"):
        client(lambda request: httpx.Response(404, json={"error": "not found"})).generate("s", "p")


def test_server_down_is_a_clear_error():
    def refuse(request):
        raise httpx.ConnectError("refused")

    llm = client(refuse)
    with pytest.raises(LLMUnavailableError, match="not reachable"):
        llm.generate("s", "p")
    with pytest.raises(LLMUnavailableError, match="not reachable"):
        llm.check_model()


def test_check_model_records_digest_in_the_cache_identity():
    tags = {"models": [{"name": MODEL, "digest": "abcdef0123456789"}]}
    llm = client(lambda request: httpx.Response(200, json=tags))
    before = llm.cache_identity()
    llm.check_model()
    assert llm.digest == "abcdef012345" and llm.cache_identity() != before


def test_check_model_fails_when_only_another_tag_is_pulled():
    tags = {"models": [{"name": "qwen2.5:3b-instruct", "digest": "x"}]}
    with pytest.raises(ModelNotFoundError):
        client(lambda request: httpx.Response(200, json=tags)).check_model()


def test_factory_ollama_judge_needs_no_gemini_key():
    settings = Settings(_env_file=None, gemini_api_key=None, judge_provider="ollama",
                        judge_model=MODEL)
    judge = build_judge_llm(settings)
    assert isinstance(judge, OllamaLLM) and judge.model == MODEL and judge.temperature == 0.0


def test_factory_ollama_judge_requires_a_model():
    settings = Settings(_env_file=None, gemini_api_key=None, judge_provider="ollama",
                        judge_model="")
    with pytest.raises(RuntimeError, match="JUDGE_MODEL"):
        build_judge_llm(settings)
