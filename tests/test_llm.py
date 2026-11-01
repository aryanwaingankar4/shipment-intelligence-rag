"""GeminiLLM behaviour with a stubbed client (no network)."""

import logging

import pytest
from google.genai import errors, types

from shipment_rag.llm import GeminiLLM, LLMUnavailableError, ModelNotFoundError


def _not_found():
    return errors.ClientError(404, {"error": {"code": 404, "message": "not found"}})


def _response(text="ok", finish="STOP", thoughts=0):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text=text)], role="model"),
                finish_reason=finish,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            thoughts_token_count=thoughts, candidates_token_count=5
        ),
    )


class FakeModels:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.configs = response, exc, []

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        if self.exc:
            raise self.exc
        return self.response

    def get(self, model):
        if self.exc:
            raise self.exc
        return object()


def make_llm(models, **kw):
    llm = GeminiLLM(api_key="test", model="gemini-test", max_retries=1, **kw)
    llm._client = type("C", (), {"models": models})()
    return llm


def test_thinking_level_and_token_budget_are_sent():
    models = FakeModels(_response())
    make_llm(models, thinking_level="low", max_output_tokens=2048).generate("sys", "q")
    cfg = models.configs[0]
    assert cfg.max_output_tokens == 2048
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert cfg.system_instruction == "sys"


def test_thinking_budget_overrides_level_for_25_models():
    models = FakeModels(_response())
    make_llm(models, thinking_level="low", thinking_budget=256).generate("s", "q")
    tc = models.configs[0].thinking_config
    assert tc.thinking_budget == 256 and tc.thinking_level is None


def test_truncation_is_logged(caplog):
    models = FakeModels(_response("partial answ", finish="MAX_TOKENS", thoughts=1900))
    with caplog.at_level(logging.WARNING, logger="shipment_rag.llm"):
        text = make_llm(models).generate("s", "q")
    assert text == "partial answ"
    rec = next(r for r in caplog.records if "truncated" in r.getMessage())
    assert rec.thoughts_tokens == 1900


def test_no_warning_on_normal_finish(caplog):
    with caplog.at_level(logging.WARNING, logger="shipment_rag.llm"):
        make_llm(FakeModels(_response())).generate("s", "q")
    assert not any("truncated" in r.getMessage() for r in caplog.records)


def test_model_404_raises_clear_error_on_check_and_generate():
    llm = make_llm(FakeModels(exc=_not_found()))
    with pytest.raises(ModelNotFoundError, match="gemini-test.*not found.*GEMINI_MODEL"):
        llm.check_model()
    with pytest.raises(ModelNotFoundError):
        llm.generate("s", "q")


def test_model_not_found_is_still_an_llm_unavailable_error():
    # so the API keeps mapping it to a clean 503
    assert issubclass(ModelNotFoundError, LLMUnavailableError)


def test_empty_reply_is_unavailable():
    with pytest.raises(LLMUnavailableError):
        make_llm(FakeModels(_response(text=""))).generate("s", "q")
