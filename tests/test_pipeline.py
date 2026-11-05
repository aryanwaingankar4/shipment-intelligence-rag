import pytest

from shipment_rag.llm import LLMUnavailableError
from shipment_rag.pipeline import REFUSAL, build_prompt
from tests.conftest import FakeLLM


def test_answer_is_grounded_and_cites_sources(make_pipeline, chunk):
    llm = FakeLLM()
    result = make_pipeline([chunk], llm).answer("Storage temp for TL-COLD-9001?")
    assert not result.refused
    assert "2°C to 8°C" in result.answer
    assert [c.source for c in result.sources] == ["product_manual.txt"]
    system, prompt = llm.calls[0]
    assert "ONLY from the numbered context" in system
    assert "[1] (source: product_manual.txt" in prompt


def test_no_relevant_chunks_refuses_without_calling_llm(make_pipeline):
    llm = FakeLLM()
    result = make_pipeline([], llm).answer("What is the capital of France?")
    assert result.refused and result.answer == REFUSAL
    assert llm.calls == [] and not result.llm_called


def test_context_keeps_prompt_passages_even_when_model_refuses(make_pipeline, chunk):
    result = make_pipeline([chunk], FakeLLM(reply=REFUSAL)).answer("q")
    assert result.llm_called and result.refused
    assert result.sources == [] and result.context == [chunk]


def test_model_refusal_drops_sources(make_pipeline, chunk):
    result = make_pipeline([chunk], FakeLLM(reply=REFUSAL)).answer("Unrelated?")
    assert result.refused
    assert result.sources == []


def test_llm_failure_propagates(make_pipeline, chunk):
    with pytest.raises(LLMUnavailableError):
        make_pipeline([chunk], FakeLLM(exc=LLMUnavailableError("429"))).answer("q")


def test_custom_instructions_are_scoped_as_preferences(chunk):
    prompt = build_prompt("q", [chunk], "answer in bullet points")
    assert prompt.rstrip().endswith("User style preferences: answer in bullet points")
