"""The Colab runner and the blind label file. Nothing here touches Colab, a GPU or a model."""

import importlib.util
import json
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from shipment_rag.config import Settings
from shipment_rag.evaluation.citation_check import cited_claims
from shipment_rag.models import RAGAnswer, RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = json.loads((ROOT / "notebooks" / "colab_eval_runner.ipynb").read_text(encoding="utf-8"))
CODE = ["".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "code"]


def cell(step: int) -> str:
    return next(c for c in CODE if c.startswith(f"# Step {step}:") or
                c.startswith(f"# Step {step} (optional):"))


# ---------------- the notebook as a file ----------------

def test_notebook_asks_for_the_t4_and_stores_no_outputs():
    assert NOTEBOOK["metadata"]["accelerator"] == "GPU"
    assert NOTEBOOK["metadata"]["colab"]["gpuType"] == "T4"
    for c in NOTEBOOK["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_there_is_one_cell_per_step_in_order_each_with_a_comment():
    assert [int(re.match(r"# Step (\d+)", c).group(1)) for c in CODE] == list(range(12))


def test_every_cell_is_valid_python():
    for code in CODE:
        compile(code, "cell", "exec")


def test_scripts_the_notebook_runs_exist():
    for script in set(re.findall(r"scripts/\w+\.py", "".join(CODE))):
        assert (ROOT / script).is_file(), script


def test_the_key_comes_only_from_colab_secrets_and_is_never_printed():
    text = "\n".join(CODE)
    assert 'userdata.get("GEMINI_API_KEY")' in text
    assert not re.search(r"AIza[0-9A-Za-z_\-]{20}", text)
    for line in text.splitlines():
        if "print(" in line:
            assert "key" not in line.lower(), line


def test_only_the_jobs_that_need_the_key_get_it():
    assert "with_gemini" not in cell(5) + cell(6) + cell(7)
    for step in (8, 9, 10):
        assert "with_gemini=True" in cell(step)


def test_the_golden_step_keeps_verdicts_out_of_the_output():
    step = cell(8)
    assert "show=False" in step and "--report" in step
    assert 'line.split("; judge")[0]' in step  # the one line shown has counts, not scores


def test_the_golden_step_needs_calibration_on_this_machine_first():
    assert "CALIBRATION_MARK.exists()" in cell(8)
    assert "--calibration-only" in cell(7) and "--no-cache" in cell(7)


# ---------------- the notebook's logic, run for real against a temp folder ----------------

@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_RUNNER_ROOT", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ns: dict = {}
    exec(cell(0), ns)
    fake_key = "local-" + "test-value-" * 3
    ns["gemini_key"] = lambda: fake_key
    ns["ollama_api"] = lambda *a, **k: (_ for _ in ()).throw(OSError("no ollama here"))
    ns["ensure_ollama"] = lambda: (_ for _ in ()).throw(RuntimeError("no ollama here"))
    repo = ns["REPO"]
    (repo / "data/eval/.cache/ab").mkdir(parents=True)
    (repo / "data/eval/.cache/ab/abc.json").write_text('{"text": "saved answer"}')
    (repo / "data/eval/reports").mkdir(parents=True)
    (repo / "data/eval/reports/r.md").write_text("report")
    (repo / "data/eval/citation_labels_todo.json").write_text("[]")
    ns["LOGS"].mkdir()
    (ns["LOGS"] / "eval_gen.log").write_text("log line")
    (ns["LOGS"] / "ollama.log").write_text("noise")
    ns["fake_key"] = fake_key
    return ns


def saved_zip(ns):
    return next(ns["ROOT"].glob("colab_results_*.zip"))


def test_results_zip_holds_reports_cache_logs_and_labels(runner):
    exec(cell(11), runner)
    with zipfile.ZipFile(saved_zip(runner)) as z:
        names = set(z.namelist())
        info = json.loads(z.read("run_info.json"))
    assert names == {"data/eval/.cache/ab/abc.json", "data/eval/reports/r.md", "logs/eval_gen.log",
                     "data/eval/citation_labels_todo.json", "run_info.json"}
    assert info["gemini_model"] == runner["GEMINI_MODEL"]


VERDICT_LOG = """| every cited passage supports its sentence | 3/4 (75%) |
| fully correct (judge: 2/2) | 4/4 (100%) |

Answer model `gemini-3.8-flash`: 3 API calls, 4 from cache; judge `qwen2.5:7b`: 3 calls
ServerError: 503 UNAVAILABLE
"""


def run_golden_step(ns, code, report=None):
    def fake_run(cmd, log_name, env=None, show=True):
        assert show is False, "verdicts must not reach the output"
        (ns["LOGS"] / log_name).write_text(VERDICT_LOG, encoding="utf-8")
        return code

    ns["run"] = fake_run
    ns["ensure_ollama"] = lambda: None
    ns["CALIBRATION_MARK"].write_text("passed")
    if report:
        (ns["REPO"] / "data/eval/reports/stage2_generation.md").write_text(report, encoding="utf-8")
    exec(cell(8), ns)


PARTIAL = "# Eval\n\n> **Partial run:** 19 of 26 questions not scored (LLM errors / quota).\n"


def test_golden_step_shows_counts_and_never_a_score(runner, capsys):
    run_golden_step(runner, 0, PARTIAL)
    out = capsys.readouterr().out
    assert "3 API calls, 4 from cache" in out and "503 errors: 1" in out
    assert "Partial run: 19 of 26" in out and "**" not in out and "Stopped early" in out
    for verdict in ["75%", "3/4", "4/4", "fully correct", "supports"]:
        assert verdict not in out


def test_golden_step_says_when_all_are_scored(runner, capsys):
    run_golden_step(runner, 0, "# Eval\n\nComplete.\n")
    assert "All 26 questions scored." in capsys.readouterr().out


def test_golden_step_explains_an_unreachable_gemini(runner, capsys):
    """A stale partial report from the branch must not be shown as this run's result."""
    run_golden_step(runner, 2, PARTIAL)
    out = capsys.readouterr().out
    assert "wasn't reachable" in out and "Partial run" not in out


def test_golden_step_refuses_without_calibration(runner):
    runner["ensure_ollama"] = lambda: None
    with pytest.raises(RuntimeError, match="Step 7"):
        exec(cell(8), runner)


@pytest.mark.parametrize("planted", ["the real one", "shaped"])
def test_a_key_in_any_file_stops_the_zip(runner, planted):
    value = runner["fake_key"] if planted == "the real one" else "AIza" + "x" * 35
    (runner["LOGS"] / "eval_gen.log").write_text(f"oops {value}")
    with pytest.raises(RuntimeError) as err:
        exec(cell(11), runner)
    assert "logs/eval_gen.log" in str(err.value) and value not in str(err.value)
    assert not list(runner["ROOT"].glob("colab_results_*.zip"))


def test_a_results_zip_can_restore_the_cache_on_the_next_run(runner):
    exec(cell(11), runner)
    cached = runner["REPO"] / "data/eval/.cache/ab/abc.json"
    cached.unlink()
    assert runner["restore_cache"](saved_zip(runner)) == 1
    assert cached.read_text() == '{"text": "saved answer"}'


def test_restore_takes_only_cache_files(runner, tmp_path):
    archive = tmp_path / "mixed.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("data/eval/.cache/cd/ok.json", "{}")
        z.writestr("src/shipment_rag/pipeline.py", "replaced")
        z.writestr(".env", "replaced")
    assert runner["restore_cache"](archive) == 1
    assert not (runner["REPO"] / ".env").exists()
    assert not (runner["REPO"] / "src").exists()


def test_restore_refuses_a_path_that_climbs_out(runner, tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("data/eval/.cache/../../../../evil.txt", "x")
    with pytest.raises(RuntimeError, match="unsafe path"):
        runner["restore_cache"](archive)
    assert not (runner["ROOT"] / "evil.txt").exists()


def test_job_env_keeps_the_key_out_unless_asked(runner, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "already-in-the-environment")
    assert "GEMINI_API_KEY" not in runner["job_env"]()
    assert runner["job_env"](with_gemini=True)["GEMINI_API_KEY"] == runner["fake_key"]


def test_job_env_matches_the_laptop_settings_that_affect_results(runner):
    env = runner["job_env"]()
    assert env["JUDGE_PROVIDER"] == "ollama" and env["JUDGE_MODEL"] == "qwen2.5:7b"
    assert env["CITATION_CHECKER"] == "nli" and env["LLM_MAX_RETRIES"] == "1"


# ---------------- the blind label file ----------------

def load_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_citation_labels", ROOT / "scripts" / "prepare_citation_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def golden(gid, kind="answerable"):
    return SimpleNamespace(id=gid, type=kind, question=f"Question {gid}?")


CHUNK = RetrievedChunk(text="Stored at 2-8°C.", source="product_manual.txt", section="S",
                       embed_score=0.5, rerank_score=1.0)


class StubPipeline:
    def __init__(self, script, not_cached):
        self.script, self.not_cached = script, not_cached

    def answer(self, question):
        gid = question.split()[1].rstrip("?")
        if gid in self.script:
            return self.script[gid]
        raise self.not_cached(gid)


def test_label_file_has_each_cited_sentence_and_no_verdicts():
    script = load_script()
    answer = "It is stored at 2-8°C [1]. It lasts 24 months [1]."
    pipeline = StubPipeline(
        {"G01": RAGAnswer(answer, [CHUNK], False, 1, context=[CHUNK], llm_called=True),
         "G02": RAGAnswer("No citation here.", [CHUNK], False, 1, context=[CHUNK], llm_called=True),
         "G03": RAGAnswer("refused", [], True, 1, llm_called=False)},
        script.NotCached)
    cases, missing, uncited = script.prepare(
        pipeline, [golden("G01"), golden("G02"), golden("G03"), golden("G04")])
    assert missing == ["G04"] and uncited == ["G02"] and [c["id"] for c in cases] == ["G01"]
    case = cases[0]
    assert case["expect"]["citation_claims"] == [None, None]
    assert case["passages"] == [{"source": "product_manual.txt", "text": "Stored at 2-8°C."}]
    assert len(cited_claims(case["answer"])) == len(case["claims"]) == 2
    text = json.dumps(cases).lower()
    for verdict_word in ["entail", "supported", "score", "verdict"]:
        assert verdict_word not in text


def test_label_file_uses_the_same_settings_as_eval_gen():
    """With the table on, G08, G09 and G21 would be different answers from the ones scored."""
    settings = load_script().eval_gen_settings(
        Settings(_env_file=None, structured_queries=True, rewrite_provider="ollama"))
    assert settings.structured_queries is False and settings.rewrite_provider == "off"


def test_cache_only_llm_serves_stored_answers_and_refuses_the_rest(tmp_path):
    script = load_script()

    class NeverCalled:
        def generate(self, *a, **k):
            raise AssertionError("must never call the model")

    llm = script.CacheOnlyLLM(NeverCalled(), tmp_path, "answer")
    with pytest.raises(script.NotCached):
        llm.generate("system", "prompt")
    key = llm._key("system", "prompt", None)
    (tmp_path / key[:2]).mkdir()
    (tmp_path / key[:2] / f"{key}.json").write_text('{"text": "stored"}', encoding="utf-8")
    assert llm.generate("system", "prompt") == "stored"
