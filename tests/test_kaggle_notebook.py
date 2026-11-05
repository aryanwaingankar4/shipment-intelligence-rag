"""The Kaggle runner: unattended, level with the Colab one. Touches no Kaggle and no model."""

import ast
import json
import re
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from shipment_rag.config import Settings

ROOT = Path(__file__).resolve().parent.parent


def load(name: str) -> tuple[dict, list[str]]:
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return notebook, ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


NOTEBOOK, CODE = load("kaggle_eval_runner.ipynb")
_, COLAB = load("colab_eval_runner.ipynb")
HASH_NAME = "ab" + "c" * 62 + ".json"


def cell(step: int, code: list[str] = CODE) -> str:
    return next(c for c in code if re.match(rf"# Step {step}( \(optional\))?:", c))


# ---------------- the notebook as a file ----------------

def test_notebook_asks_for_the_t4_and_internet_and_stores_no_outputs():
    meta = NOTEBOOK["metadata"]["kaggle"]
    assert meta["accelerator"] == "nvidiaTeslaT4" and meta["isInternetEnabled"] is True
    for c in NOTEBOOK["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_there_is_one_valid_cell_per_step_in_order():
    assert [int(re.match(r"# Step (\d+)", c).group(1)) for c in CODE] == list(range(12))
    for code in CODE:
        compile(code, "cell", "exec")


def test_scripts_the_notebook_runs_exist():
    for script in set(re.findall(r"scripts/\w+\.py", "".join(CODE))):
        assert (ROOT / script).is_file(), script


def test_nothing_waits_for_a_person():
    text = "\n".join(CODE)
    for interactive in ["google.colab", "files.upload", "drive.mount", "getpass", "input("]:
        assert not re.search(rf"(?<![\w.]){re.escape(interactive)}", text), interactive


def test_the_key_comes_only_from_kaggle_secrets_and_is_never_printed():
    text = "\n".join(CODE)
    assert 'UserSecretsClient().get_secret("GEMINI_API_KEY")' in text
    assert not re.search(r"AIza[0-9A-Za-z_\-]{20}", text)
    for line in text.splitlines():
        if "print(" in line:
            assert "key" not in line.lower(), line


def test_only_the_jobs_that_need_the_key_get_it():
    assert "with_gemini" not in cell(5) + cell(6) + cell(7)
    for step in (8, 9, 10):
        assert "with_gemini=True" in cell(step)


# ---------------- level with the Colab runner ----------------

def test_both_notebooks_and_the_code_default_use_the_same_answer_model():
    def setting(code):
        return re.search(r'^GEMINI_MODEL = "([^"]+)"', code, re.M).group(1)

    assert setting(cell(0)) == setting(cell(0, COLAB)) == "gemini-3.5-flash-lite"
    assert Settings(_env_file=None).gemini_model == "gemini-3.5-flash-lite"


def test_both_notebooks_run_the_same_commands():
    def commands(code):
        text = re.sub(r"\s+", " ", code)
        return re.findall(r"run\(\[(?:CLI|PY), .*?\], \"[\w.]+\"", text)

    for step in (5, 6, 7, 8, 9, 10):
        assert commands(cell(step)) == commands(cell(step, COLAB)), step
        assert commands(cell(step)), step


@pytest.mark.parametrize("name", ["job_env", "run", "ollama_api", "ensure_ollama", "restore_cache"])
def test_shared_helpers_are_identical(name):
    def helper(code):
        tree = ast.parse(code)
        found = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.dump(found)

    assert helper(cell(0)) == helper(cell(0, COLAB))


# ---------------- the notebook's logic, run for real against temp folders ----------------

FAKE_KEY = "local-" + "test-value-" * 3


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_RUNNER_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("KAGGLE_RUNNER_KAGGLE", str(tmp_path / "kaggle"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ns: dict = {}
    exec(cell(0), ns)
    ns["gemini_key"] = lambda: FAKE_KEY
    ns["ollama_api"] = lambda *a, **k: (_ for _ in ()).throw(OSError("no ollama here"))
    ns["ensure_ollama"] = lambda: (_ for _ in ()).throw(RuntimeError("no ollama here"))
    repo = ns["REPO"]
    (repo / "data/eval/.cache/ab").mkdir(parents=True)
    (repo / "data/eval/.cache/ab" / HASH_NAME).write_text('{"text": "saved answer"}')
    (repo / "data/eval/reports").mkdir(parents=True)
    (repo / "data/eval/reports/r.md").write_text("report")
    (repo / "data/eval/citation_labels_todo.json").write_text("[]")
    ns["LOGS"].mkdir()
    (ns["LOGS"] / "eval_gen.log").write_text("log line")
    (ns["LOGS"] / "ollama.log").write_text("noise")
    return ns


def saved_zip(ns) -> Path:
    return next(ns["WORKING"].glob("kaggle_results_*.zip"))


def clear_cache(ns) -> Path:
    cached = ns["REPO"] / "data/eval/.cache/ab" / HASH_NAME
    cached.unlink()
    return cached


def test_results_zip_is_the_only_thing_in_working_and_holds_everything(runner):
    exec(cell(11), runner)
    assert [p.name for p in runner["WORKING"].iterdir()] == [saved_zip(runner).name]
    with zipfile.ZipFile(saved_zip(runner)) as z:
        names = set(z.namelist())
        info = json.loads(z.read("run_info.json"))
    assert names == {f"data/eval/.cache/ab/{HASH_NAME}", "data/eval/reports/r.md",
                     "logs/eval_gen.log", "data/eval/citation_labels_todo.json", "run_info.json"}
    assert info["gemini_model"] == "gemini-3.5-flash-lite" and info["machine"] == "Kaggle GPU"
    assert info["failed_steps"] == {}


def test_nothing_the_jobs_write_lands_in_working(runner):
    assert runner["WORKING"] not in runner["REPO"].parents
    assert runner["WORKING"] not in runner["LOGS"].parents


@pytest.mark.parametrize("planted", ["the real one", "shaped"])
def test_a_key_in_any_file_stops_the_zip(runner, planted):
    value = FAKE_KEY if planted == "the real one" else "AIza" + "x" * 35
    (runner["LOGS"] / "eval_gen.log").write_text(f"oops {value}")
    with pytest.raises(RuntimeError) as err:
        exec(cell(11), runner)
    assert "logs/eval_gen.log" in str(err.value) and value not in str(err.value)
    assert not list(runner["WORKING"].glob("*"))


def test_a_key_in_a_failure_message_stops_the_zip(runner):
    runner["FAILED"][6] = f"oops {FAKE_KEY}"
    with pytest.raises(RuntimeError, match="run_info.json"):
        exec(cell(11), runner)


def test_zip_still_saves_when_the_secret_is_missing(runner):
    def missing():
        raise RuntimeError("No GEMINI_API_KEY")

    runner["gemini_key"] = missing
    exec(cell(11), runner)
    assert saved_zip(runner).exists()


def test_a_real_lookup_failure_gives_a_clear_message_with_no_detail(runner, monkeypatch):
    """The secrets client's own error text stays out of the notebook's saved output."""
    import sys

    class Client:
        def get_secret(self, name):
            raise ValueError("secret detail")

    monkeypatch.setitem(sys.modules, "kaggle_secrets", SimpleNamespace(UserSecretsClient=Client))
    ns: dict = {}
    exec(cell(0), ns)
    with pytest.raises(RuntimeError, match="Add-ons > Secrets") as err:
        ns["gemini_key"]()
    assert "secret detail" not in str(err.value)


# ---------------- finding saved answers, however Kaggle lays them out ----------------

def test_restore_from_a_zip_under_any_dataset_name(runner):
    cached = clear_cache(runner)
    folder = runner["INPUT"] / "some-name"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "eval_cache.zip", "w") as z:
        z.writestr(f"data/eval/.cache/ab/{HASH_NAME}", '{"text": "saved answer"}')
    assert runner["restore_from_input"]() == 1
    assert cached.read_text() == '{"text": "saved answer"}'


@pytest.mark.parametrize("layout", [
    "ds/data/eval/.cache/ab/{}",   # the zip unpacked by Kaggle
    "ds/eval_cache/data/eval/.cache/ab/{}",  # unpacked into a folder named after it
    "datasets/owner/ds/.cache/ab/{}",  # a deeper mount path
    "ds/ab/{}",  # the hidden folder dropped
    "ds/{}",  # flattened
])
def test_restore_from_unpacked_files_in_any_layout(runner, layout):
    cached = clear_cache(runner)
    source = runner["INPUT"] / layout.format(HASH_NAME)
    source.parent.mkdir(parents=True)
    source.write_text('{"text": "saved answer"}')
    assert runner["restore_from_input"]() == 1
    assert cached.read_text() == '{"text": "saved answer"}'


def test_restore_ignores_everything_else_in_the_input(runner):
    clear_cache(runner)
    folder = runner["INPUT"] / "ds"
    folder.mkdir(parents=True)
    (folder / "run_info.json").write_text("{}")
    (folder / ".env").write_text("replaced")
    with zipfile.ZipFile(folder / "mixed.zip", "w") as z:
        z.writestr("src/shipment_rag/pipeline.py", "replaced")
        z.writestr(".env", "replaced")
    assert runner["restore_from_input"]() == 0
    assert not (runner["REPO"] / ".env").exists() and not (runner["REPO"] / "src").exists()


def test_restore_with_nothing_attached_is_fine(runner, capsys):
    clear_cache(runner)
    exec(cell(4), runner)
    assert "starting with no saved answers" in capsys.readouterr().out
    assert runner["FAILED"] == {}


def test_a_climbing_zip_is_refused_and_blocks_scoring(runner):
    folder = runner["INPUT"] / "ds"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "evil.zip", "w") as z:
        z.writestr("data/eval/.cache/../../../../evil.txt", "x")
    exec(cell(4), runner)
    assert "unsafe path" in runner["FAILED"][4]
    runner["run"] = lambda *a, **k: pytest.fail("must not spend quota after a failed restore")
    exec(cell(8), runner)
    assert "Step 4" in runner["FAILED"][8]


# ---------------- Save & Run All: a failure never stops the zip ----------------

def failing_subprocess():
    def fake_run(cmd, check=False, **kwargs):
        if check:
            raise subprocess.CalledProcessError(1, cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    return SimpleNamespace(run=fake_run, CalledProcessError=subprocess.CalledProcessError,
                           DEVNULL=subprocess.DEVNULL)


def test_run_all_reaches_the_zip_when_everything_fails(runner, capsys):
    runner["subprocess"] = failing_subprocess()
    runner["run"] = lambda *a, **k: pytest.fail("no job may start without an install")
    for code in CODE[1:]:
        exec(code, runner)
    assert saved_zip(runner).exists()
    assert set(runner["FAILED"]) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    assert "skipped, because Step 1" in runner["FAILED"][3]
    with zipfile.ZipFile(saved_zip(runner)) as z:
        saved = json.loads(z.read("run_info.json"))["failed_steps"]
    assert saved["2"].startswith("CalledProcessError")
    assert "Steps that did not finish" in capsys.readouterr().out


def test_a_clean_run_reports_every_step_finished(runner, capsys):
    exec(cell(11), runner)
    assert "All steps finished." in capsys.readouterr().out


def test_a_rerun_that_works_clears_the_failure(runner):
    runner["FAILED"][4] = "earlier failure"
    exec(cell(4), runner)
    assert 4 not in runner["FAILED"]


def test_a_failed_retrieval_gate_stops_everything_that_scores(runner):
    runner["run"] = lambda cmd, log, **k: 1 if log == "retrieval_eval.log" else 0
    exec(cell(5), runner)
    assert "BELOW THE GATE" in runner["FAILED"][5]
    for step in (6, 7, 8, 9):
        exec(cell(step), runner)
        assert "Step 5" in runner["FAILED"][step] or "Step 7" in runner["FAILED"][step]


def test_a_failed_calibration_skips_scoring_but_not_the_label_file(runner):
    runner["ensure_ollama"] = lambda: None
    ran = []

    def fake_run(cmd, log_name, env=None, show=True):
        ran.append(log_name)
        (runner["LOGS"] / log_name).write_text("")
        return 1 if log_name == "calibration.log" else 0

    runner["run"] = fake_run
    exec(cell(7), runner)
    exec(cell(8), runner)
    exec(cell(9), runner)
    assert ran == ["calibration.log", "prepare_labels.log"]
    assert not runner["CALIBRATION_MARK"].exists()
    assert "Step 7" in runner["FAILED"][8] and 9 not in runner["FAILED"]


@pytest.mark.parametrize("gpu_name", ["Tesla P100-PCIE-16GB, 16280 MiB"])
def test_the_p100_is_refused(runner, gpu_name):
    runner["shutil"] = SimpleNamespace(which=lambda name: "/usr/bin/nvidia-smi")
    runner["subprocess"] = SimpleNamespace(
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout=gpu_name, stderr=""))
    exec(cell(1), runner)
    assert "T4 x2" in runner["FAILED"][1]


def test_the_t4_is_accepted(runner):
    runner["shutil"] = SimpleNamespace(which=lambda name: "/usr/bin/nvidia-smi")
    runner["subprocess"] = SimpleNamespace(
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="Tesla T4, 15360 MiB", stderr=""))
    exec(cell(1), runner)
    assert 1 not in runner["FAILED"]


# ---------------- Step 8 keeps the check blind ----------------

VERDICT_LOG = """| every cited passage supports its sentence | 3/4 (75%) |
| fully correct (judge: 2/2) | 4/4 (100%) |

Answer model `gemini-3.5-flash-lite`: 3 API calls, 4 from cache; judge `qwen2.5:7b`: 3 calls
ServerError: 503 UNAVAILABLE
"""
PARTIAL = "# Eval\n\n> **Partial run:** 19 of 26 questions not scored (LLM errors / quota).\n"


def run_golden_step(ns, code, report=None):
    def fake_run(cmd, log_name, env=None, show=True):
        assert show is False, "verdicts must not reach the output"
        assert env["GEMINI_API_KEY"] == FAKE_KEY and env["GEMINI_MODEL"] == "gemini-3.5-flash-lite"
        (ns["LOGS"] / log_name).write_text(VERDICT_LOG, encoding="utf-8")
        return code

    ns["run"] = fake_run
    ns["ensure_ollama"] = lambda: None
    ns["CALIBRATION_MARK"].write_text("passed")
    if report:
        (ns["REPO"] / "data/eval/reports/stage2_generation.md").write_text(report, encoding="utf-8")
    exec(cell(8), ns)


def test_golden_step_shows_the_model_and_counts_and_never_a_score(runner, capsys):
    run_golden_step(runner, 0, PARTIAL)
    out = capsys.readouterr().out
    assert "gemini-3.5-flash-lite" in out and "3 API calls, 4 from cache" in out
    assert "503 errors: 1" in out and "Partial run: 19 of 26" in out and "Stopped early" in out
    for verdict in ["75%", "3/4", "4/4", "fully correct", "supports"]:
        assert verdict not in out
    assert 8 not in runner["FAILED"]


def test_golden_step_says_when_all_are_scored(runner, capsys):
    run_golden_step(runner, 0, "# Eval\n\nComplete.\n")
    assert "All 26 questions scored." in capsys.readouterr().out


def test_an_unreachable_gemini_is_recorded_without_the_stale_report(runner, capsys):
    run_golden_step(runner, 2, PARTIAL)
    out = capsys.readouterr().out
    assert "wasn't reachable" in out and "Partial run" not in out
    assert "wasn't reachable" in runner["FAILED"][8]


def test_a_crash_with_any_other_code_is_not_called_a_full_score(runner, capsys):
    run_golden_step(runner, 137)
    out = capsys.readouterr().out
    assert "All 26" not in out and "exit 137" in runner["FAILED"][8]
