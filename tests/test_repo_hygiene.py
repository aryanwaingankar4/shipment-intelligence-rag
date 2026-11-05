import re
from pathlib import Path

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
SECRET_VARS = {"GEMINI_API_KEY", "API_KEYS", "API_KEY"}


def test_env_example_has_no_secret_values():
    # .env.example is committed; real values belong in the gitignored .env
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() in SECRET_VARS:
            assert not value.strip(), f"{name} must be empty in .env.example"


def test_demo_script_runs_without_ollama_and_prints_no_secrets():
    script = (ENV_EXAMPLE.parent / "scripts" / "demo.ps1").read_text(encoding="utf-8")
    assert script.isascii()  # Windows PowerShell 5.1 misreads other characters without a BOM
    assert '$answerModel = "gemini-3.5-flash-lite"' in script
    assert '$env:REWRITE_PROVIDER = "off"' in script
    assert not re.search(r"ollama(\.exe)?\s+(serve|run|pull)", script, re.I)
    for line in script.splitlines():
        if "Write-Host" in line:
            assert not re.search(r"\$env:(GEMINI_API_KEY|API_KEYS?|UI_PASSWORD)", line), line
