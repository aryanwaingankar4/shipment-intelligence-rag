from pathlib import Path

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
SECRET_VARS = {"GEMINI_API_KEY", "API_KEYS", "API_KEY"}


def test_env_example_has_no_secret_values():
    # .env.example is committed; real values belong in the gitignored .env
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() in SECRET_VARS:
            assert not value.strip(), f"{name} must be empty in .env.example"
