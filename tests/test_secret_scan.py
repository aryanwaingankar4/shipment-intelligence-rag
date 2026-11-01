import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("scan_secrets", ROOT / "scripts" / "scan_secrets.py")
scanner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scanner)

# Fake key material is assembled at runtime from short pieces, so this file never
# contains a full pattern and the repo-wide scan stays clean.
BODY = "Zq8RN6JK7_RpBM" + "596gfHUCN486rbLp" + "rVpxWIytzw1N2cT"


def kinds(line):
    return {k for k, _ in scanner.scan_line(line, allow=set())}


@pytest.mark.parametrize(
    "line,kind",
    [
        ("GEMINI_API_KEY=" + "AQ." + BODY, "google-api-key-v2"),  # the Stage 0 incident
        ("key = '" + "AI" + "za" + BODY[:35] + "'", "google-api-key"),
        ("https://u:" + "gh" + "p_" + BODY.replace("_", "") + "@github.com", "github-token"),
        ("ANTHROPIC_API_KEY=" + "sk-" + "ant-" + BODY, "sk-style-key"),
        ("aws = " + "AK" + "IA" + "ABCDEFGHIJKLMNOP", "aws-access-key"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private-key"),
    ],
)
def test_known_formats(line, kind):
    assert kind in kinds(line)


def test_generic_high_entropy_assignment():
    # unknown vendor format, only caught by the entropy layer
    assert "high-entropy-assignment" in kinds("PAYMENT_SECRET: " + "q9" + BODY[:24])


def test_generic_high_entropy_standalone_token():
    assert "high-entropy-token" in kinds("config(" + BODY + "x7Y)")


@pytest.mark.parametrize(
    "line",
    [
        "GEMINI_API_KEY=",
        "API_KEYS=your-api-key-here",
        "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2",
        "reranker = cross-encoder/ms-marco-MiniLM-L-6-v2",
        "commit 6372edc0a9f1b3c4d5e6f708192a3b4c5d6e7f80",  # hex SHA: no uppercase
        "def test_model_not_found_is_still_an_llm_unavailable_error():",
        "    SCHEMA, sort_keys=True)).encode()).hexdigest()[:8]",  # code, not a secret
        "auth_token = " + "AQ." + BODY + "  # secret-scan: allow",
    ],
)
def test_no_false_positives(line):
    assert kinds(line) == set()


def test_allowlist_is_exact_match():
    value = "q9" + BODY
    assert scanner.scan_line("x " + value, allow={value}) == []
    assert scanner.scan_line("x " + value + "Z", allow={value}) != []


def test_findings_are_masked():
    findings = scanner.scan([("f.env", 1, "K=" + "AQ." + BODY)], allow=set())
    assert findings and BODY not in findings[0]


def test_repository_is_clean():
    assert scanner.scan(scanner.iter_files(scanner.tracked_files()), scanner.load_allowlist()) == []
