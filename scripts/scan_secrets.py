"""Dependency-free secret scanner.

    python scripts/scan_secrets.py --staged   # pre-commit: lines being added
    python scripts/scan_secrets.py --all      # CI: every tracked text file
    python scripts/scan_secrets.py FILE...    # specific files

Two layers:
1. Known key formats (Google AIza… and AQ.…, GitHub, OpenAI/Anthropic-style sk-, AWS,
   Slack, private-key blocks).
2. Generic high-entropy values — either assigned to a secret-looking name
   (KEY/TOKEN/SECRET/PASSWORD…) or long mixed-case/digit tokens standing alone.

Findings print the location and a masked value only. Known false positives go in
`.secret-scan-allow` (one literal per line). Exit code 1 if anything is found.
"""

import argparse
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW_FILE = ROOT / ".secret-scan-allow"

KNOWN_PATTERNS = {
    "google-api-key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "google-api-key-v2": re.compile(r"\bAQ\.[0-9A-Za-z_\-]{30,}"),
    "github-token": re.compile(
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{40,}"
    ),
    "sk-style-key": re.compile(r"\bsk-(?:ant-|proj-)?[0-9A-Za-z_\-]{20,}"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "slack-token": re.compile(r"\bxox[abprs]-[0-9A-Za-z\-]{10,}"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}

# NAME=value, NAME: value, "name": "value" where NAME looks secret-ish
ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<name>[A-Z0-9_\-]*(?:key|token|secret|passw(?:or)?d|pwd|credential|auth)[A-Z0-9_\-]*)
    ["']?\s*[:=]\s*["']?
    (?P<value>[^\s"'`,;]{16,})
    """
)
STANDALONE = re.compile(r"(?<![0-9A-Za-z_\-])[0-9A-Za-z_\-]{32,}(?![0-9A-Za-z_\-])")

PLACEHOLDER = re.compile(
    r"(?i)^(?:x+|\*+|<.*>|\$\{.*\}|your[-_].*|changeme|example.*|dummy.*|test.*)$"
)
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".lock", ".whl", ".bin"}


def entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def mixed(s: str) -> bool:
    return all(any(test(c) for c in s) for test in (str.islower, str.isupper, str.isdigit))


def load_allowlist() -> set[str]:
    if not ALLOW_FILE.exists():
        return set()
    lines = ALLOW_FILE.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def mask(s: str) -> str:
    return f"{s[:4]}...({len(s)} chars)"


def scan_line(line: str, allow: set[str]) -> list[tuple[str, str]]:
    if "secret-scan: allow" in line:
        return []
    found = []
    for kind, pat in KNOWN_PATTERNS.items():
        found += [(kind, m.group(0)) for m in pat.finditer(line)]
    for m in ASSIGNMENT.finditer(line):
        v = m.group("value")
        if not PLACEHOLDER.match(v) and len(v) >= 20 and entropy(v) >= 3.5 and mixed(v):
            found.append(("high-entropy-assignment", v))
    for m in STANDALONE.finditer(line):
        v = m.group(0)
        if mixed(v) and entropy(v) >= 4.3:
            found.append(("high-entropy-token", v))
    # de-duplicate overlapping hits on the same value, drop allowlisted values
    seen, out = set(), []
    for kind, v in found:
        if v in allow or v in seen:
            continue
        seen.add(v)
        out.append((kind, v))
    return out


def iter_staged():
    diff = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--no-color"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    path, lineno = None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[6:] if raw.startswith("+++ b/") else None
        elif raw.startswith("@@"):
            lineno = int(re.search(r"\+(\d+)", raw).group(1)) - 1
        elif raw.startswith("+") and path:
            lineno += 1
            if Path(path).suffix.lower() not in SKIP_SUFFIXES:
                yield path, lineno, raw[1:]


def iter_files(paths):
    for p in paths:
        p = Path(p)
        if p.suffix.lower() in SKIP_SUFFIXES or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        shown = p.resolve().relative_to(ROOT) if p.resolve().is_relative_to(ROOT) else p
        for i, line in enumerate(text.splitlines(), 1):
            yield shown.as_posix(), i, line


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / f for f in out.splitlines()]


def scan(lines, allow) -> list[str]:
    return [
        f"{path}:{lineno}: {kind}: {mask(value)}"
        for path, lineno, line in lines
        for kind, value in scan_line(line, allow)
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--staged", action="store_true")
    g.add_argument("--all", action="store_true")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args(argv)

    allow = load_allowlist()
    if args.staged:
        lines = iter_staged()
    elif args.all:
        lines = iter_files(tracked_files())
    else:
        lines = iter_files(args.files)

    findings = scan(lines, allow)
    for f in findings:
        print(f, file=sys.stderr)
    if findings:
        print(
            f"\n{len(findings)} possible secret(s). Move real values to .env (gitignored). "
            "For a false positive, add the literal to .secret-scan-allow or put "
            "'secret-scan: allow' on the line.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
