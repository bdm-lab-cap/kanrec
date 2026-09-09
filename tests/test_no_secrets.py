"""
Regression test: no credential may ever live in the source tree.

This complements the gitleaks step in CI. gitleaks scans git history;
this test fails fast, locally, the moment somebody pastes a connection
string into a notebook - which is exactly how it happened once.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SCANNED_SUFFIXES = {".py", ".ipynb", ".sh", ".yml", ".yaml", ".md", ".cfg", ".toml"}

EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "node_modules", "build", "dist", "mlruns", "checkpoints",
}

# This file necessarily contains the patterns it looks for.
EXCLUDED_FILES = {"test_no_secrets.py", ".env.example"}

# A placeholder is not a secret.
PLACEHOLDER = re.compile(
    r"<[^>\s]+>|\$\{[^}]+\}|YOUR[-_]|xxx+|\.\.\.|example|placeholder|"
    r"getSecret|environ|getenv|api[_-]?key\b\W*$",
    re.IGNORECASE,
)

PATTERNS: dict[str, re.Pattern] = {
    # mongodb://user:password@host - localhost URIs without credentials are fine
    "MongoDB connection string with credentials":
        re.compile(r"mongodb(\+srv)?://[^\s:/@]+:[^\s@]+@"),
    # API_SECRET = "long-opaque-string"
    "Hardcoded credential assignment":
        re.compile(
            r"(?i)\b(api[_-]?secret|api[_-]?key|password|passwd|token|"
            r"secret[_-]?key|access[_-]?key)\b\s*[:=]\s*[\"'][^\"']{12,}[\"']"
        ),
    # sasl.password: "..."
    "SASL credential literal":
        re.compile(r"(?i)sasl\.(username|password)\"?\s*:\s*[\"'][^\"']{12,}[\"']"),
    # Private key blocks
    "Private key block":
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def _is_excluded(path: Path) -> bool:
    """Skips vendored, generated and hidden directories (except .github)."""
    for part in path.relative_to(REPO_ROOT).parts[:-1]:
        if part in EXCLUDED_PARTS:
            return True
        # .git, .venv, .pytest_cache, .backup_*, .ipynb_checkpoints ...
        if part.startswith(".") and part != ".github":
            return True
    return path.name in EXCLUDED_FILES


def _scanned_files() -> list[Path]:
    return [
        path for path in REPO_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in SCANNED_SUFFIXES
        and not _is_excluded(path)
    ]


def test_scanner_actually_scans_something():
    """Guards against the scan silently covering zero files."""
    files = _scanned_files()
    assert len(files) > 20, f"Only {len(files)} files scanned - check the filters"


@pytest.mark.parametrize("label,pattern", sorted(PATTERNS.items()))
def test_no_hardcoded_credentials(label, pattern):
    findings = []
    for path in _scanned_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = pattern.search(line)
            if match and not PLACEHOLDER.search(match.group(0)):
                rel = path.relative_to(REPO_ROOT)
                findings.append(f"  {rel}:{lineno}  {line.strip()[:90]}")

    assert not findings, (
        f"{label} found in the source tree:\n" + "\n".join(findings) +
        "\n\nUse kanrec.config.get_secret() and put the value in .env "
        "(local) or Key Vault (Fabric). If this credential was ever "
        "committed, rotate it before removing it."
    )


def test_env_example_exists_and_has_no_real_values():
    """The template must document every secret without carrying any."""
    example = REPO_ROOT / ".env.example"
    assert example.is_file(), ".env.example is required to document the secrets"

    text = example.read_text(encoding="utf-8")
    for key in ["ATLAS_URI", "CONFLUENT_API_KEY", "CONFLUENT_API_SECRET"]:
        assert key in text, f"{key} is not documented in .env.example"

    for line in text.splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        _, _, value = line.partition("=")
        value = value.strip()
        if not value or PLACEHOLDER.search(value):
            continue
        # Non-secret defaults are allowed; opaque high-entropy values are not.
        assert len(value) < 40 or "." in value, (
            f".env.example seems to contain a real value: {line.strip()[:60]}"
        )


def test_gitignore_excludes_env_file():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    entries = {line.strip() for line in gitignore}
    assert ".env" in entries, ".env must be listed in .gitignore"
    assert "!.env.example" in entries, ".env.example must stay tracked"
