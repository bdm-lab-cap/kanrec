"""Tests for kanrec.config — secret resolution and .env parsing."""
import pytest

from kanrec import config
from kanrec.config import MissingSecretError, get_secret, load_dotenv


# ── get_secret ───────────────────────────────────────────────────────────────

def test_get_secret_reads_environment(monkeypatch):
    monkeypatch.setenv("KANREC_TEST_SECRET", "s3cr3t")
    assert get_secret("KANREC_TEST_SECRET") == "s3cr3t"


def test_get_secret_raises_when_required_and_missing(monkeypatch):
    monkeypatch.delenv("KANREC_ABSENT", raising=False)
    monkeypatch.setattr(config, "_find_dotenv", lambda start=None: None)
    with pytest.raises(MissingSecretError) as exc:
        get_secret("KANREC_ABSENT")
    # The error must tell the user how to fix it, not just that it failed.
    assert "KANREC_ABSENT" in str(exc.value)
    assert ".env" in str(exc.value)


def test_get_secret_returns_default_when_optional(monkeypatch):
    monkeypatch.delenv("KANREC_ABSENT", raising=False)
    monkeypatch.setattr(config, "_find_dotenv", lambda start=None: None)
    assert get_secret("KANREC_ABSENT", default="fallback", required=False) == "fallback"


def test_key_vault_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("KANREC_TEST_SECRET", "from-env")
    monkeypatch.setattr(config, "_from_key_vault", lambda name: "from-vault")
    assert get_secret("KANREC_TEST_SECRET") == "from-vault"


def test_key_vault_absent_falls_through(monkeypatch):
    """Outside Fabric there is no notebookutils; resolution must not crash."""
    monkeypatch.setenv("KANREC_TEST_SECRET", "from-env")
    assert config._from_key_vault("KANREC_TEST_SECRET") is None
    assert get_secret("KANREC_TEST_SECRET") == "from-env"


# ── load_dotenv ──────────────────────────────────────────────────────────────

def test_load_dotenv_parses_and_ignores_comments(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comentario\n"
        "\n"
        'ATLAS_URI="mongodb://example"\n'
        "CONFLUENT_API_KEY=abc123\n"
        "sin_igual\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ATLAS_URI", raising=False)
    monkeypatch.delenv("CONFLUENT_API_KEY", raising=False)

    parsed = load_dotenv(dotenv)

    assert parsed == {"ATLAS_URI": "mongodb://example", "CONFLUENT_API_KEY": "abc123"}


def test_load_dotenv_does_not_override_real_environment(tmp_path, monkeypatch):
    """CI and Fabric secrets must always win over a local .env file."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("ATLAS_URI=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ATLAS_URI", "from-environment")

    load_dotenv(dotenv)

    import os
    assert os.environ["ATLAS_URI"] == "from-environment"


def test_load_dotenv_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "does-not-exist") == {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def test_confluent_config_shape(monkeypatch):
    monkeypatch.setenv("CONFLUENT_BOOTSTRAP", "broker:9092")
    monkeypatch.setenv("CONFLUENT_API_KEY", "key")
    monkeypatch.setenv("CONFLUENT_API_SECRET", "secret")

    cfg = config.confluent_config()

    assert cfg["bootstrap.servers"] == "broker:9092"
    assert cfg["security.protocol"] == "SASL_SSL"
    assert cfg["sasl.username"] == "key"
    assert cfg["sasl.password"] == "secret"
