"""
Centralised configuration and secret loading for KAN-REC.

No credential is ever hardcoded in the source tree. Secrets are resolved
in this order, stopping at the first hit:

  1. Azure Key Vault, when running inside a Microsoft Fabric notebook
     (``notebookutils.credentials.getSecret``).
  2. Process environment variables.
  3. A ``.env`` file at the repository root (never committed; see
     ``.env.example`` for the expected keys).

A missing required secret raises :class:`MissingSecretError` with an
actionable message instead of failing later with an opaque auth error.

Usage
-----
    from kanrec.config import get_secret, atlas_uri

    uri = atlas_uri()                       # raises if not configured
    key = get_secret("CONFLUENT_API_KEY")   # raises if not configured
    dbg = get_secret("KANREC_DEBUG", default="0", required=False)
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "MissingSecretError",
    "get_secret",
    "load_dotenv",
    "atlas_uri",
    "confluent_config",
    "KEY_VAULT_NAME",
]

#: Name of the Azure Key Vault used from Microsoft Fabric notebooks.
#: Override with the ``KANREC_KEY_VAULT`` environment variable.
KEY_VAULT_NAME = os.environ.get("KANREC_KEY_VAULT", "kanrec-kv")

_DOTENV_FILENAME = ".env"


class MissingSecretError(RuntimeError):
    """Raised when a required secret is not available in any source."""


# ── .env support (dependency-free) ───────────────────────────────────────────

def _find_dotenv(start: Path | None = None) -> Path | None:
    """Walks up from ``start`` looking for a ``.env`` file."""
    current = (start or Path(__file__).resolve().parent)
    for candidate in [current, *current.parents]:
        dotenv = candidate / _DOTENV_FILENAME
        if dotenv.is_file():
            return dotenv
    return None


def load_dotenv(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """
    Loads ``KEY=value`` pairs from a ``.env`` file into ``os.environ``.

    Existing environment variables win unless ``override=True``, so the
    real environment (CI secrets, Fabric, Docker) always takes precedence
    over a local development file.

    Returns the parsed mapping (whether or not it was applied).
    """
    dotenv = Path(path) if path is not None else _find_dotenv()
    if dotenv is None or not Path(dotenv).is_file():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in Path(dotenv).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


# ── Key Vault support (Microsoft Fabric) ─────────────────────────────────────

def _from_key_vault(name: str) -> str | None:
    """
    Reads a secret from Azure Key Vault when running inside Fabric.

    Returns ``None`` outside Fabric, or when the secret does not exist,
    so that resolution can fall through to the environment.
    """
    try:
        import notebookutils  # type: ignore[import-not-found]
    except ImportError:
        try:
            import mssparkutils as notebookutils  # type: ignore[import-not-found,no-redef]
        except ImportError:
            return None

    try:
        secret_name = name.lower().replace("_", "-")
        return notebookutils.credentials.getSecret(KEY_VAULT_NAME, secret_name)
    except Exception:  # noqa: BLE001 - vault unreachable or secret absent
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def get_secret(
    name: str,
    default: str | None = None,
    required: bool = True,
) -> str | None:
    """
    Resolves a secret from Key Vault, the environment or ``.env``.

    Args:
        name:     Secret name, e.g. ``"ATLAS_URI"``.
        default:  Value returned when the secret is absent and not required.
        required: When True, a missing secret raises ``MissingSecretError``.

    Raises:
        MissingSecretError: if ``required`` and the secret cannot be found.
    """
    value = _from_key_vault(name)
    if value:
        return value

    value = os.environ.get(name)
    if value:
        return value

    load_dotenv()
    value = os.environ.get(name)
    if value:
        return value

    if required:
        raise MissingSecretError(
            f"Secret '{name}' is not configured.\n"
            f"  · Local:  copy .env.example to .env and fill in '{name}'\n"
            f"  ·         or export {name}=... in your shell\n"
            f"  · Fabric: add secret '{name.lower().replace('_', '-')}' "
            f"to Key Vault '{KEY_VAULT_NAME}'\n"
            f"  · CI:     add {name} to the repository secrets\n"
            f"Never hardcode credentials in the source tree."
        )
    return default


def atlas_uri() -> str:
    """MongoDB Atlas connection string (secret ``ATLAS_URI``)."""
    return get_secret("ATLAS_URI")  # type: ignore[return-value]


def confluent_config(bootstrap: str | None = None) -> dict[str, str]:
    """
    Builds the confluent-kafka client configuration from secrets.

    Secrets: ``CONFLUENT_BOOTSTRAP``, ``CONFLUENT_API_KEY``,
    ``CONFLUENT_API_SECRET``.
    """
    return {
        "bootstrap.servers": bootstrap or get_secret("CONFLUENT_BOOTSTRAP"),
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": get_secret("CONFLUENT_API_KEY"),
        "sasl.password": get_secret("CONFLUENT_API_SECRET"),
    }
