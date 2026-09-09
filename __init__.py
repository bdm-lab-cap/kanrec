"""
KAN-REC — continuous numerical encoding and symbolic scoring extraction.

Public symbols are imported lazily (PEP 562) so that lightweight modules
such as :mod:`kanrec.config` can be used from a Spark or Fabric notebook
without pulling in torch, scipy or pymongo.
"""
from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.2.0"

_LAZY: dict[str, str] = {
    "KANNumericalEncoder": ".encoder",
    "KANRecModel": ".model",
    "SymbolicExtractor": ".symbolic",
    "FaithfulnessEvaluator": ".faithfulness",
    "MongoSymbolicStore": ".mongo_store",
    "get_secret": ".config",
    "atlas_uri": ".config",
    "confluent_config": ".config",
}

__all__ = [*_LAZY, "__version__"]

if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from .config import atlas_uri, confluent_config, get_secret
    from .encoder import KANNumericalEncoder
    from .faithfulness import FaithfulnessEvaluator
    from .model import KANRecModel
    from .mongo_store import MongoSymbolicStore
    from .symbolic import SymbolicExtractor


def __getattr__(name: str):
    """Imports a public symbol on first access."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'kanrec' has no attribute '{name}'")
    return getattr(import_module(module_path, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
