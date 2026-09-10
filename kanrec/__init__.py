"""
KAN-REC - continuous numerical encoding and symbolic scoring extraction.

Public symbols are imported lazily (PEP 562) so that lightweight modules
such as :mod:`kanrec.config` can be used from a Spark or Fabric notebook
without pulling in torch, scipy or pymongo.
"""
from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.3.0"

_LAZY: dict[str, str] = {
    "KANNumericalEncoder": ".encoder",
    "KANRecModel": ".model",
    "CTRModel": ".model",
    "InteractionMLP": ".model",
    "RawNumericalEncoder": ".baselines",
    "AutoDisNumericalEncoder": ".baselines",
    "build_model": ".baselines",
    "SymbolicExtractor": ".symbolic",
    "FaithfulnessEvaluator": ".faithfulness",
    "substitution_ablation": ".ablation",
    "measure_latency": ".latency",
    "VectorizedKANEncoder": ".vectorized",
    "vectorize_model": ".vectorized",
    "compare_latency": ".latency",
    "MongoSymbolicStore": ".mongo_store",
    "get_secret": ".config",
    "atlas_uri": ".config",
    "confluent_config": ".config",
    "apply_pipeline_and_unpack": ".spark_utils",
    "random_sample": ".spark_utils",
}

__all__ = [*_LAZY, "__version__"]

if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from .baselines import AutoDisNumericalEncoder, RawNumericalEncoder, build_model
    from .config import atlas_uri, confluent_config, get_secret
    from .spark_utils import apply_pipeline_and_unpack, random_sample
    from .encoder import KANNumericalEncoder
    from .ablation import substitution_ablation
    from .latency import compare_latency, measure_latency
    from .vectorized import VectorizedKANEncoder, vectorize_model
    from .faithfulness import FaithfulnessEvaluator
    from .model import CTRModel, InteractionMLP, KANRecModel
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
