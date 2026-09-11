"""
Tests del esquema unico de MongoDB y de la fijacion de semillas.

El esquema existe porque habia DOS escritores con nombres de campo distintos
('field' vs 'field_name', 'accepted' vs 'is_accepted'), de modo que los
documentos de uno eran invisibles para las agregaciones del otro.
"""
import random

import numpy as np
import pytest
import torch

from kanrec.reproducibility import set_seed
from kanrec.schema import NATURAL_KEY, SymbolicResult


class TestSymbolicResultSchema:
    def test_document_uses_the_canonical_field_names(self):
        """
        Los nombres deben ser los que consultan las agregaciones de
        MongoSymbolicStore: 'field_name' e 'is_accepted', no 'field'/'accepted'.
        """
        r = SymbolicResult(run_id="r1", dataset="criteo", seed=42,
                           field_name="I3", operator="linear",
                           formula_str="-0.72x + 0.03", r2=0.997, is_accepted=True)
        doc = r.to_document()

        assert "field_name" in doc and "field" not in doc
        assert "is_accepted" in doc and "accepted" not in doc
        assert doc["field_name"] == "I3"
        assert doc["is_accepted"] is True

    def test_natural_key_present_in_every_document(self):
        r = SymbolicResult(run_id="r1", dataset="criteo", seed=42,
                           field_name="I3", operator="linear",
                           formula_str="x", r2=0.9, is_accepted=True)
        doc = r.to_document()
        for key in NATURAL_KEY:
            assert key in doc, f"falta '{key}', necesaria para el indice unico"

    def test_from_fit_maps_the_16_dim_metrics(self):
        fit = {"operator": "linear", "params": [-0.721, 0.0319],
               "formula": "-0.721x + 0.0319", "accepted": True,
               "r2_mean": 0.9975, "r2_std": 0.0018, "r2_min": 0.9907,
               "operator_agreement": 1.0, "n_dims_fitted": 16}
        r = SymbolicResult.from_fit("run-1", "criteo", 42, "I3", fit, field_idx=2)
        doc = r.to_document()

        assert doc["operator"] == "linear"
        assert doc["param_a"] == pytest.approx(-0.721)
        assert doc["param_b"] == pytest.approx(0.0319)
        assert doc["r2_mean"] == pytest.approx(0.9975)
        assert doc["operator_agreement"] == 1.0
        assert doc["n_dims_fitted"] == 16

    def test_optional_fields_are_omitted_when_absent(self):
        r = SymbolicResult(run_id="r", dataset="d", seed=1, field_name="I1",
                           operator=None, formula_str="?", r2=0.0, is_accepted=False)
        doc = r.to_document()
        assert "curve_rel_error" not in doc
        assert "violation_rate" not in doc


class TestSetSeed:
    def test_all_generators_are_fixed(self):
        """torch.manual_seed solo no basta: numpy y random tambien deciden."""
        set_seed(123)
        t1, n1, r1 = torch.randn(5), np.random.rand(5), [random.random() for _ in range(5)]

        set_seed(123)
        t2, n2, r2 = torch.randn(5), np.random.rand(5), [random.random() for _ in range(5)]

        assert torch.equal(t1, t2), "torch no reproducible"
        assert np.allclose(n1, n2), "numpy no reproducible"
        assert r1 == r2, "random no reproducible"

    def test_different_seeds_give_different_draws(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        assert not torch.equal(a, b)

    def test_deterministic_flag_sets_cudnn(self):
        set_seed(42, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        # Restaurar para no afectar a otros tests
        torch.backends.cudnn.deterministic = False
