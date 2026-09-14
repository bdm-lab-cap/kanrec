"""Tests for SymbolicExtractor (without MongoDB — uses a mock store)."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from kanrec.symbolic import SymbolicExtractor, OPERATOR_LIBRARY


# ── Operator library tests ────────────────────────────────────────────────────

def test_operator_library_coverage():
    """All operators in the library should be callable with (x, a, b)."""
    x = np.linspace(-2, 2, 50)
    for name, fn in OPERATOR_LIBRARY.items():
        out = fn(x, 1.0, 0.0)
        assert out.shape == x.shape, f"Operator '{name}' shape mismatch"
        assert not np.any(np.isnan(out)), f"Operator '{name}' produced NaN"


def test_log_operator_known_values():
    x = np.array([0.0, 1.0, np.e - 1])
    y = OPERATOR_LIBRARY["log"](x, 1.0, 0.0)
    np.testing.assert_allclose(y, [0.0, np.log(2), 1.0], atol=1e-5)


def test_linear_operator():
    x = np.array([1.0, 2.0, 3.0])
    y = OPERATOR_LIBRARY["linear"](x, 2.0, 1.0)
    np.testing.assert_array_equal(y, [3.0, 5.0, 7.0])


# ── SymbolicExtractor unit tests (mocked model + store) ──────────────────────

def _make_mock_extractor():
    """Creates an extractor with a mocked model and disabled MongoDB."""
    mock_model = MagicMock()

    # Simulate a log-shaped curve for field 0
    import torch
    x_grid  = torch.linspace(-3, 3, 300)
    y_log   = torch.log(x_grid.abs() + 1).unsqueeze(1).expand(-1, 8)
    mock_model.numerical_encoder.get_spline_curves.return_value = (x_grid, y_log)
    mock_model.numerical_encoder.get_edge_norms.return_value = [1.0, 0.5, 0.8, 0.2, 0.9]

    # Ya no hace falta parchear MongoSymbolicStore: el extractor no abre
    # conexion salvo que se le pase mongo_uri explicitamente.
    extractor = SymbolicExtractor(mock_model, r2_threshold=0.90, l1_percentile=20)
    return extractor, mock_model


def test_prune_fields_respects_percentile():
    extractor, _ = _make_mock_extractor()
    surviving = extractor.prune_fields()
    # Bottom 20% of [1.0, 0.5, 0.8, 0.2, 0.9] is field with norm 0.2 → excluded
    assert 3 not in surviving  # field index 3 has norm 0.2


def test_fit_field_recovers_log():
    extractor, _ = _make_mock_extractor()
    result = extractor.fit_field(0)
    assert result["operator"] == "log", f"Expected 'log', got '{result['operator']}'"
    assert result["r2"] > 0.90
    assert result["accepted"] is True


def test_fit_field_formula_string_format():
    extractor, _ = _make_mock_extractor()
    result = extractor.fit_field(0)
    assert "log" in result["formula"]
    assert "|x|" in result["formula"] or "log" in result["formula"]


class TestSymbolicSinMongo:
    """
    La extraccion simbolica no debe depender de MongoDB.

    symbolic.py importaba mongo_store a nivel de modulo y el constructor
    creaba un MongoSymbolicStore contra localhost, asi que en un entorno sin
    pymongo -- Microsoft Fabric, donde no viene preinstalado -- la cadena
    `from kanrec.ablation import substitution_ablation` fallaba con
    ModuleNotFoundError antes de ejecutar nada. Persistir es una
    responsabilidad separada de extraer.
    """

    def test_no_se_importa_mongo_store_al_cargar_el_modulo(self):
        import kanrec.symbolic as mod

        fuente = open(mod.__file__).read()
        nivel_modulo = fuente.split("class SymbolicExtractor")[0]
        assert "from .mongo_store import" not in nivel_modulo, (
            "symbolic.py vuelve a importar mongo_store a nivel de modulo"
        )

    def test_extractor_se_construye_sin_uri(self):
        import torch

        from kanrec.model import KANRecModel
        from kanrec.symbolic import SymbolicExtractor

        model = KANRecModel(num_numerical=3, cat_cardinalities=[5, 5], embedding_dim=8)
        model.calibrate(torch.randn(200, 3))

        extractor = SymbolicExtractor(model)          # sin mongo_uri
        assert extractor.mongo_uri is None
        assert extractor._store is None               # no se abrio conexion

        resultado = extractor.fit_field(0, n_grid=64)
        assert resultado["operator"] is not None
        assert "operator_agreement" in resultado

    def test_acceder_al_store_sin_uri_explica_el_motivo(self):
        import torch

        from kanrec.model import KANRecModel
        from kanrec.symbolic import SymbolicExtractor

        model = KANRecModel(num_numerical=2, cat_cardinalities=[5], embedding_dim=4)
        model.calibrate(torch.randn(100, 2))
        extractor = SymbolicExtractor(model)

        with pytest.raises(RuntimeError, match="mongo_uri"):
            _ = extractor.store

    def test_close_sin_conexion_no_rompe(self):
        import torch

        from kanrec.model import KANRecModel
        from kanrec.symbolic import SymbolicExtractor

        model = KANRecModel(num_numerical=2, cat_cardinalities=[5], embedding_dim=4)
        model.calibrate(torch.randn(100, 2))
        SymbolicExtractor(model).close()      # no debe lanzar
