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


class TestDependenciasCompatiblesConFabric:
    """
    Los minimos de version declarados no deben ser mas nuevos que los del
    runtime de Fabric, o pip los actualizara al instalar el paquete en un
    entorno y dejara una instalacion mixta.

    Fallo real: con `scipy>=1.12.0` el entorno quedo con los .so de una
    version y los .py de otra, y todo el runtime dejo de funcionar con
    `ImportError: cannot import name '_promote'`. El mismo riesgo tenian
    scikit-learn, pandas y pyarrow.
    """

    #: Versiones de referencia del runtime de Fabric (Spark 3.5 / Python 3.11).
    #: Declarar un minimo por encima de estas provoca una actualizacion.
    RUNTIME_FABRIC = {
        "torch": (2, 0),
        "scipy": (1, 11),
        "scikit-learn": (1, 3),
        "pandas": (2, 0),
        "pyarrow": (14, 0),
    }

    def _minimos_declarados(self) -> dict:
        import re
        from pathlib import Path

        setup = (Path(__file__).resolve().parent.parent / "setup.py").read_text()
        bloque = setup.split("install_requires=[")[1].split("]")[0]
        minimos = {}
        for linea in bloque.split("\n"):
            m = re.search(r'"([a-zA-Z0-9_.-]+)>=(\d+)\.(\d+)', linea)
            if m:
                minimos[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        return minimos

    def test_los_minimos_no_fuerzan_actualizacion_en_fabric(self):
        declarados = self._minimos_declarados()
        assert declarados, "no se pudieron leer los minimos de setup.py"

        conflictos = [
            f"{paquete}>={v[0]}.{v[1]} supera el runtime "
            f"({self.RUNTIME_FABRIC[paquete][0]}.{self.RUNTIME_FABRIC[paquete][1]})"
            for paquete, v in declarados.items()
            if paquete in self.RUNTIME_FABRIC and v > self.RUNTIME_FABRIC[paquete]
        ]
        assert not conflictos, (
            "Estos minimos provocarian que pip actualice paquetes del runtime "
            "de Fabric y rompa la instalacion:\n  - " + "\n  - ".join(conflictos)
        )

    def test_mlflow_sigue_fuera_de_las_dependencias_base(self):
        """
        mlflow debe permanecer en el extra [train]: instalarlo en Fabric
        sobreescribe el suyo y rompe el plugin synapse.ml.mlflow.
        """
        from pathlib import Path

        setup = (Path(__file__).resolve().parent.parent / "setup.py").read_text()
        base = setup.split("install_requires=[")[1].split("]")[0]
        assert "mlflow" not in base, "mlflow ha vuelto a las dependencias base"
