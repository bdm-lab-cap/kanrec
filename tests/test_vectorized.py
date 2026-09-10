"""
Tests del encoder vectorizado.

Lo que hay que garantizar por encima de todo: la optimizacion NO puede
cambiar el modelo. Si la salida difiere, no es una optimizacion, es un
modelo distinto y todos los resultados del TFM dejarian de aplicar.
"""
import pytest
import torch

from kanrec.model import KANRecModel
from kanrec.vectorized import VectorizedKANEncoder, vectorize_model


def _trained_model(seed=0, n_fields=13, grid_size=10):
    torch.manual_seed(seed)
    model = KANRecModel(num_numerical=n_fields, cat_cardinalities=[20] * 5,
                        embedding_dim=16, kan_grid_size=grid_size)
    model.calibrate(torch.randn(1000, n_fields) * 3)
    model.eval()
    return model


class TestNumericalEquivalence:
    """La vectorizacion cambia la velocidad, nunca el resultado."""

    def test_encoder_output_matches_original(self):
        model = _trained_model()
        vec = VectorizedKANEncoder.from_field_kans(model.numerical_encoder)
        x = torch.randn(128, 13) * 2

        with torch.no_grad():
            original = model.numerical_encoder(x)
            fast = vec(x)

        assert original.shape == fast.shape
        assert torch.allclose(original, fast, atol=1e-5), (
            f"diferencia maxima {(original - fast).abs().max().item():.2e}"
        )

    def test_full_model_output_matches(self):
        model = _trained_model()
        fast = vectorize_model(model, verbose=False)
        x_num = torch.randn(64, 13)
        x_cat = torch.randint(0, 20, (64, 5))

        with torch.no_grad():
            assert torch.allclose(model(x_num, x_cat), fast(x_num, x_cat), atol=1e-5)

    @pytest.mark.parametrize("grid_size", [5, 10, 20])
    def test_equivalence_across_grid_sizes(self, grid_size):
        model = _trained_model(grid_size=grid_size)
        vec = VectorizedKANEncoder.from_field_kans(model.numerical_encoder)
        x = torch.randn(64, 13) * 2
        with torch.no_grad():
            assert torch.allclose(model.numerical_encoder(x), vec(x), atol=1e-5)

    def test_equivalence_on_extreme_inputs(self):
        """El winsorizado debe aplicarse igual en ambas versiones."""
        model = _trained_model()
        vec = VectorizedKANEncoder.from_field_kans(model.numerical_encoder)
        x = torch.randn(32, 13)
        x[0, :] = 700.0        # el outlier real de Criteo
        x[1, :] = -400.0
        with torch.no_grad():
            original, fast = model.numerical_encoder(x), vec(x)
        assert torch.isfinite(fast).all()
        assert torch.allclose(original, fast, atol=1e-5)


class TestVectorizeModel:
    def test_original_model_is_not_modified(self):
        model = _trained_model()
        before = type(model.numerical_encoder).__name__
        vectorize_model(model, verbose=False)
        assert type(model.numerical_encoder).__name__ == before

    def test_non_kan_model_returned_unchanged(self):
        from kanrec.baselines import build_model

        raw = build_model("raw", num_numerical=4, cat_cardinalities=[10])
        assert vectorize_model(raw, verbose=False) is raw

    def test_fields_stay_independent(self):
        """
        Cada phi_j debe ver SOLO su campo. Si bmm mezclara campos (como haria
        un F.linear sobre las 13 entradas), cambiar un campo alteraria el
        embedding de los demas.
        """
        model = _trained_model()
        vec = VectorizedKANEncoder.from_field_kans(model.numerical_encoder)

        x1 = torch.zeros(1, 13)
        x2 = x1.clone()
        x2[0, 0] = 5.0          # solo se modifica el campo 0

        with torch.no_grad():
            out1, out2 = vec(x1), vec(x2)

        assert not torch.allclose(out1[:, 0], out2[:, 0]), "el campo 0 deberia cambiar"
        assert torch.allclose(out1[:, 1:], out2[:, 1:], atol=1e-6), (
            "los demas campos NO deben cambiar: se estan mezclando"
        )
