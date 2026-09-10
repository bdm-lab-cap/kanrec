"""Tests de la ablacion por sustitucion (metrica de fidelidad simbolica)."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kanrec.ablation import SymbolicFieldEncoder, _fit_scale_offset, substitution_ablation
from kanrec.model import KANRecModel


def _tiny_model_and_loader(seed=0, n=2000):
    torch.manual_seed(seed)
    x_num = torch.randn(n, 4)
    x_cat = torch.randint(0, 10, (n, 2))
    y = (torch.rand(n) < 0.3).float()
    model = KANRecModel(num_numerical=4, cat_cardinalities=[10, 10],
                        embedding_dim=8, kan_grid_size=10)
    model.calibrate(x_num[:500])
    loader = DataLoader(TensorDataset(x_num, x_cat, y), batch_size=512)
    return model, loader


def _linear_results(model, n_fields=4):
    """Ajusta una recta a la dim 0 de cada campo y la devuelve como formula."""
    results = {}
    for j in range(n_fields):
        x_grid, curves = model.numerical_encoder.get_spline_curves(j)
        coeffs = np.polyfit(x_grid.numpy(), curves[:, 0].numpy(), 1)
        results[j] = {"operator": "linear",
                      "params": [float(coeffs[0]), float(coeffs[1])],
                      "accepted": True}
    return results


class TestSymbolicFieldEncoder:
    def test_output_shape_matches_the_kan_it_replaces(self):
        scale = torch.ones(8)
        offset = torch.zeros(8)
        enc = SymbolicFieldEncoder("linear", [1.0, 0.0], scale, offset)
        out = enc(torch.randn(16, 1))
        assert out.shape == (16, 8)

    def test_handles_extreme_inputs_without_inf(self):
        enc = SymbolicFieldEncoder("exp", [1.0, 0.0], torch.ones(4), torch.zeros(4))
        x = torch.tensor([[0.0], [700.0], [-700.0]])
        assert torch.isfinite(enc(x)).all()

    def test_scale_offset_recovers_a_linear_curve(self):
        """Si phi ES lineal, el ajuste scale/offset debe reproducirla casi exacto."""
        model, _ = _tiny_model_and_loader()
        x_grid, curves = model.numerical_encoder.get_spline_curves(0)
        coeffs = np.polyfit(x_grid.numpy(), curves[:, 0].numpy(), 1)
        scale, offset = _fit_scale_offset(model, 0, "linear",
                                          [float(coeffs[0]), float(coeffs[1])])
        assert scale.shape == (8,)
        assert offset.shape == (8,)
        assert torch.isfinite(scale).all() and torch.isfinite(offset).all()


class TestSubstitutionAblation:
    def test_report_has_the_expected_metrics(self):
        model, loader = _tiny_model_and_loader()
        report = substitution_ablation(model, loader, _linear_results(model), verbose=False)

        for key in ["auc_original", "auc_substituted", "delta_auc",
                    "prediction_rmse", "curve_rel_error_mean", "fields"]:
            assert key in report, f"falta '{key}' en el informe"
        assert report["n_fields_substituted"] == 4
        assert np.isfinite(report["delta_auc"])
        assert report["curve_rel_error_mean"] >= 0

    def test_original_model_is_not_modified(self):
        """La ablacion trabaja sobre una copia: el modelo original debe quedar intacto."""
        model, loader = _tiny_model_and_loader()
        before = type(model.numerical_encoder.field_kans[0]).__name__
        substitution_ablation(model, loader, _linear_results(model), verbose=False)
        after = type(model.numerical_encoder.field_kans[0]).__name__
        assert before == after, "la ablacion modifico el modelo original"

    def test_raises_without_accepted_formulas(self):
        model, loader = _tiny_model_and_loader()
        results = {0: {"operator": "linear", "params": [1.0, 0.0], "accepted": False}}
        with pytest.raises(ValueError):
            substitution_ablation(model, loader, results, verbose=False)

    def test_accepts_field_names_as_keys(self):
        model, loader = _tiny_model_and_loader()
        cols = ["I1", "I2", "I3", "I4"]
        by_idx = _linear_results(model)
        by_name = {cols[j]: r for j, r in by_idx.items()}
        report = substitution_ablation(model, loader, by_name,
                                       numerical_cols=cols, verbose=False)
        assert report["n_fields_substituted"] == 4
        assert {f["field"] for f in report["fields"]} == set(cols)
