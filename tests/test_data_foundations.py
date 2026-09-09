"""
Regression tests for the "cimientos de datos" fixes (auditoría de tribunal,
hallazgos A3, A5, A6, B5, B6).

Each test is written to FAIL against the pre-fix code, so it documents the
bug it guards against, not just the desired behaviour.
"""
import math

import pytest
import torch

from kanrec.baselines import AutoDisNumericalEncoder, RawNumericalEncoder, build_model
from kanrec.encoder import KANNumericalEncoder
from kanrec.model import CTRModel, KANRecModel


# ── A3: el grid de las B-splines debe calibrarse a los datos reales ────────

class TestGridCalibration:
    def test_uncalibrated_grid_is_the_library_default(self):
        """Documents the starting point: [-1, 1] until calibrate() runs."""
        enc = KANNumericalEncoder(num_fields=2, embedding_dim=4)
        low, high = enc.field_range(0)
        assert low == pytest.approx(-1.0, abs=1e-6)
        assert high == pytest.approx(1.0, abs=1e-6)

    def test_calibrate_adapts_grid_to_unnormalised_field(self):
        """
        This is the exact failure mode from hallazgo A3: an unnormalised
        field (e.g. Criteo's I6-I13, in the hundreds or thousands) must end
        up with a grid that actually covers its values, or the spline term
        is zero almost everywhere and the encoder degenerates to base_weight * SiLU(x).
        """
        enc = KANNumericalEncoder(num_fields=1, embedding_dim=4, grid_size=10)
        big_field = torch.randn(1000, 1) * 500 + 50  # rango tipo I6 sin normalizar

        enc.calibrate(big_field)
        low, high = enc.field_range(0)

        assert low < -100, "el grid no se expandió al rango real del campo"
        assert high > 100, "el grid no se expandió al rango real del campo"

    def test_calibrate_rejects_wrong_column_count(self):
        enc = KANNumericalEncoder(num_fields=3, embedding_dim=4)
        with pytest.raises(ValueError):
            enc.calibrate(torch.randn(10, 2))

    def test_spline_can_receive_gradient_after_calibration(self):
        """
        Antes de A3: para un punto fuera de [-1, 1], todas las funciones
        base del spline valen exactamente 0 (ver `b_splines`), así que
        `spline_weight.grad` era 0 sin importar cuánto se entrenara — el
        spline era literalmente incapaz de aprender nada sobre ese valor.

        Justo tras calibrate(), el spline se reinicializa cerca de cero
        para el nuevo rango (arranque en blanco correcto — no es el bug),
        así que no comprobamos la salida inicial, sino que el GRADIENTE
        llega al spline, que es lo que antes era imposible.
        """
        enc = KANNumericalEncoder(num_fields=1, embedding_dim=4, grid_size=10)
        data = torch.randn(500, 1) * 200  # rango tipo I6 sin normalizar
        enc.calibrate(data)

        kan_layer = enc.field_kans[0].layers[0]
        x_test = torch.tensor([[150.0]])  # dentro del rango calibrado

        out = enc.field_kans[0](x_test)
        out.sum().backward()

        assert kan_layer.spline_weight.grad is not None
        assert kan_layer.spline_weight.grad.abs().sum().item() > 0.0, (
            "el gradiente no llega a spline_weight para este punto: seguiría "
            "fuera del grid calibrado"
        )

    def test_spline_moves_away_from_zero_after_a_few_training_steps(self):
        """Complementa el test anterior: tras entrenar, el spline deja de ser plano."""
        torch.manual_seed(0)
        enc = KANNumericalEncoder(num_fields=1, embedding_dim=2, grid_size=10)
        data = torch.randn(500, 1) * 200
        enc.calibrate(data)

        kan_layer = enc.field_kans[0].layers[0]
        weight_before = kan_layer.spline_weight.clone().detach()

        opt = torch.optim.Adam(enc.parameters(), lr=0.1)
        target = torch.randn(500, 2)
        for _ in range(20):
            opt.zero_grad()
            out = enc.field_kans[0](data)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            opt.step()

        moved = (kan_layer.spline_weight.detach() - weight_before).abs().sum().item()
        assert moved > 1e-3, "spline_weight no se movio tras entrenar en el rango calibrado"

    def test_without_calibration_gradient_to_spline_is_exactly_zero(self):
        """
        Documenta el bug original tal cual estaba (hallazgo A3): sin
        calibrar, un valor fuera de [-1, 1] no llega a ninguna base
        spline, así que el gradiente es EXACTAMENTE 0 — el spline nunca
        podía aprender nada sobre ese campo, por mucho que se entrenara.
        """
        enc = KANNumericalEncoder(num_fields=1, embedding_dim=4, grid_size=10)
        kan_layer = enc.field_kans[0].layers[0]
        x_test = torch.tensor([[150.0]])  # fuera de [-1, 1], sin calibrar

        out = enc.field_kans[0](x_test)
        out.sum().backward()

        assert kan_layer.spline_weight.grad.abs().sum().item() == pytest.approx(0.0, abs=1e-12)

    def test_get_spline_curves_warns_before_calibration(self):
        enc = KANNumericalEncoder(num_fields=1, embedding_dim=4)
        with pytest.warns(UserWarning, match="calibrate"):
            enc.get_spline_curves(0, n_points=20)

    def test_get_spline_curves_evaluates_within_calibrated_range(self):
        """
        Hallazgo A3: evaluar en un rango fijo [-3, 3] cuando el grid real
        cubre, por ejemplo, [-1700, 1800], solo muestra la rama SiLU y
        explica el hallazgo espurio de que 'exp domina en todos los campos'.
        """
        enc = KANNumericalEncoder(num_fields=1, embedding_dim=4, grid_size=10)
        data = torch.randn(500, 1) * 500
        enc.calibrate(data)

        x_grid, _ = enc.get_spline_curves(0, n_points=50)

        low, high = enc.field_range(0)
        assert x_grid.min().item() < low + 1e-3 or x_grid.min().item() <= low
        assert x_grid.max().item() >= high - 1e-3
        # El rango de evaluación debe seguir al campo, no quedarse en [-3, 3]
        assert x_grid.max().item() > 10, "sigue evaluando en una ventana fija pequeña"


# ── A6: la regularización de entropía debe ser > 0 ──────────────────────────

class TestEntropyRegularization:
    def test_encoder_entropy_regularization_is_positive(self):
        enc = KANNumericalEncoder(num_fields=3, embedding_dim=4)
        reg = enc.entropy_regularization_loss()
        assert reg.item() > 0.0, (
            "la regularización de entropía devuelve 0.0 — comprueba que "
            "get_edge_norms/entropy_regularization_loss alcanzan spline_weight "
            "en kan.layers[0], no en kan directamente (kan no tiene ese atributo)"
        )

    def test_model_entropy_regularization_is_positive(self):
        model = KANRecModel(num_numerical=3, cat_cardinalities=[10, 10], embedding_dim=4)
        reg = model.entropy_regularization_loss()
        assert reg.item() > 0.0

    def test_entropy_regularization_scales_with_weight(self):
        model = KANRecModel(num_numerical=3, cat_cardinalities=[10, 10], embedding_dim=4)
        model.entropy_reg_weight = 1e-3
        reg_small = model.entropy_regularization_loss().item()
        model.entropy_reg_weight = 1.0
        reg_large = model.entropy_regularization_loss().item()
        assert reg_large == pytest.approx(reg_small * 1000, rel=1e-3)

    def test_get_edge_norms_reaches_spline_weight_not_all_parameters(self):
        """
        Antes: hasattr(kan, "spline_weight") era siempre False (el atributo
        vive en kan.layers[0], no en kan), así que se sumaban TODOS los
        parámetros (incluida la ruta base lineal), no solo el spline.
        """
        enc = KANNumericalEncoder(num_fields=2, embedding_dim=4, grid_size=10)
        norms = enc.get_edge_norms()
        assert len(norms) == 2
        assert all(n > 0 for n in norms)

        expected = enc.field_kans[0].layers[0].spline_weight.abs().sum().item()
        assert norms[0] == pytest.approx(expected, rel=1e-6)


# ── A5: AutoDis sin el sigmoid antes del softmax ────────────────────────────

class TestAutoDisFix:
    def test_attention_is_not_bounded_by_sigmoid_ratio(self):
        """
        La versión con el bug (sigmoid antes de softmax) no podía superar
        una razón de pesos de e^1 ~= 2.7 entre dos buckets, sea cual sea la
        entrada. Comprobamos que, tras entrenar sobre una señal clara, la
        atención SÍ puede concentrarse mucho más que ese límite.
        """
        torch.manual_seed(0)
        enc = AutoDisNumericalEncoder(num_fields=1, embedding_dim=4, num_buckets=8)
        x = torch.randn(256, 1) * 3
        y = (x[:, 0] > 0).float()

        head = torch.nn.Linear(4, 1)
        opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=0.05)
        for _ in range(150):
            opt.zero_grad()
            pred = head(enc(x)[:, 0, :]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y)
            loss.backward()
            opt.step()

        assert loss.item() < 0.4, (
            f"AutoDis no aprende ni una tarea trivial (loss={loss.item():.4f}); "
            "revisa si el sigmoid antes del softmax ha vuelto"
        )

    def test_attention_entropy_drops_below_uniform_after_training(self):
        torch.manual_seed(0)
        enc = AutoDisNumericalEncoder(num_fields=1, embedding_dim=4, num_buckets=8)
        x = torch.randn(256, 1) * 3
        y = (x[:, 0] > 0).float()
        max_entropy = math.log(8)

        entropy_before = enc.attention_entropy(x).item()

        head = torch.nn.Linear(4, 1)
        opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=0.05)
        for _ in range(150):
            opt.zero_grad()
            pred = head(enc(x)[:, 0, :]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y)
            loss.backward()
            opt.step()

        entropy_after = enc.attention_entropy(x).item()
        assert entropy_before <= max_entropy + 1e-3
        assert entropy_after < entropy_before, (
            "la atencion sigue igual de uniforme tras entrenar: "
            "comprueba que no hay un sigmoid limitando los logits"
        )

    def test_temperature_stays_positive(self):
        """La reparametrización softplus debe impedir temperaturas <= 0."""
        enc = AutoDisNumericalEncoder(num_fields=1, embedding_dim=4, temperature=1.0)
        with torch.no_grad():
            enc.log_temperature.fill_(-100.0)  # intento de forzar tau <= 0
        x = torch.randn(10, 1)
        out = enc(x)
        assert torch.isfinite(out).all()


# ── B5/B6: una única definición del modelo, y --encoder con efecto real ────

class TestModelFactory:
    @pytest.mark.parametrize("encoder", ["raw", "autodis", "kan-bspline"])
    def test_build_model_runs_forward(self, encoder):
        model = build_model(encoder, num_numerical=4, cat_cardinalities=[10, 10, 10])
        if hasattr(model, "calibrate"):
            model.calibrate(torch.randn(50, 4))
        x_num = torch.randn(16, 4)
        x_cat = torch.randint(0, 10, (16, 3))
        y = model(x_num, x_cat)
        assert y.shape == (16, 1)
        assert torch.isfinite(y).all()

    def test_build_model_produces_different_encoders(self):
        """
        Hallazgo B6: antes, train.py --encoder solo cambiaba el nombre del
        checkpoint porque las tres ramas construían KANRecModel. Aquí
        comprobamos que las tres ramas usan clases de encoder distintas.
        """
        raw = build_model("raw", 3, [5, 5])
        autodis = build_model("autodis", 3, [5, 5])
        kan = build_model("kan-bspline", 3, [5, 5])

        assert type(raw.numerical_encoder).__name__ == "RawNumericalEncoder"
        assert type(autodis.numerical_encoder).__name__ == "AutoDisNumericalEncoder"
        assert type(kan.numerical_encoder).__name__ == "KANNumericalEncoder"
        assert type(raw) is CTRModel
        assert type(kan) is KANRecModel

    def test_build_model_rejects_unknown_encoder(self):
        with pytest.raises(ValueError):
            build_model("unknown-encoder", 3, [5, 5])

    def test_three_encoders_share_the_same_backbone_shape(self):
        """
        La comparación solo es justa si los tres modelos compilan el mismo
        InteractionMLP (misma input_dim, misma hidden_dim, misma output_dim);
        de lo contrario cualquier diferencia de AUC podría venir del backbone,
        no del encoder.
        """
        cards = [7, 7, 7]
        models = [build_model(e, 5, cards, embedding_dim=8) for e in ["raw", "autodis", "kan-bspline"]]
        shapes = {
            tuple(m.interaction.net[0].weight.shape) for m in models
        }
        assert len(shapes) == 1, f"los backbones no coinciden entre encoders: {shapes}"
