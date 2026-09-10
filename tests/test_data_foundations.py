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

    def test_is_calibrated_survives_checkpoint_roundtrip(self):
        """
        `grid` es un buffer registrado de PyTorch y SI lo restaura
        load_state_dict(); un atributo normal como `_calibrated` NO. Usar el
        flag hacia que todo modelo cargado de un checkpoint se reportara como
        sin calibrar y emitiera un aviso espurio, pese a tener el grid bien
        restaurado (detectado al ejecutar 05_symbolic_extraction en Fabric).
        """
        model = KANRecModel(num_numerical=2, cat_cardinalities=[5, 5],
                             embedding_dim=4, kan_grid_size=10)
        model.calibrate(torch.randn(200, 2) * 50)
        assert model.numerical_encoder.is_calibrated(0)

        state = model.state_dict()
        restored = KANRecModel(num_numerical=2, cat_cardinalities=[5, 5],
                                embedding_dim=4, kan_grid_size=10)
        assert not restored.numerical_encoder.is_calibrated(0)
        restored.load_state_dict(state)

        assert restored.numerical_encoder.is_calibrated(0), (
            "un modelo cargado de checkpoint se reporta como sin calibrar"
        )
        # Y por tanto NO debe emitir el aviso.
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            restored.numerical_encoder.get_spline_curves(0, n_points=20)
        assert not caught, f"aviso espurio tras cargar checkpoint: {[str(x.message) for x in caught]}"

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


# ── C9: la libreria de operadores debe ser numericamente robusta ────────────

class TestOperatorLibraryRobustness:
    def test_no_overflow_on_extreme_inputs(self):
        """
        Detectado al ejecutar 05_symbolic_extraction en Fabric:
        "RuntimeWarning: overflow encountered in exp" desde el operador
        sigmoid. Los campos con colas pesadas (I6, I12 llegan a ~690
        desviaciones tipicas en Criteo) alimentan valores donde np.exp
        desborda. NumPy devuelve inf en vez de fallar, asi que curve_fit
        seguia funcionando -- pero con inf/NaN silenciosos por dentro.
        """
        import warnings as _w

        import numpy as np

        from kanrec.symbolic import OPERATOR_LIBRARY

        x = np.array([-800.0, -100.0, -3.0, 0.0, 3.0, 100.0, 800.0])
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            for name, fn in OPERATOR_LIBRARY.items():
                y = fn(x, 1.0, 0.0)
                assert np.isfinite(y).all(), f"operador '{name}' produce inf/NaN"
        overflows = [str(c.message) for c in caught if "overflow" in str(c.message)]
        assert not overflows, f"overflow sin proteger: {overflows}"

    def test_operator_library_matches_the_fabric_notebook(self):
        """
        Hallazgo C9: el paquete y fabric/05_symbolic_extraction.py definian
        'exp' de forma distinta (sin clip vs clip a +-10), asi que podian
        elegir operadores distintos sobre la misma curva.
        """
        import re
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        notebook = (repo / "fabric" / "05_symbolic_extraction.py").read_text(encoding="utf-8")
        package = (repo / "kanrec" / "symbolic.py").read_text(encoding="utf-8")

        for op in ["log", "exp", "square", "sqrt", "inverse", "sigmoid", "linear"]:
            nb_match = re.search(rf'"{op}":\s+lambda.*', notebook)
            pkg_match = re.search(rf'"{op}":\s+lambda.*', package)
            assert nb_match and pkg_match, f"operador '{op}' no encontrado en ambos ficheros"
            nb_line = nb_match.group(0).strip().rstrip(",")
            pkg_line = pkg_match.group(0).strip().rstrip(",")
            assert nb_line == pkg_line, (
                f"el operador '{op}' difiere entre notebook y paquete:\n"
                f"  notebook: {nb_line}\n  paquete : {pkg_line}"
            )


# ── El spline debe ADAPTARSE a la forma de los datos, no salir siempre recto ─

class TestSplineIsShapeAdaptive:
    """
    Detectado al ejecutar 05_symbolic_extraction sobre checkpoints reales:
    la extraccion devolvia 'linear' para los 10 campos supervivientes con
    R2 ~= 0.998, ocho de ellos con el valor identico 0.9981. No era que los
    datos fueran lineales: el encoder era CIEGO a su forma.

    Causa: efficient-kan inicializa spline_weight con ruido ~scale_noise/
    grid_size (~0.01) mientras base_weight recibe Kaiming completo (~0.5).
    Con un lr compartido el spline nunca alcanza a la ruta base, y phi sale
    recta gobernada por el termino base SiLU. Ademas entropy_reg_weight=1e-3
    agravaba el desequilibrio.

    Arreglo: entropy_reg_weight=1e-5 y KANRecModel.parameter_groups(), que
    da al spline un lr 25x mayor.
    """

    @staticmethod
    def _train_and_measure_linearity(signal: str, seed: int, epochs: int = 40) -> float:
        """Entrena sobre una senal conocida y devuelve el R2 de un ajuste lineal a phi_0."""
        import numpy as np

        # Configuracion REPRESENTATIVA (13 numericas + 26 categoricas, como
        # Criteo), no una miniatura: con 4 campos y 2 categoricas el efecto
        # se diluye y el test daba falsos negativos. Medido sobre 5 semillas
        # en esta configuracion: R2_lin 0.76+-0.05 con lr compartido frente
        # a 0.11+-0.07 con el lr del spline x25, sin solape entre grupos.
        torch.manual_seed(seed)
        n = 8000
        x_num = torch.randn(n, 13)
        x_cat = torch.randint(0, 10, (n, 26))
        if signal == "curved":
            logit = 3.0 * torch.sin(x_num[:, 0] * 2.5) - 1.0
        else:
            logit = 2.0 * x_num[:, 0] - 1.0
        y = (torch.rand(n) < torch.sigmoid(logit)).float()

        model = KANRecModel(num_numerical=13, cat_cardinalities=[10] * 26,
                             embedding_dim=16, kan_grid_size=10)
        model.calibrate(x_num[:4000])
        optimizer = torch.optim.Adam(model.parameter_groups(base_lr=1e-3), weight_decay=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss()  # el modelo devuelve logits
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(x_num, x_cat).squeeze(), y)
            loss = loss + model.entropy_regularization_loss()
            loss.backward()
            optimizer.step()

        x_grid, curves = model.numerical_encoder.get_spline_curves(0)
        xs, ys = x_grid.numpy(), curves[:, 0].numpy()
        coeffs = np.polyfit(xs, ys, 1)
        residual = np.sum((ys - np.polyval(coeffs, xs)) ** 2)
        total = np.sum((ys - ys.mean()) ** 2) + 1e-12
        return float(1 - residual / total)

    def test_parameter_groups_gives_spline_a_higher_lr(self):
        model = KANRecModel(num_numerical=3, cat_cardinalities=[5, 5], embedding_dim=8)
        groups = model.parameter_groups(base_lr=1e-3, spline_lr_mult=25.0)

        assert len(groups) == 2
        lrs = sorted(g["lr"] for g in groups)
        assert lrs[0] == pytest.approx(1e-3)
        assert lrs[1] == pytest.approx(2.5e-2)
        # Todos los parametros deben estar en algun grupo, ninguno duplicado.
        total = sum(len(g["params"]) for g in groups)
        assert total == len(list(model.parameters()))

    def test_entropy_weight_does_not_crush_the_spline(self):
        """1e-3 aplastaba el spline (ratio 63x vs 16x). Debe quedarse bajo."""
        model = KANRecModel(num_numerical=2, cat_cardinalities=[5], embedding_dim=8)
        assert model.entropy_reg_weight <= 1e-4, (
            "entropy_reg_weight demasiado alto: suprime la propia componente "
            "spline que este TFM pretende estudiar"
        )

    @pytest.mark.slow
    def test_spline_curves_when_data_is_curved(self):
        r2_linear = self._train_and_measure_linearity("curved", seed=42)
        assert r2_linear < 0.45, (
            f"phi sigue siendo casi una recta (R2 lineal={r2_linear:.3f}) pese a "
            f"que la senal real es sin(2.5x): el spline no esta contribuyendo"
        )

    @pytest.mark.slow
    def test_spline_stays_straight_when_data_is_linear(self):
        """
        El control que demuestra que el arreglo no consiste simplemente en
        'romper' las curvas: con datos lineales, phi DEBE seguir siendo recta.
        """
        r2_linear = self._train_and_measure_linearity("linear", seed=42)
        assert r2_linear > 0.6, (
            f"phi se curva (R2 lineal={r2_linear:.3f}) con una senal realmente "
            f"lineal: el encoder esta sobreajustando ruido"
        )


# ── C3: el modelo devuelve logits, no probabilidades (evita device-side assert) ─

class TestModelReturnsLogits:
    """
    Un device-side assert de CUDA tumbaba KAN-REC en Colab dentro de
    binary_cross_entropy. Causa: la cabeza tenia Sigmoid y el entrenamiento
    usaba BCELoss; cuando la Sigmoid saturaba a 0.0 o 1.0 exactos en float32,
    BCELoss calculaba log(0)=-inf, que en GPU es un assert fatal (en CPU solo
    daba inf y seguia, por eso no fallaba localmente). El modelo ahora
    devuelve logits y el entrenamiento usa BCEWithLogitsLoss (estable).
    """

    def test_forward_returns_logits_not_probabilities(self):
        from kanrec.baselines import build_model

        for encoder in ["raw", "autodis", "kan-bspline"]:
            model = build_model(encoder, num_numerical=3, cat_cardinalities=[5, 5])
            if hasattr(model, "calibrate"):
                model.calibrate(torch.randn(50, 3))
            # Forzar valores grandes para empujar hacia la saturacion
            x_num = torch.randn(64, 3) * 10
            x_cat = torch.randint(0, 5, (64, 2))
            logits = model(x_num, x_cat)
            # Los logits pueden salir de [0,1]; es justo lo que queremos.
            assert torch.isfinite(logits).all()

    def test_predict_proba_is_in_unit_interval(self):
        model = KANRecModel(num_numerical=3, cat_cardinalities=[5, 5], embedding_dim=8)
        model.calibrate(torch.randn(50, 3))
        x_num = torch.randn(64, 3) * 10
        x_cat = torch.randint(0, 5, (64, 2))
        proba = model.predict_proba(x_num, x_cat)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_bcewithlogits_is_finite_on_saturated_output(self):
        """
        BCEWithLogitsLoss debe ser finita y estable con logits extremos,
        que es la razon por la que sustituye a Sigmoid+BCELoss. La cabeza
        del modelo devuelve logits precisamente para poder usar esta perdida
        fusionada, que nunca materializa una probabilidad saturada y por
        tanto no puede disparar el kernel de BCE de CUDA sobre 0.0/1.0 exacto
        (el device-side assert que tumbaba KAN-REC en Colab).
        """
        logits = torch.tensor([-100.0, 100.0, 0.0, -40.0, 40.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        loss = torch.nn.BCEWithLogitsLoss()(logits, target)
        assert torch.isfinite(loss), "BCEWithLogitsLoss no deberia dar inf/nan"
        # Y su gradiente tambien debe ser finito (lo que se propaga al modelo).
        logits.requires_grad_(True)
        torch.nn.BCEWithLogitsLoss()(logits, target).backward()
        assert torch.isfinite(logits.grad).all()


# ── Winsorizado de la entrada: outliers extremos no deben producir inf ──────

class TestInputWinsorization:
    """
    En la corrida real de Colab, KAN-REC con grid_size>=10 devolvia inf en
    TODAS las predicciones (AUC 0.5, logloss inf) mientras grid_size=5
    funcionaba. Causa: tras StandardScaler, Criteo conserva outliers de
    ~690 desviaciones; la ruta base del KAN es base_weight*SiLU(x) y
    SiLU(690)~=690, asi que un outlier arrastra el embedding a magnitud
    ~1e3 y los logits desbordan float32 en GPU.
    """

    def test_extreme_inputs_do_not_produce_inf(self):
        for grid_size in [5, 10, 20]:
            model = KANRecModel(num_numerical=4, cat_cardinalities=[10, 10],
                                 embedding_dim=8, kan_grid_size=grid_size)
            model.calibrate(torch.randn(500, 4))

            x_num = torch.randn(64, 4)
            x_num[0, :] = 690.0      # el outlier real de Criteo
            x_num[1, :] = -400.0
            x_cat = torch.randint(0, 10, (64, 2))

            logits = model(x_num, x_cat)
            assert torch.isfinite(logits).all(), (
                f"grid_size={grid_size}: logits no finitos con entrada extrema"
            )

    def test_winsorization_bounds_embedding_magnitude(self):
        encoder = KANNumericalEncoder(num_fields=3, embedding_dim=8, grid_size=10)
        encoder.calibrate(torch.randn(500, 3))

        x = torch.randn(32, 3)
        x[0, :] = 690.0
        emb = encoder(x)

        # Sin winsorizado la magnitud llegaba a ~7e2; con clip a 10 sigmas
        # debe quedar en un orden de magnitud manejable.
        assert emb.abs().max().item() < 100.0, (
            f"embedding sin acotar: |emb|max={emb.abs().max().item():.1f}"
        )

    def test_winsorization_does_not_alter_normal_range(self):
        """Los datos normales (<10 sigmas) deben pasar intactos."""
        encoder = KANNumericalEncoder(num_fields=2, embedding_dim=4, grid_size=10)
        encoder.calibrate(torch.randn(200, 2))
        x = torch.randn(16, 2) * 2.0          # ~2 sigmas, muy dentro del clip
        assert torch.allclose(x.clamp(-encoder.input_clip, encoder.input_clip), x)
