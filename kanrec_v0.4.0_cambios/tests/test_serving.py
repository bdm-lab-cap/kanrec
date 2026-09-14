"""
Tests de kanrec.serving: carga autocontenida de checkpoints, scoring y
manifiesto. Requieren torch (se saltan si no esta instalado).
"""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kanrec.baselines import build_model
from kanrec.drift import calibrated_ranges
from kanrec.serving import (
    Scorer, check_manifest, infer_architecture, load_manifest, load_model, write_manifest,
)
from kanrec.vectorized import VectorizedKANEncoder, VectorizedRawEncoder

N_NUM, CARDS, D = 5, [7, 3, 11], 8
NUM_COLS = [f"I{i}" for i in range(1, N_NUM + 1)]
IDX_COLS = [f"C{i}_idx" for i in range(1, len(CARDS) + 1)]


def _kan(seed=0, grid_size=6, order=3):
    torch.manual_seed(seed)
    m = build_model("kan-bspline", num_numerical=N_NUM, cat_cardinalities=CARDS,
                    embedding_dim=D, kan_grid_size=grid_size, kan_spline_order=order)
    m.calibrate(torch.randn(500, N_NUM) * torch.arange(1, N_NUM + 1))
    m.eval()
    return m


class TestInferArchitecture:
    def test_kan_hyperparameters_are_recovered(self):
        arch = infer_architecture(_kan(grid_size=6, order=3).state_dict())
        assert arch == {"encoder": "kan-bspline", "num_numerical": N_NUM, "embedding_dim": D,
                        "cat_cardinalities": CARDS, "grid_size": 6, "spline_order": 3}

    @pytest.mark.parametrize("grid_size,order", [(5, 3), (10, 3), (20, 3), (8, 2)])
    def test_grid_and_order_for_several_configs(self, grid_size, order):
        arch = infer_architecture(_kan(grid_size=grid_size, order=order).state_dict())
        assert (arch["grid_size"], arch["spline_order"]) == (grid_size, order)

    def test_raw_and_autodis(self):
        raw = build_model("raw", num_numerical=N_NUM, cat_cardinalities=CARDS, embedding_dim=D)
        assert infer_architecture(raw.state_dict())["encoder"] == "raw"
        ad = build_model("autodis", num_numerical=N_NUM, cat_cardinalities=CARDS,
                         embedding_dim=D, autodis_num_buckets=12)
        arch = infer_architecture(ad.state_dict())
        assert (arch["encoder"], arch["num_buckets"]) == ("autodis", 12)


class TestLoadModel:
    def test_round_trip_is_exact_and_vectorized(self, tmp_path):
        model = _kan()
        ckpt = tmp_path / "m.pt"
        torch.save(model.state_dict(), ckpt)

        loaded = load_model(str(ckpt))
        assert isinstance(loaded.numerical_encoder, VectorizedKANEncoder)
        # El grid calibrado viaja en el checkpoint: mismos rangos.
        assert calibrated_ranges(loaded.numerical_encoder) == pytest.approx(
            calibrated_ranges(model.numerical_encoder), abs=1e-6)
        x_num, x_cat = torch.randn(32, N_NUM), torch.randint(0, 3, (32, len(CARDS)))
        with torch.no_grad():
            assert torch.allclose(model(x_num, x_cat), loaded(x_num, x_cat), atol=1e-5)

    def test_raw_is_loaded_vectorized_and_unvectorized_option_works(self, tmp_path):
        raw = build_model("raw", num_numerical=N_NUM, cat_cardinalities=CARDS, embedding_dim=D)
        ckpt = tmp_path / "raw.pt"
        torch.save(raw.state_dict(), ckpt)
        assert isinstance(load_model(str(ckpt)).numerical_encoder, VectorizedRawEncoder)
        assert not isinstance(load_model(str(ckpt), vectorize=False).numerical_encoder,
                              VectorizedRawEncoder)


class TestScorer:
    def test_probabilities_shape_and_range(self, tmp_path):
        ckpt = tmp_path / "m.pt"
        torch.save(_kan().state_dict(), ckpt)
        scorer = Scorer.from_checkpoint(str(ckpt), NUM_COLS, IDX_COLS)
        p = scorer.predict_proba(np.random.randn(100, N_NUM), np.zeros((100, len(CARDS)), dtype=int),
                                 batch_size=32)
        assert p.shape == (100,)
        assert np.all((p >= 0) & (p <= 1))

    def test_out_of_range_indices_are_clamped_not_fatal(self, tmp_path):
        """Un indice categorico mayor que el embedding no aborta: se recorta
        y se cuenta, que es lo que un servicio debe hacer."""
        ckpt = tmp_path / "m.pt"
        torch.save(_kan().state_dict(), ckpt)
        scorer = Scorer.from_checkpoint(str(ckpt), NUM_COLS, IDX_COLS)
        cat = np.zeros((10, len(CARDS)), dtype=int)
        cat[0, 1] = 999          # fuera de rango
        cat[1, 0] = -5           # negativo
        p = scorer.predict_proba(np.zeros((10, N_NUM)), cat)
        assert np.all(np.isfinite(p))
        assert scorer.n_clamped == 2

    def test_predict_frame_matches_predict_proba_and_fills_nan(self, tmp_path):
        ckpt = tmp_path / "m.pt"
        torch.save(_kan().state_dict(), ckpt)
        scorer = Scorer.from_checkpoint(str(ckpt), NUM_COLS, IDX_COLS)
        num = np.random.randn(20, N_NUM)
        cat = np.random.randint(0, 3, (20, len(CARDS)))
        pdf = pd.DataFrame(np.hstack([num, cat]), columns=NUM_COLS + IDX_COLS)
        pdf.loc[0, "I1"] = np.nan
        p_frame = scorer.predict_frame(pdf)
        num_filled = num.copy()
        num_filled[0, 0] = 0.0
        assert np.allclose(p_frame, scorer.predict_proba(num_filled, cat), atol=1e-6)

    def test_column_count_mismatch_is_an_error(self, tmp_path):
        ckpt = tmp_path / "m.pt"
        torch.save(_kan().state_dict(), ckpt)
        with pytest.raises(ValueError):
            Scorer.from_checkpoint(str(ckpt), NUM_COLS, IDX_COLS[:-1])


class TestManifest:
    def _setup(self, tmp_path):
        ckpt = tmp_path / "m.pt"
        torch.save(_kan().state_dict(), ckpt)
        stats = tmp_path / "scaler_stats.json"
        stats.write_text(json.dumps({"I4": {"mean": 0.1, "std": 2.0}}))
        write_manifest(str(ckpt), NUM_COLS, IDX_COLS, scaler_stats_path=str(stats),
                       extra={"seed": 42, "test_auc": 0.785})
        return ckpt, stats

    def test_manifest_describes_the_checkpoint(self, tmp_path):
        ckpt, _ = self._setup(tmp_path)
        m = load_manifest(str(ckpt))
        assert m["encoder"] == "kan-bspline" and m["grid_size"] == 6
        assert m["numerical_cols"] == NUM_COLS and m["cat_cardinalities"] == CARDS
        assert m["extra"]["seed"] == 42
        assert len(m["checkpoint_sha256"]) == 64 and m["kanrec_version"]

    def test_check_passes_on_untouched_files(self, tmp_path):
        ckpt, stats = self._setup(tmp_path)
        check_manifest(str(ckpt), scaler_stats_path=str(stats), numerical_cols=NUM_COLS)

    def test_changed_scaler_stats_are_rejected(self, tmp_path):
        """Servir con estadisticos distintos de los de entrenamiento es
        exactamente el fallo que el manifiesto existe para impedir."""
        ckpt, stats = self._setup(tmp_path)
        stats.write_text(json.dumps({"I4": {"mean": 0.0, "std": 1.0}}))
        with pytest.raises(RuntimeError, match="scaler_stats"):
            check_manifest(str(ckpt), scaler_stats_path=str(stats))

    def test_changed_checkpoint_is_rejected(self, tmp_path):
        ckpt, stats = self._setup(tmp_path)
        torch.save(_kan(seed=1).state_dict(), ckpt)      # otro modelo, mismo nombre
        with pytest.raises(RuntimeError, match="checkpoint"):
            check_manifest(str(ckpt))

    def test_reordered_columns_are_rejected(self, tmp_path):
        ckpt, _ = self._setup(tmp_path)
        with pytest.raises(RuntimeError, match="columnas"):
            check_manifest(str(ckpt), numerical_cols=list(reversed(NUM_COLS)))

    def test_wrong_column_count_fails_at_write_time(self, tmp_path):
        ckpt = tmp_path / "m.pt"
        torch.save(_kan().state_dict(), ckpt)
        with pytest.raises(ValueError):
            write_manifest(str(ckpt), NUM_COLS[:-1], IDX_COLS)
