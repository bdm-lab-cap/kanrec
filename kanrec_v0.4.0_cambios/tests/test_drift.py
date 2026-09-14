"""
Tests de kanrec.drift.

Todo es numpy puro salvo la clase que toca el encoder, que se salta si
torch no está instalado. Cada test de comportamiento está escrito para
distinguir "detecta deriva" de "devuelve un número": se comprueba tanto el
caso sin deriva (debe callar) como el caso con deriva (debe avisar).
"""
import json

import numpy as np
import pytest

from kanrec.drift import (
    COVERAGE_ALERT, PSI_ALERT, PSI_WARNING,
    DriftReference, drift_report, psi, psi_level, range_coverage, summarize,
)

RNG = np.random.default_rng(0)
FIELDS = ["I1", "I2", "I3"]


def _ref_data(n=20_000):
    return np.column_stack([
        RNG.normal(0, 1, n),          # I1: normal estandar
        RNG.exponential(1.0, n),      # I2: cola pesada
        np.where(RNG.random(n) < 0.6, 0.0, RNG.normal(2, 0.5, n)),  # I3: 60% ceros
    ])


class TestPSI:
    def test_identical_distributions_give_zero(self):
        p = np.array([0.2, 0.3, 0.5])
        assert psi(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_is_symmetric_and_positive(self):
        p, q = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
        assert psi(p, q) > 0
        assert psi(p, q) == pytest.approx(psi(q, p))

    def test_empty_bins_do_not_produce_inf(self):
        assert np.isfinite(psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]))

    def test_levels_follow_standard_thresholds(self):
        assert psi_level(PSI_WARNING - 1e-9) == "ok"
        assert psi_level(PSI_WARNING) == "warning"
        assert psi_level(PSI_ALERT) == "alert"


class TestRangeCoverage:
    def test_all_inside(self):
        assert range_coverage(np.linspace(-1, 1, 50), -1, 1) == 1.0

    def test_nan_counts_as_outside(self):
        assert range_coverage(np.array([0.0, np.nan]), -1, 1) == 0.5

    def test_empty_is_nan(self):
        assert np.isnan(range_coverage(np.array([]), -1, 1))


class TestDriftReference:
    def test_expected_frequencies_sum_to_one(self):
        ref = DriftReference.fit(_ref_data(), FIELDS, n_bins=10)
        for exp in ref.expected:
            assert sum(exp) == pytest.approx(1.0)

    def test_edges_are_open_at_both_ends(self):
        ref = DriftReference.fit(_ref_data(), FIELDS)
        for e in ref.edges:
            assert e[0] == -np.inf and e[-1] == np.inf

    def test_repeated_quantiles_are_deduplicated(self):
        """I3 tiene 60% de ceros: varios cuantiles coinciden en 0 y no
        pueden convertirse en bins vacios de ancho cero."""
        ref = DriftReference.fit(_ref_data(), FIELDS, n_bins=10)
        edges_i3 = ref.edges[2]
        assert len(edges_i3) == len(set(edges_i3))
        assert len(edges_i3) < 11 + 1          # menos bins efectivos que n_bins

    def test_json_round_trip_preserves_inf(self):
        ref = DriftReference.fit(_ref_data(), FIELDS, calibrated_ranges=[(-3, 3)] * 3)
        again = DriftReference.from_json(ref.to_json())
        assert again.field_names == ref.field_names
        assert again.edges == ref.edges
        assert again.expected == ref.expected
        assert again.calibrated_ranges == ref.calibrated_ranges
        json.loads(ref.to_json())                # es JSON valido de verdad

    def test_column_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            DriftReference.fit(_ref_data(), FIELDS[:2])


class TestDriftReport:
    def test_no_drift_stays_quiet(self):
        """Un lote de la misma distribucion que train no debe alertar."""
        ref = DriftReference.fit(_ref_data(), FIELDS, calibrated_ranges=[(-4, 4), (0, 8), (-1, 4)])
        rows = drift_report(_ref_data(5_000), ref)
        assert [r["psi_level"] for r in rows] == ["ok"] * 3
        assert all(not r["coverage_alert"] for r in rows)
        assert all(r["coverage"] >= COVERAGE_ALERT for r in rows)

    def test_shifted_distribution_is_detected(self):
        """Desplazar I1 dos desviaciones debe disparar PSI y cobertura, y
        SOLO en I1: las senales son por campo."""
        ref = DriftReference.fit(_ref_data(), FIELDS, calibrated_ranges=[(-3, 3), (0, 8), (-1, 4)])
        batch = _ref_data(5_000)
        batch[:, 0] += 2.0
        rows = drift_report(batch, ref)
        assert rows[0]["psi_level"] == "alert"
        assert rows[0]["coverage_alert"]
        assert rows[0]["share_above"] > rows[0]["share_below"]
        assert rows[1]["psi_level"] == "ok" and rows[2]["psi_level"] == "ok"

    def test_out_of_range_without_distribution_change(self):
        """Caso que SOLO detecta la cobertura: la distribucion del lote es
        la misma, pero el modelo fue calibrado sobre un rango mas estrecho
        (p.ej. un checkpoint antiguo). PSI calla, cobertura avisa."""
        ref = DriftReference.fit(_ref_data(), FIELDS)
        narrow = [(-0.5, 0.5), (0, 8), (-1, 4)]
        rows = drift_report(_ref_data(5_000), ref, calibrated=narrow)
        assert rows[0]["psi_level"] == "ok"
        assert rows[0]["coverage_alert"]
        assert rows[0]["coverage"] == pytest.approx(0.383, abs=0.03)   # P(|N(0,1)| < 0.5)

    def test_no_ranges_anywhere_reports_nan_not_alert(self):
        ref = DriftReference.fit(_ref_data(), FIELDS)
        rows = drift_report(_ref_data(1_000), ref)
        assert all(np.isnan(r["coverage"]) for r in rows)
        assert all(not r["coverage_alert"] for r in rows)

    def test_summary_names_the_offending_fields(self):
        ref = DriftReference.fit(_ref_data(), FIELDS, calibrated_ranges=[(-3, 3), (0, 8), (-1, 4)])
        batch = _ref_data(5_000)
        batch[:, 2] += 5.0
        s = summarize(drift_report(batch, ref))
        assert s["worst_psi_field"] == "I3"
        assert s["fields_psi_alert"] == ["I3"]
        assert s["fields_coverage_alert"] == ["I3"]


class TestCalibratedRangesFromEncoder:
    def test_ranges_match_encoder_field_range(self):
        torch = pytest.importorskip("torch")
        from kanrec.drift import calibrated_ranges
        from kanrec.encoder import KANNumericalEncoder
        from kanrec.vectorized import VectorizedKANEncoder

        torch.manual_seed(0)
        enc = KANNumericalEncoder(num_fields=4, embedding_dim=8, grid_size=10)
        enc.calibrate(torch.randn(2_000, 4) * torch.tensor([1.0, 2.0, 0.5, 3.0]))
        ranges = calibrated_ranges(enc)
        assert len(ranges) == 4
        assert all(lo < hi for lo, hi in ranges)
        # calibrar sobre una escala mayor debe dar un rango mayor
        assert (ranges[3][1] - ranges[3][0]) > (ranges[2][1] - ranges[2][0])
        # la version vectorizada expone el mismo grid
        vec = VectorizedKANEncoder.from_field_kans(enc)
        assert calibrated_ranges(vec) == pytest.approx(ranges, abs=1e-6)
