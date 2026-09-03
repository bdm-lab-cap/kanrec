"""
FaithfulnessEvaluator — measures how well the symbolic formula approximates
the trained KAN model.

Metrics:
  - Gap RMSE:            ‖ŷ_KAN − ŷ_symbolic‖ on test set (target < 0.01)
  - Stability:           % seeds recovering the same operator (via MongoDB query)
  - Monotonicity audit:  violation rate per field
"""
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from .symbolic import OPERATOR_LIBRARY
from .mongo_store import MongoSymbolicStore


class FaithfulnessEvaluator:
    """
    Args:
        model:      Trained KANRecModel.
        extractor:  SymbolicExtractor with results already populated.
        test_loader: PyTorch DataLoader for the test set.
        mongo_uri:  MongoDB URI for stability queries.
    """

    def __init__(self, model, extractor, test_loader,
                 mongo_uri: str = "mongodb://localhost:27017"):
        self.model       = model
        self.extractor   = extractor
        self.test_loader = test_loader
        self.store       = MongoSymbolicStore(uri=mongo_uri)

    def compute_gap_rmse(self) -> float:
        """
        Computes ‖ŷ_KAN − ŷ_symbolic‖ on the test set.
        A gap < 0.01 indicates the formula faithfully represents the model.
        """
        kan_preds, sym_preds = [], []
        self.model.eval()

        with torch.no_grad():
            for x_num, x_cat, _ in self.test_loader:
                y_kan = self.model(x_num, x_cat).squeeze().numpy()
                y_sym = self._apply_symbolic(x_num.numpy())
                kan_preds.extend(y_kan)
                sym_preds.extend(y_sym)

        gap = float(np.sqrt(mean_squared_error(kan_preds, sym_preds)))
        print(f"Gap RMSE (KAN vs symbolic): {gap:.6f}  {'✓ OK' if gap < 0.01 else '⚠ HIGH'}")
        return gap

    def _apply_symbolic(self, x_num: np.ndarray) -> np.ndarray:
        """Evaluates the symbolic formula on a batch of numerical inputs."""
        result = np.zeros(len(x_num))
        for j, r in self.extractor.results.items():
            if not r.get("accepted"):
                continue
            op, params = r["operator"], r["params"]
            x_field    = x_num[:, j]
            result    += OPERATOR_LIBRARY[op](x_field, *params)
        return 1 / (1 + np.exp(-result))   # sigmoid to get probability

    def stability_report(self, dataset: str, seeds: list[int]) -> list[dict]:
        """
        Queries MongoDB for operator stability across seeds.
        Returns a list of {field, operator, seeds_count, stability_pct, avg_r2}.
        """
        report = self.store.stability_report(dataset, seeds)
        print("\n── Stability report ─────────────────────────────────────────")
        print(f"{'Field':<8} {'Operator':<10} {'Seeds':<7} {'Stability':<12} {'Avg R²'}")
        print("-" * 52)
        for r in report:
            print(f"{r['field']:<8} {r['operator']:<10} "
                  f"{r['seeds_count']:<7} {r['stability_pct']:.0%}{'':8} {r['avg_r2']:.4f}")
        return report

    def monotonicity_audit(
        self, expected_direction: dict[int, str]
    ) -> dict[int, float]:
        """
        For each field in expected_direction, computes the fraction of
        grid points where the learned curve violates the expected direction.

        Args:
            expected_direction: {field_idx: 'increasing' | 'decreasing'}

        Returns:
            {field_idx: violation_rate}
        """
        print("\n── Monotonicity audit ───────────────────────────────────────")
        violations = {}
        for j, direction in expected_direction.items():
            x_grid, y_curves = self.model.numerical_encoder.get_spline_curves(j)
            y  = y_curves[:, 0].numpy()
            diffs = np.diff(y)
            if direction == "increasing":
                n_viol = int(np.sum(diffs < -1e-4))
            else:
                n_viol = int(np.sum(diffs >  1e-4))
            rate = n_viol / len(diffs)
            violations[j] = rate
            status = "✓ OK" if rate < 0.05 else "⚠ VIOLATES"
            print(f"  F{j} ({direction}): violation_rate={rate:.2%}  {status}")
        return violations

    def update_gap_in_mongo(
        self, dataset: str, seed: int, gap_rmse: float
    ):
        """Updates gap_rmse in all MongoDB documents for this run."""
        self.store.col.update_many(
            {"dataset": dataset, "seed": seed},
            {"$set": {"gap_rmse": gap_rmse}},
        )

    def close(self):
        self.store.close()
