"""
FaithfulnessEvaluator — measures how well the symbolic formula approximates
the trained KAN model.

Metrics:
  - Gap RMSE:            ‖ŷ_KAN − ŷ_symbolic‖ on test set (target < 0.01)
  - Stability:           % seeds recovering the same operator (via MongoDB query)
  - Monotonicity audit:  violation rate per field
"""
from collections import Counter

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
                # forward() ahora devuelve LOGITS; sigmoid para comparar en
                # el mismo espacio de probabilidad que _apply_symbolic.
                y_kan = torch.sigmoid(self.model(x_num, x_cat).squeeze()).numpy()
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
        self,
        field_indices: list[int] | None = None,
        expected_direction: dict[int, str] | None = None,
        field_names: list[str] | None = None,
        tol: float = 1e-4,
    ) -> dict[int, dict]:
        """
        Audita la monotonia de las curvas phi_j aprendidas.

        Dos modos:

        1. **Descriptivo** (por defecto, sin `expected_direction`): mide que
           fraccion de los tramos de cada curva va en la direccion MINORITARIA.
           Un valor cercano a 0 significa curva monotona; cercano a 0.5,
           curva sin direccion dominante. Es el modo apropiado para Criteo,
           donde las variables I1..I13 son anonimas y no existe una direccion
           "esperada" que se pueda afirmar sin inventarla.

        2. **Contra expectativa** (pasando `expected_direction`): mide la tasa
           de violacion respecto a una direccion de dominio conocida. Solo
           tiene sentido cuando la semantica de la variable es conocida
           (p.ej. "a mayor precio, menor score").

        Se promedia sobre TODAS las dimensiones del embedding, no solo la 0:
        afirmar que una curva es monotona mirando una sola de sus 16
        proyecciones no lo demuestra (mismo criterio que en fit_field).

        Returns:
            {field_idx: {"violation_rate", "direction", "per_dim_rates"}}
        """
        print("\n── Auditoria de monotonia ───────────────────────────────────")
        if field_indices is None:
            field_indices = list(range(self.model.numerical_encoder.num_fields))

        results = {}
        for j in field_indices:
            x_grid, y_curves = self.model.numerical_encoder.get_spline_curves(j)
            per_dim_rates, per_dim_dirs = [], []

            for d in range(y_curves.shape[1]):
                diffs = np.diff(y_curves[:, d].numpy())
                n_up = int(np.sum(diffs > tol))
                n_down = int(np.sum(diffs < -tol))
                n_moves = n_up + n_down
                if n_moves == 0:            # curva plana
                    per_dim_rates.append(0.0)
                    per_dim_dirs.append("flat")
                    continue
                if expected_direction and j in expected_direction:
                    want = expected_direction[j]
                    rate = (n_down if want == "increasing" else n_up) / len(diffs)
                    per_dim_dirs.append(want)
                else:
                    # descriptivo: la direccion minoritaria es la "violacion"
                    rate = min(n_up, n_down) / len(diffs)
                    per_dim_dirs.append("increasing" if n_up >= n_down else "decreasing")
                per_dim_rates.append(rate)

            rate_mean = float(np.mean(per_dim_rates))
            dominant = Counter(per_dim_dirs).most_common(1)[0][0]
            name = field_names[j] if field_names else f"I{j+1}"
            results[j] = {"field": name, "violation_rate": rate_mean,
                          "violation_rate_std": float(np.std(per_dim_rates)),
                          "direction": dominant, "per_dim_rates": per_dim_rates}
            status = "monotona" if rate_mean < 0.05 else (
                     "casi monotona" if rate_mean < 0.15 else "NO monotona")
            print(f"  {name:>5} ({dominant:>10}): violacion={rate_mean:.2%} "
                  f"±{np.std(per_dim_rates):.2%}  {status}")
        return results

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
