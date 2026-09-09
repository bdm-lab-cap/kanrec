"""
SymbolicExtractor — post-training pipeline that converts learned KAN splines
into closed-form scoring rules.

Steps:
  1. Prune edges by L1 norm (+ optional ShapKAN importance)
  2. Fit each surviving curve to an operator from the library
  3. Persist results to MongoDB via MongoSymbolicStore
"""
import numpy as np
from typing import Callable, Optional
from scipy.optimize import curve_fit

from .mongo_store import MongoSymbolicStore

# Operator library: each function takes (x, a, b)
OPERATOR_LIBRARY: dict[str, Callable] = {
    "log":     lambda x, a, b: a * np.log(np.abs(x) + 1) + b,
    # np.clip en exp/sigmoid: sin el, valores de |x| grandes desbordan
    # (RuntimeWarning: overflow encountered in exp). No cambia el ajuste --
    # exp(700) ya es inf en float64 -- pero evita ruido en la salida y
    # NaN silenciosos dentro de curve_fit. El clip a +-500 es holgado:
    # exp(500) ~ 1e217, muy por encima de cualquier valor util aqui.
    "exp":     lambda x, a, b: a * np.exp(np.clip(x, -500, 500)) + b,
    "square":  lambda x, a, b: a * x ** 2 + b,
    "sqrt":    lambda x, a, b: a * np.sqrt(np.abs(x)) + b,
    "inverse": lambda x, a, b: a / (np.abs(x) + 1e-6) + b,
    "sigmoid": lambda x, a, b: a / (1 + np.exp(np.clip(-x, -500, 500))) + b,
    "linear":  lambda x, a, b: a * x + b,
}

FORMULA_TEMPLATES = {
    "log":     "{a}·log(|x|+1) + {b}",
    "exp":     "{a}·exp(x) + {b}",
    "square":  "{a}·x² + {b}",
    "sqrt":    "{a}·√|x| + {b}",
    "inverse": "{a}/(|x|+ε) + {b}",
    "sigmoid": "{a}·σ(x) + {b}",
    "linear":  "{a}·x + {b}",
}


class SymbolicExtractor:
    """
    Args:
        model:            Trained KANRecModel.
        r2_threshold:     Minimum R² to accept a symbolic operator (default 0.95).
        l1_percentile:    Bottom percentile of L1 norms to prune (default 20).
        mongo_uri:        MongoDB connection string.
    """

    def __init__(
        self,
        model,
        r2_threshold: float = 0.95,
        l1_percentile: float = 20,
        mongo_uri: str = "mongodb://localhost:27017",
    ):
        self.model         = model
        self.r2_threshold  = r2_threshold
        self.l1_percentile = l1_percentile
        self.store         = MongoSymbolicStore(uri=mongo_uri)
        self.results: dict[int, dict] = {}

    # ── Pruning ───────────────────────────────────────────────────────────────

    def prune_fields(self) -> list[int]:
        """Returns field indices whose L1 norm exceeds the pruning threshold."""
        norms     = self.model.numerical_encoder.get_edge_norms()
        threshold = np.percentile(norms, self.l1_percentile)
        surviving = [j for j, n in enumerate(norms) if n >= threshold]
        print(f"L1 pruning: {len(surviving)}/{len(norms)} fields survive "
              f"(threshold={threshold:.4f})")
        return surviving

    # ── Symbolic fitting ──────────────────────────────────────────────────────

    def fit_field(self, field_idx: int, n_grid: int = 500) -> dict:
        """Fits the best operator from OPERATOR_LIBRARY to curve φⱼ."""
        x_grid, y_curves = self.model.numerical_encoder.get_spline_curves(
            field_idx, n_points=n_grid
        )
        x_np = x_grid.numpy()
        y_np = y_curves[:, 0].numpy()

        best = {"operator": None, "r2": -1.0, "params": None,
                "formula": "?", "accepted": False}

        for name, fn in OPERATOR_LIBRARY.items():
            try:
                params, _ = curve_fit(fn, x_np, y_np, maxfev=5000)
                y_pred = fn(x_np, *params)
                ss_res = np.sum((y_np - y_pred) ** 2)
                ss_tot = np.sum((y_np - y_np.mean()) ** 2) + 1e-10
                r2     = float(1 - ss_res / ss_tot)
                if r2 > best["r2"]:
                    a, b = round(float(params[0]), 4), round(float(params[1]), 4)
                    best = {
                        "operator": name,
                        "r2":       r2,
                        "params":   params,
                        "formula":  FORMULA_TEMPLATES[name].format(a=a, b=b),
                        "accepted": r2 >= self.r2_threshold,
                    }
            except Exception:
                continue

        return best

    def _is_monotone(self, field_idx: int) -> bool:
        """Returns True if φⱼ is >95% monotone (increasing or decreasing)."""
        import torch
        _, y = self.model.numerical_encoder.get_spline_curves(field_idx)
        diffs    = torch.diff(y[:, 0])
        dominant = max((diffs > 0).sum().item(), (diffs < 0).sum().item())
        return (len(diffs) - dominant) / len(diffs) < 0.05

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def extract_and_persist(
        self,
        dataset: str,
        seed: int,
        field_names: list[str],
        run_id: str = "no-mlflow",
        feature_selection: Optional[dict] = None,
    ) -> dict:
        """
        Full pipeline: prune → fit → persist to MongoDB.

        Args:
            dataset:           'criteo' or 'avazu'
            seed:              training seed
            field_names:       names of selected numerical fields
            run_id:            MLflow run ID for traceability
            feature_selection: dict with chisq/spearman importance scores
        Returns:
            dict of {field_idx: result_dict}
        """
        fs = feature_selection or {}
        surviving = self.prune_fields()

        print("\n── Symbolic fitting ─────────────────────────────────────────")
        for j in surviving:
            result = self.fit_field(j)
            self.results[j] = result

            field_name = field_names[j] if j < len(field_names) else f"F{j}"
            is_mono    = self._is_monotone(j)
            status     = "✓" if result["accepted"] else "✗"
            params     = result["params"]

            doc_id = self.store.save_result(
                run_id=run_id,
                dataset=dataset,
                seed=seed,
                field_name=field_name,
                field_idx=j,
                operator=result["operator"],
                param_a=float(params[0]) if params is not None else None,
                param_b=float(params[1]) if params is not None else None,
                formula_str=result["formula"],
                r2=result["r2"],
                gap_rmse=0.0,        # updated later by FaithfulnessEvaluator
                is_monotone=is_mono,
                is_accepted=result["accepted"],
                importance_chisq_rank=fs.get(field_name, {}).get("chisq_rank"),
                importance_spearman=fs.get(field_name, {}).get("spearman"),
            )
            print(f"  {status} {field_name}: {result['operator']:8s} "
                  f"R²={result['r2']:.4f}  mono={is_mono}  "
                  f"→ MongoDB {doc_id[:8]}…")

        return self.results

    def print_scoring_formula(self) -> str:
        """Prints the aggregated scoring formula from accepted operators."""
        accepted = {j: r for j, r in self.results.items() if r.get("accepted")}
        if not accepted:
            return "No accepted operators found."
        lines = ["ŷ ≈ f(" + ", ".join(f"F{j}" for j in sorted(accepted)) + ") where:"]
        for j, r in sorted(accepted.items()):
            lines.append(f"  φ_F{j}(x) ≈ {r['formula']}")
        formula = "\n".join(lines)
        print("\n── Scoring formula ──────────────────────────────────────────")
        print(formula)
        return formula

    def close(self):
        self.store.close()
