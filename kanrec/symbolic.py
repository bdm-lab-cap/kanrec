"""
SymbolicExtractor — post-training pipeline that converts learned KAN splines
into closed-form scoring rules.

Steps:
  1. Prune edges by L1 norm (+ optional ShapKAN importance)
  2. Fit each surviving curve to an operator from the library
  3. Persist results to MongoDB via MongoSymbolicStore
"""
from collections import Counter

import numpy as np
from typing import Callable, Optional
from scipy.optimize import curve_fit

from .mongo_store import MongoSymbolicStore

# Operator library: each function takes (x, a, b)
OPERATOR_LIBRARY: dict[str, Callable] = {
    "log":     lambda x, a, b: a * np.log(np.abs(x) + 1) + b,
    # np.clip acota el exponente para que exp(x) no desborde por si solo.
    # NO basta para eliminar todos los avisos: el desbordamiento real viene
    # del producto a * exp(x) durante la optimizacion, y 'a' no esta acotado
    # (ver el np.errstate en fit_field). El clip es una defensa parcial.
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

    def _fit_curve(self, x_np, y_np) -> dict:
        """Ajusta la libreria de operadores a UNA curva y devuelve el mejor."""
        best = {"operator": None, "r2": -1.0, "params": None, "formula": "?"}

        for name, fn in OPERATOR_LIBRARY.items():
            try:
                # curve_fit explora valores extremos del parametro 'a', donde
                # a * exp(x) desborda float64. NumPy devuelve inf/NaN, el
                # optimizador los descarta y sigue: el aviso es ruido, no un
                # error. Acotar el exponente no lo evita, porque el
                # desbordamiento viene del producto y 'a' no esta acotado.
                # Se silencia el aviso y se descarta todo ajuste no finito.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    params, _ = curve_fit(fn, x_np, y_np, maxfev=5000)
                    y_pred = fn(x_np, *params)
                if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(params)):
                    continue
                ss_tot = np.sum((y_np - y_np.mean()) ** 2)
                if ss_tot < 1e-12:          # curva plana: R2 no definido
                    continue
                r2 = float(1 - np.sum((y_np - y_pred) ** 2) / (ss_tot + 1e-10))
                if r2 > best["r2"]:
                    a, b = round(float(params[0]), 4), round(float(params[1]), 4)
                    best = {
                        "operator": name,
                        "r2":       r2,
                        "params":   [float(params[0]), float(params[1])],
                        "formula":  FORMULA_TEMPLATES[name].format(a=a, b=b),
                    }
            except Exception:
                continue
        return best

    def fit_field(self, field_idx: int, n_grid: int = 500) -> dict:
        """
        Ajusta la libreria de operadores a las `embedding_dim` dimensiones de
        phi_j, evaluadas en el rango calibrado del campo.

        Ajustar solo la dimension 0 y reportarlo como si describiera el campo
        entero asume, sin demostrarlo, que esa proyeccion es representativa.
        Sobre una senal sinusoidal conocida se comprobo que no lo es: la
        dimension 0 elegia un operador distinto del dominante entre las 16.
        Por eso se ajustan todas y se reporta:

          operator            operador dominante (el que gana en mas dimensiones)
          r2_mean/std/min     dispersion del ajuste entre dimensiones
          operator_agreement  fraccion de dimensiones que eligen el dominante

        `r2` se mantiene como alias de `r2_mean` por compatibilidad, y
        `dim0_operator` permite comparar con resultados anteriores al cambio.
        """
        x_grid, y_curves = self.model.numerical_encoder.get_spline_curves(
            field_idx, n_points=n_grid
        )
        x_np = x_grid.numpy()

        per_dim = []
        for d in range(y_curves.shape[1]):
            fit = self._fit_curve(x_np, y_curves[:, d].numpy())
            if fit["operator"] is not None:
                per_dim.append(fit)

        if not per_dim:
            return {"operator": None, "r2": -1.0, "params": None,
                    "formula": "?", "accepted": False, "n_dims_fitted": 0}

        ops = [f["operator"] for f in per_dim]
        dominant, n_dom = Counter(ops).most_common(1)[0]
        r2s = np.array([f["r2"] for f in per_dim], dtype=float)

        # Formula representativa: el mejor ajuste ENTRE las dimensiones que
        # eligieron el operador dominante.
        representative = max((f for f in per_dim if f["operator"] == dominant),
                             key=lambda f: f["r2"])

        return {
            "operator":           dominant,
            "r2":                 float(r2s.mean()),
            "params":             representative["params"],
            "formula":            representative["formula"],
            "accepted":           bool(r2s.mean() >= self.r2_threshold),
            "n_dims_fitted":      len(per_dim),
            "operator_agreement": n_dom / len(per_dim),
            "r2_mean":            float(r2s.mean()),
            "r2_std":             float(r2s.std()),
            "r2_min":             float(r2s.min()),
            "r2_max":             float(r2s.max()),
            "dim0_operator":      per_dim[0]["operator"],
            "dim0_r2":            per_dim[0]["r2"],
        }

    def monotonicity(self, field_idx: int, tol: float = 1e-4) -> dict:
        """
        Monotonia observada de phi_j, promediada sobre todas las dimensiones.

        Se reporta la fraccion de tramos que va en la direccion minoritaria:
        cerca de 0 significa curva monotona. No se compara contra una
        direccion "esperada" porque las variables de Criteo son anonimas y
        esa expectativa no puede afirmarse sin inventarla.
        """
        import torch

        _, curves = self.model.numerical_encoder.get_spline_curves(field_idx)
        rates, directions = [], []
        for d in range(curves.shape[1]):
            diffs = torch.diff(curves[:, d])
            n_up = int((diffs > tol).sum())
            n_down = int((diffs < -tol).sum())
            if n_up + n_down == 0:
                rates.append(0.0)
                directions.append("flat")
                continue
            rates.append(min(n_up, n_down) / len(diffs))
            directions.append("increasing" if n_up >= n_down else "decreasing")

        rate = float(np.mean(rates))
        return {
            "violation_rate": rate,
            "violation_std":  float(np.std(rates)),
            "direction":      Counter(directions).most_common(1)[0][0],
            "is_monotone":    rate < 0.05,
        }

    def _is_monotone(self, field_idx: int) -> bool:
        """Compatibilidad: True si phi_j es monotona (violacion < 5 %)."""
        return self.monotonicity(field_idx)["is_monotone"]

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
