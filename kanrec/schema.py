"""
Esquema único de los documentos de resultados simbólicos.

Por qué existe este módulo
--------------------------
El proyecto tenía **dos escritores con esquemas incompatibles** para la misma
colección de Atlas:

  - `kanrec/mongo_store.py` escribía y agregaba con `field_name` / `is_accepted`
  - `experiments/mongodb_import.py` insertaba con `field` / `accepted`

Los documentos del segundo eran **invisibles** para el informe de estabilidad
del primero: no pasaban el `$match` sobre `is_accepted` y agrupaban por `null`
al no tener `field_name`. El pipeline de agregación devolvía vacío o resultados
parciales, lo que explica que la colección mostrara 8 documentos cuando tres
semillas por diez campos deberían dar treinta.

La solución es que el esquema se defina **una sola vez**, aquí, y que ambos
escritores construyan sus documentos con `SymbolicResult.to_document()`. Un
cambio de nombre de campo pasa a ser imposible de desincronizar: o compila en
los dos sitios, o en ninguno.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class SymbolicResult:
    """
    Un resultado de extracción simbólica: la fórmula ajustada a φⱼ para un
    campo, una semilla y una ejecución concretos.

    Los nombres de los atributos son exactamente los nombres de los campos en
    MongoDB. La clave natural del documento es
    `(run_id, dataset, seed, field_name)`, sobre la que se define un índice
    único para que una reimportación actualice en lugar de duplicar.
    """

    # Identificación (clave natural)
    run_id: str
    dataset: str
    seed: int
    field_name: str

    # Resultado del ajuste
    operator: Optional[str]
    formula_str: str
    r2: float
    is_accepted: bool

    # Contexto
    field_idx: Optional[int] = None
    encoder: str = "kan-bspline"
    param_a: Optional[float] = None
    param_b: Optional[float] = None

    # Métricas sobre las dimensiones del embedding (ajuste sobre las 16, no
    # solo la dimensión 0). Opcionales por compatibilidad con resultados
    # anteriores a ese cambio.
    r2_mean: Optional[float] = None
    r2_std: Optional[float] = None
    r2_min: Optional[float] = None
    operator_agreement: Optional[float] = None
    n_dims_fitted: Optional[int] = None

    # Fidelidad y coherencia estructural
    curve_rel_error: Optional[float] = None
    gap_rmse: Optional[float] = None
    is_monotone: Optional[bool] = None
    violation_rate: Optional[float] = None

    timestamp: datetime = dc_field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_document(self) -> dict[str, Any]:
        """Documento listo para insertar, omitiendo los campos sin valor."""
        doc = {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "seed": self.seed,
            "field_name": self.field_name,
            "encoder": self.encoder,
            "operator": self.operator,
            "formula_str": self.formula_str,
            "r2": self.r2,
            "is_accepted": self.is_accepted,
            "timestamp": self.timestamp,
        }
        opcionales = {
            "field_idx": self.field_idx,
            "param_a": self.param_a,
            "param_b": self.param_b,
            "r2_mean": self.r2_mean,
            "r2_std": self.r2_std,
            "r2_min": self.r2_min,
            "operator_agreement": self.operator_agreement,
            "n_dims_fitted": self.n_dims_fitted,
            "curve_rel_error": self.curve_rel_error,
            "gap_rmse": self.gap_rmse,
            "is_monotone": self.is_monotone,
            "violation_rate": self.violation_rate,
        }
        doc.update({k: v for k, v in opcionales.items() if v is not None})
        return doc

    @classmethod
    def from_fit(cls, run_id: str, dataset: str, seed: int, field_name: str,
                 fit: dict, field_idx: int | None = None,
                 encoder: str = "kan-bspline") -> "SymbolicResult":
        """
        Construye el resultado desde la salida de `fit_field`, que devuelve
        operador dominante, parámetros y métricas sobre las 16 dimensiones.
        """
        params = fit.get("params") or [None, None]
        return cls(
            run_id=run_id,
            dataset=dataset,
            seed=seed,
            field_name=field_name,
            field_idx=field_idx,
            encoder=encoder,
            operator=fit.get("operator"),
            formula_str=fit.get("formula", "?"),
            r2=float(fit.get("r2", fit.get("r2_mean", 0.0))),
            is_accepted=bool(fit.get("accepted", False)),
            param_a=params[0],
            param_b=params[1] if len(params) > 1 else None,
            r2_mean=fit.get("r2_mean"),
            r2_std=fit.get("r2_std"),
            r2_min=fit.get("r2_min"),
            operator_agreement=fit.get("operator_agreement"),
            n_dims_fitted=fit.get("n_dims_fitted"),
        )


#: Clave natural del documento. El índice único sobre estos campos evita
#: duplicados al reimportar los mismos resultados.
NATURAL_KEY = ("run_id", "dataset", "seed", "field_name")
