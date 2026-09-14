"""
Detección de deriva sobre las entradas numéricas del modelo en servicio.

Dos señales, ambas calculadas por campo y por lote del stream:

1. **Cobertura del rango calibrado.** Cada φⱼ del encoder KAN tiene un
   grid de nudos ajustado a la distribución de entrenamiento (ver
   ``KANNumericalEncoder.calibrate``). Fuera de ese rango todas las bases
   B-spline valen cero y φⱼ degenera en la ruta base ``w·SiLU(x)``: el
   modelo sigue devolviendo un número, pero ya no es la curva que se
   auditó ni la fórmula que se extrajo. La cobertura es la fracción de
   valores del lote que caen dentro del rango calibrado. Es una señal que
   sólo existe porque el encoder es interpretable: una capa densa no tiene
   un rango "válido" que vigilar.

2. **PSI (Population Stability Index).** Divergencia entre la distribución
   de referencia (train) y la del lote, sobre bins de cuantiles fijados en
   la referencia. Umbrales habituales en scoring: < 0,10 estable,
   0,10–0,25 vigilar, > 0,25 deriva.

Diseño
------
- numpy puro. torch sólo se toca en ``calibrated_ranges``, que importa el
  encoder de forma perezosa, para que el módulo sea usable desde un
  notebook de Spark sin cargar el modelo.
- La referencia (bordes de bins + frecuencias esperadas) se calcula una
  vez sobre train y se persiste como JSON (``DriftReference``), de modo
  que el procesamiento del stream no relee train en cada ejecución.
- La tercera señal que se evaluó y descartó en la memoria (deriva de la
  forma funcional de φⱼ reajustada sobre el lote) no está aquí a propósito:
  las dimensiones del embedding no son identificables entre entrenamientos
  y la medida no discrimina.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# Umbrales estándar de PSI en modelos de scoring.
PSI_WARNING = 0.10
PSI_ALERT = 0.25
# Por debajo de esta cobertura, una parte relevante del lote se está
# evaluando fuera de la curva auditada.
COVERAGE_ALERT = 0.99


# ── Referencia (se calcula una vez sobre train) ──────────────────────────────

@dataclass
class DriftReference:
    """
    Bins de cuantiles y frecuencias esperadas por campo, ajustados sobre la
    distribución de entrenamiento. Serializable a JSON.

    Attributes:
        field_names: nombres de los campos numéricos, en el orden del modelo.
        edges:       bordes de bins por campo (n_bins + 1 valores). Los
                     extremos son -inf/+inf para que ningún valor quede fuera.
        expected:    frecuencia relativa de train en cada bin (suma 1).
        n_ref:       número de filas de referencia usadas.
    """
    field_names: list[str]
    edges: list[list[float]]
    expected: list[list[float]]
    n_ref: int = 0
    calibrated_ranges: list[list[float]] = field(default_factory=list)

    # ── construcción ──
    @classmethod
    def fit(cls, x_ref: np.ndarray, field_names: list[str], n_bins: int = 10,
            calibrated_ranges: list[tuple[float, float]] | None = None) -> "DriftReference":
        """
        Ajusta bins de cuantiles sobre la referencia.

        Los bordes se toman de los cuantiles de train, no equiespaciados:
        con colas tan pesadas como las de Criteo, bins uniformes dejarían
        casi todo el volumen en uno o dos bins y el PSI sería ciego.
        Cuantiles repetidos (campos con muchos ceros) se deduplican, así que
        el número efectivo de bins puede ser menor que ``n_bins``.
        """
        x_ref = _as_2d(x_ref)
        if x_ref.shape[1] != len(field_names):
            raise ValueError(
                f"x_ref tiene {x_ref.shape[1]} columnas y field_names {len(field_names)}"
            )
        edges_all, expected_all = [], []
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        for j in range(x_ref.shape[1]):
            col = x_ref[:, j]
            col = col[np.isfinite(col)]
            inner = np.unique(np.quantile(col, qs)) if col.size else np.array([])
            edges = np.concatenate([[-np.inf], inner, [np.inf]])
            counts, _ = np.histogram(col, bins=edges)
            expected = counts / max(counts.sum(), 1)
            edges_all.append([float(e) for e in edges])
            expected_all.append([float(p) for p in expected])
        ranges = [[float(lo), float(hi)] for lo, hi in (calibrated_ranges or [])]
        return cls(field_names=list(field_names), edges=edges_all,
                   expected=expected_all, n_ref=int(x_ref.shape[0]),
                   calibrated_ranges=ranges)

    # ── serialización ──
    def to_json(self) -> str:
        return json.dumps({
            "field_names": self.field_names,
            "edges": [[_inf_to_str(e) for e in row] for row in self.edges],
            "expected": self.expected,
            "n_ref": self.n_ref,
            "calibrated_ranges": self.calibrated_ranges,
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "DriftReference":
        d = json.loads(text)
        return cls(
            field_names=d["field_names"],
            edges=[[_str_to_inf(e) for e in row] for row in d["edges"]],
            expected=d["expected"],
            n_ref=int(d.get("n_ref", 0)),
            calibrated_ranges=d.get("calibrated_ranges", []),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "DriftReference":
        with open(path) as f:
            return cls.from_json(f.read())


# ── Señales ──────────────────────────────────────────────────────────────────

def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """
    Population Stability Index entre dos vectores de frecuencias relativas
    sobre los mismos bins.

        PSI = Σ (actual - expected) · ln(actual / expected)

    ``eps`` evita log(0) en bins vacíos; el valor es el habitual en la
    práctica de scoring y su efecto sobre el resultado es despreciable
    salvo en bins que la referencia también tiene vacíos.
    """
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_level(value: float) -> str:
    """Clasifica un PSI en 'ok' / 'warning' / 'alert' con los umbrales estándar."""
    if value >= PSI_ALERT:
        return "alert"
    if value >= PSI_WARNING:
        return "warning"
    return "ok"


def range_coverage(x: np.ndarray, low: float, high: float) -> float:
    """
    Fracción de valores de ``x`` dentro de ``[low, high]`` (inclusive).
    Valores no finitos cuentan como fuera de rango: un NaN tampoco cae en
    la curva calibrada.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return float("nan")
    inside = np.isfinite(x) & (x >= low) & (x <= high)
    return float(inside.mean())


def calibrated_ranges(encoder) -> list[tuple[float, float]]:
    """
    Rango calibrado ``(low, high)`` de cada campo de un
    ``KANNumericalEncoder`` (o de su versión vectorizada), leído del buffer
    ``grid`` del checkpoint. Importa torch de forma perezosa.
    """
    if hasattr(encoder, "field_range"):                       # KANNumericalEncoder
        return [tuple(encoder.field_range(j)) for j in range(encoder.num_fields)]
    if hasattr(encoder, "grid") and hasattr(encoder, "spline_order"):  # VectorizedKANEncoder
        k = encoder.spline_order
        out = []
        for j in range(encoder.num_fields):
            core = encoder.grid[j, k:-k] if k > 0 else encoder.grid[j]
            out.append((float(core[0]), float(core[-1])))
        return out
    raise TypeError(
        f"{type(encoder).__name__} no expone un grid calibrado; sólo el encoder KAN lo tiene."
    )


# ── Informe por lote ─────────────────────────────────────────────────────────

def drift_report(x_batch: np.ndarray, reference: DriftReference,
                 calibrated: list[tuple[float, float]] | None = None) -> list[dict]:
    """
    Calcula, para un lote ``x_batch`` [n, n_fields], una fila por campo con:

        field, n, psi, psi_level, coverage, coverage_alert,
        share_below, share_above, cal_low, cal_high

    ``calibrated`` puede omitirse si la referencia ya lleva los rangos
    (``DriftReference.fit(..., calibrated_ranges=...)``). Si no hay rangos
    en ninguno de los dos sitios, la cobertura se reporta como NaN y sin
    alerta: no se inventa un rango.
    """
    x = _as_2d(x_batch)
    if x.shape[1] != len(reference.field_names):
        raise ValueError(
            f"el lote tiene {x.shape[1]} columnas y la referencia {len(reference.field_names)}"
        )
    ranges = calibrated if calibrated is not None else (
        [tuple(r) for r in reference.calibrated_ranges] or None
    )
    rows = []
    for j, name in enumerate(reference.field_names):
        col = x[:, j]
        finite = col[np.isfinite(col)]
        counts, _ = np.histogram(finite, bins=np.asarray(reference.edges[j]))
        actual = counts / max(counts.sum(), 1)
        value = psi(reference.expected[j], actual)

        row = {
            "field": name, "n": int(col.size),
            "psi": value, "psi_level": psi_level(value),
            "coverage": float("nan"), "coverage_alert": False,
            "share_below": float("nan"), "share_above": float("nan"),
            "cal_low": float("nan"), "cal_high": float("nan"),
        }
        if ranges is not None:
            lo, hi = ranges[j]
            cov = range_coverage(col, lo, hi)
            row.update({
                "coverage": cov,
                "coverage_alert": bool(np.isfinite(cov) and cov < COVERAGE_ALERT),
                "share_below": float(np.mean(col < lo)) if col.size else float("nan"),
                "share_above": float(np.mean(col > hi)) if col.size else float("nan"),
                "cal_low": float(lo), "cal_high": float(hi),
            })
        rows.append(row)
    return rows


def summarize(report: list[dict]) -> dict:
    """Resumen de un informe: peor PSI, peor cobertura y campos en alerta."""
    psis = [r["psi"] for r in report]
    covs = [r["coverage"] for r in report if np.isfinite(r["coverage"])]
    return {
        "n_fields": len(report),
        "max_psi": float(max(psis)) if psis else float("nan"),
        "worst_psi_field": report[int(np.argmax(psis))]["field"] if psis else None,
        "min_coverage": float(min(covs)) if covs else float("nan"),
        "fields_psi_alert": [r["field"] for r in report if r["psi_level"] == "alert"],
        "fields_psi_warning": [r["field"] for r in report if r["psi_level"] == "warning"],
        "fields_coverage_alert": [r["field"] for r in report if r["coverage_alert"]],
    }


# ── utilidades ───────────────────────────────────────────────────────────────

def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"se esperaba un array 2D, llegó ndim={x.ndim}")
    return x


def _inf_to_str(v: float):
    if v == np.inf:
        return "inf"
    if v == -np.inf:
        return "-inf"
    return float(v)


def _str_to_inf(v) -> float:
    if v == "inf":
        return np.inf
    if v == "-inf":
        return -np.inf
    return float(v)
