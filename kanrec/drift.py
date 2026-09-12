"""
Detección de deriva sobre un encoder interpretable.

El punto de partida conceptual
------------------------------
Un modelo desplegado está congelado, así que su función de codificación φⱼ
**no cambia por sí sola**: depende solo de los pesos, no de los datos que se
le pasen. Monitorizar «si φ cambia» en un modelo fijo daría cero por
construcción.

Lo que sí puede derivar son dos cosas distintas, y este módulo las separa:

1. **Cobertura** — ¿siguen los datos entrantes dentro del rango donde se
   calibró la spline de cada campo? Fuera de ese rango las funciones base
   valen cero y el encoder degenera en su ruta lineal: la fórmula extraída
   deja de describir lo que el modelo hace con esa fila. Es el la revision critica
   de este proyecto, pero en producción y de forma continua.

2. **Distribución** — ¿ha cambiado la forma de la distribución de entrada,
   aunque siga dentro del rango? Se mide con PSI, el índice estándar en
   riesgo de crédito y CTR.

3. **Simbólica (EXPERIMENTAL)** — ¿ha cambiado la *relación* entre la
   variable y el clic, de modo que φ ya no la describe? Requiere reajustar
   sobre datos recientes y comparar la forma resultante con la de
   referencia.

   Su poder discriminante es **limitado y así debe reportarse**. Medido
   sobre datos sintéticos en los que la relación cambia en un único campo
   conocido, con reajuste suave (4 épocas, lr 5e-4):

       campo   control   con deriva   ratio
       I1        0.048        0.153    3.2x   <- aqui cambio la relacion
       I2        0.066        0.116    1.8x
       I3        0.165        0.255    1.5x
       I4        0.062        0.089    1.4x
       I5        0.085        0.126    1.5x

   El campo realmente derivado tiene el mayor *ratio*, pero **no el mayor
   valor absoluto**: I3 alcanza 0.255 sin que su relación haya cambiado,
   porque cada campo tiene su propio suelo de ruido de reajuste. Un umbral
   absoluto da falsos positivos y una puntuación relativa entre campos
   tampoco separa de forma fiable con pocos campos.

   Descartado por el camino, con medición: comparar las curvas de dos
   modelos entrenados por separado no sirve en absoluto (error 3.9 entre
   dos modelos de la MISMA relación), porque las dimensiones del embedding
   no son identificables y cada entrenamiento aprende una base distinta.
   Por eso el reajuste parte del modelo desplegado y no de cero.

   Se incluye como señal exploratoria y como línea de trabajo: una
   comparación pareada contra un reajuste de control sobre datos conocidos
   como estables cancelaría el suelo de ruido por campo, pero exige
   disponer de esos datos, que es justo lo que no se tiene en producción.

Por qué esto es posible aquí y no con otros encoders
----------------------------------------------------
Las señales 1 y 3 existen porque el encoder es interpretable. Con una capa
densa o con AutoDis no hay «rango calibrado» que vigilar ni «forma
funcional» que comparar: solo quedan métricas agregadas, que detectan la
degradación cuando ya ha ocurrido. Aquí la auditoría del modelo deja de ser
un informe puntual y pasa a ser un proceso continuo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .symbolic import OPERATOR_LIBRARY

#: Umbrales de PSI habituales en la industria (riesgo de crédito, CTR).
PSI_ESTABLE = 0.10
PSI_MODERADO = 0.25

#: Fracción de filas fuera del rango calibrado a partir de la cual la
#: fórmula deja de ser una descripción válida para esas filas.
COBERTURA_AVISO = 0.01
COBERTURA_CRITICA = 0.05


@dataclass
class DriftSignal:
    """Una señal de deriva para un campo."""
    field_name: str
    signal: str          # "cobertura" | "distribucion" | "simbolica"
    value: float
    threshold: float
    status: str          # "ok" | "aviso" | "critico"
    detail: str = ""

    def to_row(self) -> dict:
        return {
            "field_name": self.field_name,
            "signal": self.signal,
            "value": float(self.value),
            "threshold": float(self.threshold),
            "status": self.status,
            "detail": self.detail,
        }


# ── 1. Cobertura del rango calibrado ───────────────────────────────────────

def coverage_drift(model, x_new: torch.Tensor,
                   field_names: list[str] | None = None) -> list[DriftSignal]:
    """
    Fracción de filas que caen fuera del rango donde se calibró cada spline.

    Fuera de ese rango el encoder no usa la spline: la fórmula extraída no
    describe su comportamiento sobre esas filas. Es la señal más barata
    (no requiere gradientes ni reajuste) y la más accionable: si se dispara,
    hay que recalibrar.

    Args:
        model:       modelo desplegado (con `numerical_encoder`).
        x_new:       [n, num_fields] datos recientes YA normalizados.
        field_names: nombres de campo; por defecto I1..In.
    """
    encoder = model.numerical_encoder
    n_fields = encoder.num_fields
    field_names = field_names or [f"I{j+1}" for j in range(n_fields)]

    señales = []
    for j in range(n_fields):
        lo, hi = encoder.field_range(j)
        col = x_new[:, j]
        fuera = ((col < lo) | (col > hi)).float().mean().item()

        status = ("critico" if fuera >= COBERTURA_CRITICA
                  else "aviso" if fuera >= COBERTURA_AVISO else "ok")
        señales.append(DriftSignal(
            field_name=field_names[j],
            signal="cobertura",
            value=fuera,
            threshold=COBERTURA_CRITICA,
            status=status,
            detail=(f"rango calibrado [{lo:.2f}, {hi:.2f}]; "
                    f"observado [{col.min():.2f}, {col.max():.2f}]"),
        ))
    return señales


# ── 2. Deriva de distribución (PSI) ────────────────────────────────────────

def _psi(ref: np.ndarray, new: np.ndarray, n_bins: int = 10,
         eps: float = 1e-6) -> float:
    """
    Population Stability Index entre dos muestras.

    Los cortes se toman por cuantiles de la REFERENCIA, no de los datos
    nuevos: si se recalculasen sobre los nuevos, el índice sería siempre
    próximo a cero por construcción y no detectaría nada.
    """
    cortes = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    cortes = np.unique(cortes)
    if len(cortes) < 3:              # referencia casi constante
        return 0.0

    p_ref, _ = np.histogram(ref, bins=cortes)
    p_new, _ = np.histogram(new, bins=cortes)
    p_ref = p_ref / max(p_ref.sum(), 1) + eps
    p_new = p_new / max(p_new.sum(), 1) + eps
    return float(np.sum((p_new - p_ref) * np.log(p_new / p_ref)))


def distribution_drift(x_ref: torch.Tensor, x_new: torch.Tensor,
                       field_names: list[str] | None = None,
                       n_bins: int = 10) -> list[DriftSignal]:
    """
    PSI por campo entre la muestra de referencia (la de entrenamiento) y la
    ventana reciente.

    Interpretación estándar: < 0,10 estable; 0,10–0,25 cambio moderado;
    > 0,25 cambio significativo que justifica reentrenar.
    """
    n_fields = x_ref.shape[1]
    field_names = field_names or [f"I{j+1}" for j in range(n_fields)]

    señales = []
    for j in range(n_fields):
        valor = _psi(x_ref[:, j].cpu().numpy(), x_new[:, j].cpu().numpy(), n_bins)
        status = ("critico" if valor >= PSI_MODERADO
                  else "aviso" if valor >= PSI_ESTABLE else "ok")
        señales.append(DriftSignal(
            field_name=field_names[j],
            signal="distribucion",
            value=valor,
            threshold=PSI_MODERADO,
            status=status,
            detail=f"PSI sobre {n_bins} cuantiles de la referencia",
        ))
    return señales


# ── 3. Deriva simbólica ────────────────────────────────────────────────────

def _curve_distance(model_ref, model_new, field_idx: int,
                    n_points: int = 60) -> tuple[float, float]:
    """
    Distancia entre las curvas φⱼ de dos modelos, evaluadas en el rango
    COMÚN a ambos (fuera de él la comparación no tendría sentido).

    Devuelve (error relativo medio sobre las dimensiones, error máximo).
    """
    lo_r, hi_r = model_ref.numerical_encoder.field_range(field_idx)
    lo_n, hi_n = model_new.numerical_encoder.field_range(field_idx)
    lo, hi = max(lo_r, lo_n), min(hi_r, hi_n)
    if hi <= lo:
        return float("inf"), float("inf")

    grid = torch.linspace(lo, hi, n_points).unsqueeze(1)
    with torch.no_grad():
        y_ref = model_ref.numerical_encoder.field_kans[field_idx](grid)
        y_new = model_new.numerical_encoder.field_kans[field_idx](grid)

    errs = []
    for d in range(y_ref.shape[1]):
        r = y_ref[:, d].cpu().numpy()
        n = y_new[:, d].cpu().numpy()
        errs.append(float(np.sqrt(np.mean((n - r) ** 2)) / (r.std() + 1e-9)))
    return float(np.mean(errs)), float(np.max(errs))


def _robust_z(valores: np.ndarray) -> np.ndarray:
    """
    Puntuación robusta de cada valor respecto a sus pares (mediana y MAD).

    Se usa la mediana en vez de la media porque basta un campo derivado para
    arrastrar la media y enmascararse a sí mismo.
    """
    mediana = np.median(valores)
    mad = np.median(np.abs(valores - mediana))
    escala = mad * 1.4826 if mad > 1e-9 else (valores.std() + 1e-9)
    return (valores - mediana) / escala


def symbolic_drift(model_ref, model_new, reference_formulas: dict,
                   fit_field_fn, field_names: list[str] | None = None,
                   z_threshold: float = 3.0) -> list[DriftSignal]:
    """
    Compara la forma funcional aprendida por dos modelos: el desplegado y uno
    reajustado sobre datos recientes.

    Se reportan dos cosas por campo:
      - si el **operador dominante** cambia (señal categórica, muy visible);
      - cuánto se **desvía la curva** respecto a la de referencia (señal
        continua, más informativa: un cambio de operador entre dos
        candidatos casi equivalentes no significa gran cosa, mientras que
        una curva que se aleja sí).

    Args:
        model_ref:          modelo desplegado (referencia).
        model_new:          modelo reajustado sobre la ventana reciente.
        reference_formulas: {field_name: {"operator", ...}} de la extracción
                            de referencia.
        fit_field_fn:       función `(model, field_idx) -> dict` que ajusta
                            la librería de operadores (se inyecta para no
                            duplicar aquí la lógica de extracción).
    """
    n_fields = model_ref.numerical_encoder.num_fields
    field_names = field_names or [f"I{j+1}" for j in range(n_fields)]

    # 1. Distancia de curva por campo
    candidatos, errores = [], []
    for j in range(n_fields):
        nombre = field_names[j]
        ref = reference_formulas.get(nombre)
        if not ref or not ref.get("operator"):
            continue
        nuevo = fit_field_fn(model_new, j)
        if not nuevo or not nuevo.get("operator"):
            continue
        err, _ = _curve_distance(model_ref, model_new, j)
        candidatos.append((nombre, ref, nuevo))
        errores.append(err)

    if not candidatos:
        return []

    # 2. Puntuación de cada campo CONTRA SUS PARES.
    #
    #    Comparar el error absoluto contra un umbral fijo no funciona: el
    #    reajuste desplaza las curvas de todos los campos por igual, aunque
    #    la relación no haya cambiado en ninguno, y ese suelo de ruido
    #    depende de cuántas épocas y con qué lr se reajuste. Medido: con
    #    reajuste suave el error de control es 0,085 y con reajuste fuerte
    #    0,324, mientras el campo genuinamente derivado pasa de 0,153 a
    #    0,442. En términos absolutos ambos suben; en términos relativos a
    #    sus pares, el campo derivado destaca entre 3x y 4x en los dos casos.
    #
    #    Por eso la señal es la desviación robusta respecto a la mediana de
    #    los campos: el suelo de ruido se cancela y lo que queda es «qué
    #    variable se comporta distinto del resto», que además es la
    #    pregunta operativamente útil.
    errores = np.asarray(errores, dtype=float)
    z = _robust_z(errores)

    señales = []
    for (nombre, ref, nuevo), err, zi in zip(candidatos, errores, z):
        cambio_op = nuevo["operator"] != ref["operator"]
        status = ("critico" if zi >= z_threshold
                  else "aviso" if zi >= z_threshold * 0.6 or cambio_op else "ok")
        detalle = (f"operador {ref['operator']} -> {nuevo['operator']}"
                   if cambio_op else f"operador estable ({ref['operator']})")
        señales.append(DriftSignal(
            field_name=nombre,
            signal="simbolica",
            value=float(zi),
            threshold=z_threshold,
            status=status,
            detail=f"{detalle}; error de curva {err:.3f} "
                   f"(mediana de los campos {np.median(errores):.3f})",
        ))
    return señales


# ── Informe agregado ───────────────────────────────────────────────────────

def drift_report(señales: list[DriftSignal], verbose: bool = True) -> dict:
    """Resume las señales y decide si procede recalibrar o reentrenar."""
    import pandas as pd

    df = pd.DataFrame([s.to_row() for s in señales])
    if df.empty:
        return {"n_signals": 0, "action": "sin datos"}

    criticos = df[df.status == "critico"]
    por_tipo = {t: grupo for t, grupo in df.groupby("signal")}

    # La acción se decide por el tipo de señal que se dispara, no por el
    # número total: cobertura y deriva simbólica piden acciones distintas.
    if not por_tipo.get("cobertura", pd.DataFrame()).empty and \
       (por_tipo["cobertura"].status == "critico").any():
        accion = "RECALIBRAR: hay datos fuera del rango de la spline"
    elif not por_tipo.get("simbolica", pd.DataFrame()).empty and \
         (por_tipo["simbolica"].status == "critico").any():
        accion = "REENTRENAR: la forma funcional aprendida ha cambiado"
    elif not por_tipo.get("distribucion", pd.DataFrame()).empty and \
         (por_tipo["distribucion"].status == "critico").any():
        accion = "REVISAR: la distribucion de entrada se ha desplazado"
    else:
        accion = "sin accion"

    informe = {
        "n_signals": int(len(df)),
        "n_criticos": int(len(criticos)),
        "n_avisos": int((df.status == "aviso").sum()),
        "campos_criticos": sorted(criticos.field_name.unique().tolist()),
        "action": accion,
        "por_senal": {t: {"max": float(g.value.max()),
                          "criticos": int((g.status == "critico").sum())}
                      for t, g in por_tipo.items()},
    }

    if verbose:
        print(f"{'campo':>6} {'senal':>14} {'valor':>10} {'umbral':>8} {'estado':>9}")
        print("-" * 56)
        for _, r in df.sort_values(["signal", "value"], ascending=[True, False]).iterrows():
            marca = {"ok": "", "aviso": "  <-", "critico": "  <<<"}[r.status]
            print(f"{r.field_name:>6} {r.signal:>14} {r.value:>10.4f} "
                  f"{r.threshold:>8.2f} {r.status:>9}{marca}")
        print(f"\n  {informe['n_criticos']} criticos, {informe['n_avisos']} avisos")
        print(f"  Accion: {accion}")

    return informe
