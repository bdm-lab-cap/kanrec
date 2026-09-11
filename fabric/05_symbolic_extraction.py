# Microsoft Fabric Notebook — 05_symbolic_extraction
# Extracts symbolic scoring rules from trained KAN-REC models.
# Run AFTER 04_model_comparison. Attach kanrec_lakehouse before running.
#
# Requiere el paquete kanrec instalado en esta sesion:
#   %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"
#
# Fixes aplicados (auditoria de tribunal):
#
#   B5 — Este notebook redefinia KANNumericalEncoder/KANRecModel de forma
#        LOCAL, con una arquitectura de interaccion distinta a la que de
#        verdad entrenaron los notebooks de entrenamiento. Como resultado,
#        load_state_dict() lanzaba una excepcion de claves incompatibles
#        contra los checkpoints reales — este script, tal como estaba, no
#        podia ejecutarse contra los resultados reales del TFM.
#        Ahora se importa `from kanrec.model import KANRecModel`: una unica
#        definicion, la misma que entrena 04_model_comparison.
#
#   A3 — get_spline_curves evaluaba siempre en [-3, 3], una ventana fija
#        que para casi todos los campos cae fuera del rango donde el
#        grid del spline esta activo. Como el buffer `grid` de cada KAN se
#        guarda dentro del checkpoint (es un buffer registrado de PyTorch),
#        cargar el checkpoint restaura tambien la calibracion — y
#        get_spline_curves (ya corregido en el paquete) evalua sobre esa
#        misma calibracion, no sobre una ventana arbitraria.
#
#   (pendiente, fuera del alcance de este paso — hallazgo A7): la formula
#   ajustada aqui describe UNA de las `embedding_dim` dimensiones del
#   embedding del campo, no la funcion de scoring completa. Este notebook
#   sigue documentando eso mismo mas abajo; la metrica de fidelidad
#   (sustituir phi_j dentro del modelo y medir la caida de AUC) es tarea
#   del siguiente paso (interpretabilidad), no de este.

import json, os
import numpy as np
import torch
from scipy.optimize import curve_fit
from collections import Counter

from kanrec.model import KANRecModel

CKPT_PATH = "/lakehouse/default/Files/checkpoints"
with open("/lakehouse/default/Files/config/feature_selection.json") as f:
    sel = json.load(f)
NUMERICAL_COLS   = sel["selected"]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]

# Grid size usado en la comparativa principal (04_model_comparison, CELDA 4
# la deja en su valor por defecto = 10). Si cambias el default alli, cambia
# tambien aqui: el buffer `grid` que se restaura al cargar el checkpoint
# tiene una forma que depende de grid_size, asi que hay que instanciar el
# modelo con el MISMO grid_size antes de cargar los pesos.
KAN_GRID_SIZE = 10
EMBEDDING_DIM = 16

# Operator library
OPERATOR_LIBRARY = {
    "log":     lambda x, a, b: a * np.log(np.abs(x) + 1) + b,
    # Mismo clip que kanrec/symbolic.py (+-500), no +-10: un clip agresivo a
    # +-10 truncaba la curva DENTRO del rango de datos reales. Ambos ficheros
    # deben usar la MISMA libreria de operadores (hallazgo C9).
    "exp":     lambda x, a, b: a * np.exp(np.clip(x, -500, 500)) + b,
    "square":  lambda x, a, b: a * x ** 2 + b,
    "sqrt":    lambda x, a, b: a * np.sqrt(np.abs(x)) + b,
    "inverse": lambda x, a, b: a / (np.abs(x) + 1e-6) + b,
    "sigmoid": lambda x, a, b: a / (1 + np.exp(np.clip(-x, -500, 500))) + b,
    "linear":  lambda x, a, b: a * x + b,
}
FORMULA_TEMPLATES = {
    "log": "{a}·log(|x|+1) + {b}", "exp": "{a}·exp(x) + {b}",
    "square": "{a}·x² + {b}", "sqrt": "{a}·√|x| + {b}",
    "inverse": "{a}/(|x|+ε) + {b}", "sigmoid": "{a}·σ(x) + {b}", "linear": "{a}·x + {b}",
}


def _fit_one_curve(x_np, y_np, field_idx, dim):
    """Ajusta la libreria de operadores a UNA curva y devuelve el mejor."""
    best = {"operator": None, "r2": -1.0, "params": None, "formula": "?"}
    for name, fn in OPERATOR_LIBRARY.items():
        try:
            # np.errstate: curve_fit explora deliberadamente valores extremos
            # del parametro 'a', donde a * exp(x) desborda float64. NumPy
            # devuelve inf/NaN, el optimizador los descarta y sigue -- el
            # aviso es ruido, no un error. Acotar el exponente no lo evita:
            # el desbordamiento viene del PRODUCTO, y 'a' no esta acotado.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                params, _ = curve_fit(fn, x_np, y_np, maxfev=5000)
                y_pred = fn(x_np, *params)
            if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(params)):
                continue
            ss_tot = np.sum((y_np - y_np.mean()) ** 2)
            if ss_tot < 1e-12:      # curva plana: el R2 no esta definido
                continue
            r2 = float(1 - np.sum((y_np - y_pred) ** 2) / (ss_tot + 1e-10))
            if r2 > best["r2"]:
                a, b = round(float(params[0]), 4), round(float(params[1]), 4)
                best = {"operator": name, "r2": r2,
                        "params": [float(params[0]), float(params[1])],
                        "formula": FORMULA_TEMPLATES[name].format(a=a, b=b)}
        except Exception as exc:
            print(f"    (aviso: operador '{name}' no ajusto en campo {field_idx} dim {dim}: {exc})")
            continue
    return best


def fit_field(model: KANRecModel, field_idx: int, r2_threshold: float = 0.90) -> dict:
    """
    Ajusta la libreria de operadores a las EMBEDDING_DIM dimensiones de
    phi_j, evaluadas en el rango REAL calibrado del campo.

    Antes solo se ajustaba la dimension 0 y se reportaba como si describiera
    el campo entero, asumiendo sin demostrarlo que esa dimension era
    representativa. Ahora se ajustan todas y se reporta:
      - el operador DOMINANTE (el que gana en mas dimensiones),
      - r2_mean / r2_std / r2_min sobre las 16, para poder afirmar que la
        forma detectada es una propiedad del campo y no de una proyeccion,
      - operator_agreement: en que fraccion de las dimensiones gana el
        operador dominante.
    Se conserva `dim0_*` para poder comparar con los resultados anteriores.

    NOTA (hallazgo A7): esto describe phi_j, la funcion de CODIFICACION del
    campo, no la funcion de scoring completa (que ademas pasa por las 26
    categoricas, la interaccion y la cabeza). Ese alcance debe quedar
    explicito en la memoria.
    """
    x_grid, y_curves = model.numerical_encoder.get_spline_curves(field_idx)
    x_np = x_grid.numpy()
    n_dims = y_curves.shape[1]

    per_dim = []
    for d in range(n_dims):
        fit = _fit_one_curve(x_np, y_curves[:, d].numpy(), field_idx, d)
        if fit["operator"] is not None:
            per_dim.append(fit)

    if not per_dim:
        return {"operator": None, "r2": -1.0, "params": None, "formula": "?",
                "accepted": False, "n_dims_fitted": 0}

    ops = [f["operator"] for f in per_dim]
    dominant, n_dom = Counter(ops).most_common(1)[0]
    r2s = np.array([f["r2"] for f in per_dim], dtype=float)

    # La formula representativa: la del ajuste con mejor R2 ENTRE las
    # dimensiones que eligieron el operador dominante.
    dom_fits = [f for f in per_dim if f["operator"] == dominant]
    representative = max(dom_fits, key=lambda f: f["r2"])

    return {
        "operator": dominant,
        "r2": float(r2s.mean()),
        "params": representative["params"],
        "formula": representative["formula"],
        "accepted": bool(r2s.mean() >= r2_threshold),
        # Evidencia de que la forma es del campo, no de una proyeccion
        "n_dims_fitted": len(per_dim),
        "operator_agreement": n_dom / len(per_dim),
        "r2_mean": float(r2s.mean()),
        "r2_std": float(r2s.std()),
        "r2_min": float(r2s.min()),
        "r2_max": float(r2s.max()),
        # Comparabilidad con la version anterior (solo dim 0)
        "dim0_operator": per_dim[0]["operator"],
        "dim0_r2": per_dim[0]["r2"],
    }


def load_model_from_checkpoint(ckpt_path: str) -> KANRecModel:
    """
    Reconstruye el modelo con la MISMA arquitectura usada al entrenar
    (kanrec.model.KANRecModel, no una copia local) y carga los pesos —
    incluido el buffer `grid` calibrado de cada campo.
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # Los pesos de nn.Embedding tienen forma (card + 1, embedding_dim)
    # porque KANRecModel construye nn.Embedding(card + 1, ...). Hay que
    # restarle 1 antes de volver a pasarlo al constructor, o se crearia
    # un embedding de una fila mas de la que el checkpoint tiene.
    cat_cardinalities = [
        state_dict[f"cat_embeddings.{i}.weight"].shape[0] - 1
        for i in range(len(CATEGORICAL_COLS))
    ]

    model = KANRecModel(
        num_numerical=len(NUMERICAL_COLS),
        cat_cardinalities=cat_cardinalities,
        embedding_dim=EMBEDDING_DIM,
        kan_grid_size=KAN_GRID_SIZE,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Deteccion del origen de los checkpoints ────────────────────────────────
# Este notebook puede correr sobre dos origenes distintos:
#   - 04_model_comparison.py       -> escribe "best_kan-bspline_gs{N}_s{seed}.pt"
#                                     con 3 semillas (42, 123, 256)
#   - 04_model_comparison_TRIAL.py -> escribe "trial_kan-bspline_gs{N}_s{seed}.pt"
#                                     con 1 sola semilla (42), escala reducida
# Antes esto estaba hardcodeado al prefijo "best_" y a las 3 semillas, asi que
# tras ejecutar solo el trial imprimia "Missing:" tres veces y generaba un
# symbolic_results.json VACIO sin avisar de que no habia extraido nada.
def discover_checkpoints(ckpt_dir: str, grid_size: int) -> list[tuple[int, str]]:
    """Devuelve [(seed, ruta)] para los checkpoints de kan-bspline disponibles."""
    if not os.path.isdir(ckpt_dir):
        return []
    found = []
    for prefix in ("best", "trial"):
        for seed in (42, 123, 256):
            path = f"{ckpt_dir}/{prefix}_kan-bspline_gs{grid_size}_s{seed}.pt"
            if os.path.exists(path):
                found.append((seed, path))
        if found:  # si hay checkpoints "best", se prefieren sobre los "trial"
            print(f"Usando checkpoints con prefijo '{prefix}_' "
                  f"({len(found)} semilla(s): {[s for s, _ in found]})")
            if prefix == "trial":
                print("  AVISO: son checkpoints del notebook TRIAL (escala "
                      "reducida, 1 semilla). El informe de estabilidad entre "
                      "semillas no sera significativo -- para eso hacen falta "
                      "los checkpoints de la comparativa completa.")
            return found
    return []


# ── Extraccion ─────────────────────────────────────────────────────────────
checkpoints = discover_checkpoints(CKPT_PATH, KAN_GRID_SIZE)
if not checkpoints:
    raise FileNotFoundError(
        f"No se encontro ningun checkpoint de kan-bspline con grid_size="
        f"{KAN_GRID_SIZE} en {CKPT_PATH}.\n"
        f"Ejecuta antes 04_model_comparison.py (o 04_model_comparison_TRIAL.py) "
        f"y comprueba que KAN_GRID_SIZE aqui coincide con el grid_size usado alli."
    )

results_all_seeds = {}
for seed, ckpt in checkpoints:
    print(f"\n{'='*60}\nSeed {seed}")
    model = load_model_from_checkpoint(ckpt)

    norms = model.numerical_encoder.get_edge_norms()
    surviving = [j for j, n in enumerate(norms) if n >= np.percentile(norms, 20)]
    # ATENCION al interpretar esto (hallazgo B2): el percentil 20 descarta
    # ~20% de los campos POR CONSTRUCCION, tengan o no importancia real.
    # No es una medida de importancia: es un criterio de conveniencia para
    # acotar cuantas formulas hay que inspeccionar. En la memoria debe
    # presentarse asi, no como si "10 de 13 campos resultaran relevantes".
    # Una poda por caida de AUC al anular cada campo seria mas rigurosa y
    # queda como linea de trabajo futura.
    print(f"Poda por norma L1 del spline (percentil {20}): se retienen "
          f"{len(surviving)}/{len(norms)} campos para inspeccion")
    print("  (criterio de conveniencia, NO una medida de importancia: el "
          "percentil descarta ~20% por construccion)")

    seed_results = {}
    for j in surviving:
        r = fit_field(model, j)
        field = NUMERICAL_COLS[j]
        seed_results[field] = r
        mark = "✓" if r["accepted"] else "✗"
        # R2 medio sobre las 16 dimensiones +- desviacion, y en que fraccion
        # de ellas gana el operador dominante: es la evidencia de que la
        # forma detectada es del campo y no de una proyeccion concreta.
        print(f"  {mark} {field}: {r['operator']:8s} "
              f"R²={r['r2_mean']:.4f}±{r['r2_std']:.4f} (min {r['r2_min']:.4f}) "
              f"acuerdo={r['operator_agreement']:.0%} de {r['n_dims_fitted']} dims  "
              f"{r['formula']}")
    results_all_seeds[seed] = seed_results

# ── Informe de estabilidad ──────────────────────────────────────────────────
print("\nSTABILITY REPORT")
print("=" * 60)
field_ops = {}
for seed, results in results_all_seeds.items():
    for field, r in results.items():
        if r["accepted"]:
            field_ops.setdefault(field, []).append(r["operator"])

if not field_ops:
    print("  (ninguna formula supero el umbral de R2 >= 0.90 -- no hay nada "
          "que reportar aqui)")
for field, ops in sorted(field_ops.items()):
    dominant, count = Counter(ops).most_common(1)[0]
    n_seeds = len(results_all_seeds)
    print(f"  {field:<6}: {dominant:<10} {count}/{n_seeds} seeds ({count/n_seeds:.0%})")

if len(results_all_seeds) < 2:
    print("\n  AVISO: solo hay 1 semilla disponible, asi que la columna "
          "'stability' es trivialmente 100% y NO mide estabilidad real. "
          "Para un informe de estabilidad con sentido hacen falta los 3 "
          "checkpoints de la comparativa completa (04_model_comparison.py).")

# ── Guardado de resultados ──────────────────────────────────────────────────
os.makedirs("/lakehouse/default/Files/results", exist_ok=True)
n_seeds = len(results_all_seeds)
with open("/lakehouse/default/Files/results/symbolic_results.json", "w") as f:
    json.dump({
        "kan_grid_size": KAN_GRID_SIZE,
        # Trazabilidad: de que checkpoints salio este fichero. Sin esto era
        # imposible saber si un symbolic_results.json venia de la corrida
        # completa o del trial de escala reducida.
        "source_checkpoints": [os.path.basename(p) for _, p in checkpoints],
        "n_seeds": n_seeds,
        "results_by_seed": {
            str(k): {fld: v for fld, v in res.items()}
            for k, res in results_all_seeds.items()
        },
        "stability": {
            f: {"dominant": Counter(ops).most_common(1)[0][0],
                "stability": Counter(ops).most_common(1)[0][1] / n_seeds}
            for f, ops in field_ops.items()
        },
    }, f, indent=2)
print("\nsymbolic_results.json saved to Files/results/")
print(
    "\nRECORDATORIO para la memoria (hallazgo A7): la formula describe phi_j, "
    "la funcion de CODIFICACION del campo (ajustada sobre las 16 dimensiones "
    "de su embedding), NO la funcion de scoring completa: el scoring pasa "
    "ademas por las 26 categoricas, la capa de interaccion y la cabeza. "
    "Presentalo como 'que hace el modelo con esta variable', no como 'la "
    "formula del scoring'. Cuantificar la brecha entre phi_j y la prediccion "
    "final requiere la ablacion por sustitucion (reemplazar phi_j por su "
    "formula dentro del modelo y medir la caida de AUC), que es el siguiente paso."
)

# ============================================================================
# ABLACION POR SUSTITUCION — la metrica de fidelidad
# ============================================================================
# Reemplaza phi_j por su formula simbolica DENTRO del modelo (dejando intactas
# las 26 categoricas, la interaccion y la cabeza) y mide el efecto. Es la
# unica forma de validar la afirmacion de interpretabilidad: si sustituir la
# formula apenas cambia el modelo, la formula describe fielmente lo que el
# encoder hacia.
#
# Se reportan dos metricas y la que manda es la SEGUNDA:
#   - delta AUC: cuanto se degrada la prediccion. Poco sensible en CTR,
#     donde el AUC lo dominan las categoricas.
#   - error de curva (relativo): cuanto se desvia la curva sustituida de la
#     real. Es la medida directa de fidelidad de la formula. Verificado que
#     discrimina: sobre una senal sin(1.5x) el campo con esa senal sale con
#     error 0.58 mientras los demas quedan por debajo de 0.19.
from kanrec.ablation import substitution_ablation
from torch.utils.data import DataLoader, TensorDataset

# Reconstruir un loader de test desde la tabla (misma logica que 04)
from kanrec.spark_utils import random_sample
_test = random_sample(spark.read.table("test"), n_rows=20000, seed=42).toPandas()
_x_num = torch.tensor(_test[NUMERICAL_COLS].fillna(0).values.astype("float32"))
_idx_cols_present = [c for c in [f"C{i}_idx" for i in range(1, 27)] if c in _test.columns]
_x_cat = torch.tensor(_test[_idx_cols_present].fillna(0).values.astype("int64"))
_y = torch.tensor(_test["label"].values.astype("float32"))
_loader = DataLoader(TensorDataset(_x_num, _x_cat, _y), batch_size=2048)

# Usar los resultados de la ULTIMA semilla procesada
_last_seed = max(results_all_seeds)
_results = results_all_seeds[_last_seed]
_model = load_model_from_checkpoint(dict(checkpoints)[_last_seed])

print(f"\n{'='*70}")
print(f"ABLACION POR SUSTITUCION (seed {_last_seed})")
print(f"{'='*70}")
_ablation = substitution_ablation(_model, _loader, _results,
                                  numerical_cols=NUMERICAL_COLS)

# Persistir junto al resto de resultados
with open("/lakehouse/default/Files/results/ablation_report.json", "w") as f:
    json.dump(_ablation, f, indent=2)
print("\nablation_report.json guardado en Files/results/")


# ============================================================================
# TABLAS DELTA PARA POWER BI
# ============================================================================
# powerbi/README.md documenta cinco tablas, de las que solo existian dos
# (experiment_results y streaming_processed). Las tres que faltaban se escriben
# aqui, de modo que el panel sea reproducible en lugar de construirse a mano:
#
#   symbolic_results  — una fila por campo y semilla, con operador y metricas
#   spline_curves     — el grid evaluado de cada curva phi, para graficarla
#                       en Direct Lake (el visual mas ilustrativo del trabajo)
#   baseline_metrics  — resumen por encoder para la pagina comparativa
import pandas as pd
from pyspark.sql import functions as F

print(f"\n{'='*70}")
print("ESCRITURA DE TABLAS DELTA PARA POWER BI")
print(f"{'='*70}")

# ── symbolic_results ───────────────────────────────────────────────────────
_rows = []
for _seed, _res in results_all_seeds.items():
    for _field, _r in _res.items():
        _rows.append({
            "seed": int(_seed),
            "field_name": _field,
            "operator": _r["operator"],
            "formula": _r["formula"],
            "r2_mean": float(_r.get("r2_mean", _r["r2"])),
            "r2_std": float(_r.get("r2_std", 0.0)),
            "r2_min": float(_r.get("r2_min", _r["r2"])),
            "operator_agreement": float(_r.get("operator_agreement", 1.0)),
            "accepted": bool(_r["accepted"]),
        })
if _rows:
    spark.createDataFrame(pd.DataFrame(_rows)) \
        .write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable("symbolic_results")
    print(f"  symbolic_results: {len(_rows)} filas")

# ── spline_curves ──────────────────────────────────────────────────────────
# El grid evaluado por campo y dimension. Permite dibujar phi_j en Power BI
# directamente desde Delta, sin exportar imagenes.
_curve_rows = []
for _seed, _ck in checkpoints:
    _m = load_model_from_checkpoint(_ck)
    for _j in range(len(NUMERICAL_COLS)):
        _xg, _yc = _m.numerical_encoder.get_spline_curves(_j, n_points=60)
        _xn = _xg.numpy()
        for _d in range(min(4, _yc.shape[1])):      # 4 dims bastan para el visual
            _yd = _yc[:, _d].numpy()
            for _i in range(len(_xn)):
                _curve_rows.append({
                    "seed": int(_seed),
                    "field_name": NUMERICAL_COLS[_j],
                    "dim": int(_d),
                    "x": float(_xn[_i]),
                    "phi": float(_yd[_i]),
                })
if _curve_rows:
    spark.createDataFrame(pd.DataFrame(_curve_rows)) \
        .write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable("spline_curves")
    print(f"  spline_curves: {len(_curve_rows):,} puntos")

# ── baseline_metrics ───────────────────────────────────────────────────────
try:
    _exp = spark.read.table("experiment_results")
    # Se agregan SOLO las columnas presentes: el esquema de
    # experiment_results varia entre la version trial (sin tiempos) y la
    # completa. Construir la lista dinamicamente evita que la tabla entera
    # falle por una columna opcional ausente.
    _cols = set(_exp.columns)
    _aggs = [F.count("*").alias("n_seeds")]
    if "test_auc" in _cols:
        _aggs += [F.avg("test_auc").alias("auc_mean"),
                  F.stddev("test_auc").alias("auc_std")]
    if "test_logloss" in _cols:
        _aggs += [F.avg("test_logloss").alias("logloss_mean"),
                  F.stddev("test_logloss").alias("logloss_std")]
    if "train_seconds" in _cols:
        _aggs += [F.avg("train_seconds").alias("train_seconds_mean")]

    (_exp.groupBy("encoder").agg(*_aggs)
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").saveAsTable("baseline_metrics"))
    print(f"  baseline_metrics: resumen por encoder "
          f"({len(_aggs)} metricas desde {sorted(_cols)})")
except Exception as _e:
    print(f"  (baseline_metrics omitida: {_e})")

print("\nTablas listas para Power BI (Direct Lake).")
