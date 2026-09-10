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
    "\nRECORDATORIO para la memoria (hallazgo A7): esta formula describe la "
    "dimension 0 del embedding de cada campo, no la funcion de scoring "
    "completa (que tambien pasa por las 26 categoricas, la interaccion y "
    "la cabeza). La metrica de fidelidad que cuantifica esa brecha es "
    "tarea del siguiente paso."
)
