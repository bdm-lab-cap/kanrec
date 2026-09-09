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
    "exp":     lambda x, a, b: a * np.exp(np.clip(x, -10, 10)) + b,
    "square":  lambda x, a, b: a * x**2 + b,
    "sqrt":    lambda x, a, b: a * np.sqrt(np.abs(x)) + b,
    "inverse": lambda x, a, b: a / (np.abs(x) + 1e-6) + b,
    "sigmoid": lambda x, a, b: a / (1 + np.exp(-x)) + b,
    "linear":  lambda x, a, b: a * x + b,
}
FORMULA_TEMPLATES = {
    "log": "{a}·log(|x|+1) + {b}", "exp": "{a}·exp(x) + {b}",
    "square": "{a}·x² + {b}", "sqrt": "{a}·√|x| + {b}",
    "inverse": "{a}/(|x|+ε) + {b}", "sigmoid": "{a}·σ(x) + {b}", "linear": "{a}·x + {b}",
}


def fit_field(model: KANRecModel, field_idx: int, r2_threshold: float = 0.90) -> dict:
    """
    Ajusta la libreria de operadores sobre la dimension 0 del embedding del
    campo, evaluada en el rango REAL calibrado del campo (no en [-3, 3]
    fijo — ver KANNumericalEncoder.get_spline_curves).

    NOTA (hallazgo A7, pendiente): esto describe una de las EMBEDDING_DIM
    dimensiones de phi_j, no la funcion de scoring completa del modelo.
    Ese alcance debe quedar explicito en la memoria, no solo aqui.
    """
    x_grid, y_curves = model.numerical_encoder.get_spline_curves(field_idx)
    x_np, y_np = x_grid.numpy(), y_curves[:, 0].numpy()
    best = {"operator": None, "r2": -1.0, "params": None, "formula": "?", "accepted": False}
    for name, fn in OPERATOR_LIBRARY.items():
        try:
            params, _ = curve_fit(fn, x_np, y_np, maxfev=5000)
            y_pred = fn(x_np, *params)
            r2 = float(1 - np.sum((y_np - y_pred) ** 2) / (np.sum((y_np - y_np.mean()) ** 2) + 1e-10))
            if r2 > best["r2"]:
                a, b = round(float(params[0]), 4), round(float(params[1]), 4)
                best = {"operator": name, "r2": r2, "params": [float(params[0]), float(params[1])],
                        "formula": FORMULA_TEMPLATES[name].format(a=a, b=b), "accepted": r2 >= r2_threshold}
        except Exception as exc:
            print(f"    (aviso: el operador '{name}' no ajusto para el campo {field_idx}: {exc})")
            continue
    return best


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
    print(f"L1 pruning (norma del spline, no de todos los parametros): "
          f"{len(surviving)}/{len(norms)} campos sobreviven")
    # NOTA (hallazgo B2, pendiente): el umbral del percentil 20 elimina,
    # por construccion, ~20% de los campos tengan o no importancia real
    # para la prediccion. Un criterio basado en la caida de AUC al anular
    # cada campo es mas defendible y es tarea del paso de interpretabilidad.

    seed_results = {}
    for j in surviving:
        r = fit_field(model, j)
        field = NUMERICAL_COLS[j]
        seed_results[field] = r
        mark = "✓" if r["accepted"] else "✗"
        print(f"  {mark} {field}: {r['operator']:8s} R²={r['r2']:.4f}  {r['formula']}")
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
