# Microsoft Fabric Notebook — 07_drift_monitoring
# Monitorización de deriva del encoder desplegado sobre el flujo de streaming.
# Attach kanrec_lakehouse before running.
#
#   %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"
#
# Qué hace y por qué
# ------------------
# El modelo desplegado está congelado, así que su función de codificación φ
# no cambia por sí sola. Lo que se vigila es si SIGUE SIENDO VÁLIDA sobre los
# datos que llegan ahora:
#
#   1. Cobertura   — ¿los datos entrantes caen dentro del rango donde se
#                    calibró cada spline? Fuera de él, la fórmula extraída no
#                    describe lo que el modelo hace con esas filas. Es el
#                    la revision critica de este proyecto, pero continuo y en producción.
#   2. Distribución — PSI por campo entre la muestra de entrenamiento y la
#                    ventana reciente del stream.
#
# Ambas señales son posibles porque el encoder es interpretable: con una capa
# densa no hay «rango calibrado» que vigilar. Con un encoder opaco solo queda
# esperar a que caiga el AUC, que es detectar el problema cuando ya ocurrió.
#
# La señal de deriva simbólica (reajustar y comparar formas) está implementada
# en kanrec.drift pero NO se ejecuta aquí: su poder discriminante resultó
# limitado y está documentado como tal en el módulo y en la memoria.
# ============================================================================

# ---------------------------------------------------------------------------
# IMPORTANTE — instalacion de kanrec en ejecucion por PIPELINE
#
# `%pip install` esta DESHABILITADO cuando un notebook se ejecuta desde un
# Data Pipeline: solo funciona en sesiones interactivas. Verificado en Fabric:
#   MagicUsageError: %pip magic command is disabled
#
# Por eso la primera celda de cada notebook NO instala nada. El paquete se
# resuelve por una de estas dos vias, ambas compatibles con pipeline:
#
#   A) Carpeta en Files (rapida, sin publicar entorno). Subir la carpeta
#      `kanrec/` a Files/libs/ y anadir al inicio del notebook:
#
#          import sys
#          sys.path.insert(0, "/lakehouse/default/Files/libs")
#
#   B) Entorno de Fabric (la via formal). Workspace -> Nuevo -> Entorno ->
#      Bibliotecas personalizadas -> subir kanrec-0.3.2-py3-none-any.whl ->
#      Publicar -> asignar el entorno al workspace o al notebook.
#
# Las dependencias (torch, scipy, scikit-learn, pandas, pyarrow) ya vienen en
# el runtime de Fabric, asi que ninguna de las dos vias necesita resolverlas.
# ---------------------------------------------------------------------------
import json

import numpy as np
import pandas as pd
import torch
from pyspark.sql import functions as F

from kanrec.drift import coverage_drift, distribution_drift, drift_report
from kanrec.spark_utils import random_sample

NUMERICAL_COLS = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
VENTANA_FILAS = 50_000          # tamaño de la ventana reciente a auditar
REFERENCIA_FILAS = 50_000       # muestra de entrenamiento como referencia

# ── 1. Modelo desplegado ───────────────────────────────────────────────────
import glob

from kanrec.model import KANRecModel

_ckpts = sorted(glob.glob("/lakehouse/default/Files/checkpoints/*kan-bspline*.pt"))
if not _ckpts:
    raise FileNotFoundError(
        "No hay checkpoint de kan-bspline. Ejecuta antes 04_model_comparison."
    )
CKPT = _ckpts[0]
print(f"Modelo auditado: {CKPT.split('/')[-1]}")

_sd = torch.load(CKPT, map_location="cpu")
_cards = [_sd[f"cat_embeddings.{i}.weight"].shape[0] - 1
          for i in range(len(CATEGORICAL_COLS))]
model = KANRecModel(num_numerical=len(NUMERICAL_COLS), cat_cardinalities=_cards,
                    embedding_dim=16, kan_grid_size=10)
model.load_state_dict(_sd)
model.eval()

# ── 2. Referencia (entrenamiento) y ventana reciente (stream) ──────────────
print("\nCargando referencia y ventana reciente...")

ref_pd = random_sample(spark.read.table("train"), n_rows=REFERENCIA_FILAS,
                       seed=42).toPandas()
x_ref = torch.tensor(ref_pd[NUMERICAL_COLS].fillna(0).values.astype("float32"))
print(f"  referencia (train):      {len(ref_pd):,} filas")

# La ventana reciente sale del stream ya procesado, que aplica EXACTAMENTE
# los mismos estadísticos que el batch (scaler_stats.json de 01). Sin esa
# consistencia, cualquier "deriva" detectada sería un artefacto de haber
# normalizado de dos formas distintas, no un cambio real en los datos.
try:
    stream = spark.read.table("streaming_processed")
    n_stream = stream.count()
    if n_stream < 1000:
        raise ValueError(f"solo {n_stream} filas en streaming_processed")
    new_pd = random_sample(stream, n_rows=min(VENTANA_FILAS, n_stream),
                            seed=7).toPandas()
    origen = "streaming_processed"
except Exception as e:
    # Sin stream suficiente se audita el conjunto de test: la mecánica es la
    # misma y permite validar el notebook aunque el Eventstream lleve poco
    # tiempo activo.
    print(f"  (stream no disponible: {e}; se usa 'test' como ventana)")
    new_pd = random_sample(spark.read.table("test"), n_rows=VENTANA_FILAS,
                            seed=7).toPandas()
    origen = "test"

x_new = torch.tensor(new_pd[NUMERICAL_COLS].fillna(0).values.astype("float32"))
print(f"  ventana reciente ({origen}): {len(new_pd):,} filas")

# ── 3. Señales de deriva ───────────────────────────────────────────────────
print(f"\n{'='*70}\nAUDITORÍA DE DERIVA\n{'='*70}")

señales = (coverage_drift(model, x_new, NUMERICAL_COLS)
           + distribution_drift(x_ref, x_new, NUMERICAL_COLS))
informe = drift_report(señales, verbose=True)

# ── 4. Persistencia para Power BI y Data Activator ─────────────────────────
filas = [s.to_row() for s in señales]
for f in filas:
    f["checkpoint"] = CKPT.split("/")[-1]
    f["ventana_origen"] = origen
    f["n_filas_ventana"] = len(new_pd)

drift_df = spark.createDataFrame(pd.DataFrame(filas)) \
    .withColumn("evaluado_en", F.current_timestamp())

# append, no overwrite: la serie histórica es lo que permite ver una
# tendencia y no solo una foto puntual.
(drift_df.write.format("delta").mode("append")
    .option("mergeSchema", "true").saveAsTable("drift_signals"))

print(f"\ndrift_signals: {len(filas)} señales añadidas "
      f"(histórico: {spark.read.table('drift_signals').count()} filas)")

with open("/lakehouse/default/Files/results/drift_report.json", "w") as f:
    json.dump(informe, f, indent=2)

# ── 5. Métrica escalar para la regla de Data Activator ─────────────────────
# Data Activator reacciona mejor a una serie numérica que a un texto: se
# expone el número de señales críticas como métrica vigilable.
resumen = spark.createDataFrame(pd.DataFrame([{
    "checkpoint": CKPT.split("/")[-1],
    "n_criticos": informe["n_criticos"],
    "n_avisos": informe["n_avisos"],
    "accion": informe["action"],
    "cobertura_max": informe["por_senal"].get("cobertura", {}).get("max", 0.0),
    "psi_max": informe["por_senal"].get("distribucion", {}).get("max", 0.0),
}])).withColumn("evaluado_en", F.current_timestamp())

(resumen.write.format("delta").mode("append")
    .option("mergeSchema", "true").saveAsTable("drift_summary"))

print("\ndrift_summary actualizada. Regla sugerida en Data Activator:")
print("  alertar cuando n_criticos > 0  ->  la fórmula publicada ha dejado")
print("  de describir el comportamiento del modelo sobre los datos actuales.")
