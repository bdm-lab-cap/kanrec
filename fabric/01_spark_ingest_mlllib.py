# Microsoft Fabric Notebook — 01_spark_ingest_mlllib
# Attach kanrec_lakehouse before running.
#
# Requires the kanrec package installed in this session (first cell of
# every notebook in this project):
#   %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"
#
# Fix applied (2026-09, auditoria de tribunal - hallazgo A2)
# --------------------------------------------------------------
# StandardScaler used to write its output into a NEW vector column
# ("num_scaled") that no downstream notebook ever read: every training
# notebook loaded I1..I13 straight from the Delta table, where I6-I13
# were still in their raw (unnormalised) scale. The fix unpacks the
# scaled vector back into the I6..I13 scalar columns themselves, so the
# Delta table that everything downstream reads is the normalised one.
#
# A `describe()` call at the end proves it: I6..I13 must show mean ~0.0
# and stddev ~1.0 in the printed summary. Keep that output for the
# memoria's Anexo D (evidence that the pipeline does what it claims).

import json, os
import numpy as np
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from kanrec.spark_utils import apply_pipeline_and_unpack

NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
LOG_COLS = [f"I{i}" for i in range(1, 6)]   # log1p (recuento/frecuencia)
STD_COLS = [f"I{i}" for i in range(6, 14)]  # StandardScaler (magnitud libre)

print("Loading Criteo...")
raw = (spark.read.option("sep", "\t").option("inferSchema", "true")
       .csv("Files/raw/criteo_10m.tsv")
       .toDF("label", *NUMERICAL_COLS, *CATEGORICAL_COLS))

# NOTA (deferred, hallazgo B9): los nulos se imputan a 0.0 sin una mascara
# de "is_null" separada, así que "ausente" y "vale cero" quedan fusionados.
# En Criteo el patron de ausencia es predictivo por si mismo; anadir 13
# columnas binarias I{j}_is_null es la mejora natural del siguiente pase,
# no de este (que se centra en que la normalizacion llegue al modelo).
for c in NUMERICAL_COLS:
    raw = raw.withColumn(c, F.when(F.col(c).isNull(), 0.0)
                           .otherwise(F.abs(F.col(c).cast("float"))))

total = raw.count()
print(f"Total rows: {total:,}")

train_raw, val_raw, test_raw = raw.randomSplit([0.8, 0.1, 0.1], seed=42)

def log1p_safe(df, cols):
    for c in cols:
        df = df.withColumn(c, F.log1p(F.greatest(F.col(c), F.lit(0.0))))
    return df

train_log = log1p_safe(train_raw, LOG_COLS)
val_log   = log1p_safe(val_raw,   LOG_COLS)
test_log  = log1p_safe(test_raw,  LOG_COLS)

assembler = VectorAssembler(inputCols=STD_COLS, outputCol="num_raw", handleInvalid="keep")
scaler    = StandardScaler(inputCol="num_raw", outputCol="num_scaled", withMean=True, withStd=True)
indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
             for c in CATEGORICAL_COLS]

pipeline = Pipeline(stages=[assembler, scaler] + indexers)
print("Fitting MLlib Pipeline on TRAIN...")
pm = pipeline.fit(train_log)
pm.write().overwrite().save("Files/models/mlllib_pipeline")


train_t = apply_pipeline_and_unpack(train_log, pm, STD_COLS)
val_t   = apply_pipeline_and_unpack(val_log,   pm, STD_COLS)
test_t  = apply_pipeline_and_unpack(test_log,  pm, STD_COLS)

selection = {"selected": NUMERICAL_COLS, "excluded": [], "chisq_ranking": NUMERICAL_COLS, "spearman_scores": {}}
os.makedirs("/lakehouse/default/Files/config", exist_ok=True)
with open("/lakehouse/default/Files/config/feature_selection.json", "w") as f:
    json.dump(selection, f, indent=2)

train_t.write.format("delta").mode("overwrite").save("Tables/train")
val_t.write.format("delta").mode("overwrite").save("Tables/val")
test_t.write.format("delta").mode("overwrite").save("Tables/test")
print(f"TRAIN: {train_t.count():,} | VAL: {val_t.count():,} | TEST: {test_t.count():,}")

# ── Verificacion (guardar esta salida para el Anexo D) ─────────────────────
# I1..I5 deben mostrar un rango tipico de log1p (valores pequenos, no negativos).
# I6..I13 deben mostrar media ~0.0 y stddev ~1.0: es la prueba de que el
# StandardScaler llega de verdad a las columnas que el modelo lee, y no a
# una columna vectorial que nadie consume.
print("\nVerificacion — I1..I5 (log1p):")
train_t.select(LOG_COLS).describe().show()
print("Verificacion — I6..I13 (StandardScaler, esperado: mean~0, stddev~1):")
train_t.select(STD_COLS).describe().show()

print("Done.")


# ============================================================================
# CELDA 4 (opcional) — Exportar una muestra para Colab
# ============================================================================
# Por que existe esta celda: la capacidad trial de Fabric NO admite cola de
# trabajos (un pico de uso se rechaza al momento con 430, no espera) y la
# sesion de Spark se desaloja tras 20 min de inactividad. La comparativa
# completa de encoders (3 semillas x 3 modelos + ablacion de grid_size) se
# ejecuta por eso en Google Colab Pro (GPU, sin este limite), sobre esta
# MISMA muestra ya normalizada por el pipeline de Fabric — no sobre datos
# reprocesados aparte, para que ambos entornos vean exactamente los mismos
# valores. Ver notebook de Colab: colab/kanrec_full_comparison.ipynb
from kanrec.spark_utils import random_sample

N_EXPORT_TRAIN, N_EXPORT_VAL, N_EXPORT_TEST = 500_000, 100_000, 100_000
EXPORT_PATH = "/lakehouse/default/Files/exports"
os.makedirs(EXPORT_PATH, exist_ok=True)

for split_name, df_split, n in [
    ("train", train_t, N_EXPORT_TRAIN),
    ("val",   val_t,   N_EXPORT_VAL),
    ("test",  test_t,  N_EXPORT_TEST),
]:
    sample = random_sample(df_split, n_rows=n, seed=42)
    sample.toPandas().to_parquet(f"{EXPORT_PATH}/{split_name}_sample.parquet", index=False)
    print(f"  {split_name}_sample.parquet escrito ({n:,} filas objetivo)")

print(f"\nDescarga estos 3 ficheros desde el panel Files de Fabric "
      f"({EXPORT_PATH}) y subelos a Colab (o a tu Google Drive).")
