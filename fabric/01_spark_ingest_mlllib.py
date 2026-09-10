# Microsoft Fabric Notebook — 01_spark_ingest_mlllib
# Attach kanrec_lakehouse before running.
#
# Requires the kanrec package installed in this session (first cell of
# every notebook in this project):
#   %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"
#
# IMPORTANTE (fallo real, verificado en Fabric el 2026-09-09): este
# notebook NUNCA llama a mlflow ni lo instala aparte, pero rompia igual
# con "wrapper() got an unexpected keyword argument 'expected_status'".
# Causa: kanrec<0.3.2 declaraba mlflow como dependencia obligatoria, asi
# que %pip install kanrec instalaba de forma TRANSITIVA un mlflow de PyPI
# sin fijar, que sobreescribe el mlflow propio que Fabric ya trae
# integrado con su plugin synapse.ml.mlflow. Fabric registra
# automaticamente CADA ejecucion de notebook en su propio tracking de
# experimentos (via synapse.ml.mlflow.default_experiment_registry), de
# forma transparente -- por eso el fallo aparecia aunque este notebook no
# use mlflow para nada. kanrec>=0.3.2 ya no arrastra mlflow (paso a un
# extra [train], solo para entrenamiento local/Colab). Si tras el fix
# sigue fallando, reinicia el kernel de PySpark: el mlflow incompatible
# puede haber quedado instalado en el entorno pip aislado de la sesion
# anterior y un simple re-run de la celda no lo revierte.
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

# Limpiar artefactos de ejecuciones anteriores antes de regenerar. Un
# PipelineModel serializado o una tabla Delta de una corrida previa puede
# quedar en un estado que aborta el saveAsTable con
# "MALFORMED_RECORD_IN_PARSING [null,null,null]". Empezar en limpio elimina
# esa clase de fallo. (En Fabric notebookutils.fs.rm siempre existe.)
try:
    import notebookutils
    for _p in ["Files/models/mlllib_pipeline", "Tables/train", "Tables/val", "Tables/test"]:
        try:
            notebookutils.fs.rm(_p, recurse=True)
            print(f"Limpiado artefacto previo: {_p}")
        except Exception:
            pass  # no existia, normal en la primera ejecucion
except ImportError:
    pass  # fuera de Fabric (tests locales)

print("Loading Criteo...")
# Esquema explicito en vez de inferSchema=true. Motivos:
#   - inferSchema fuerza una pasada completa extra sobre los 2.26 GB solo
#     para deducir tipos (mas lento; mala practica en Spark a esta escala).
#   - inferSchema + modo FAILFAST (el default) aborta el job entero con
#     "MALFORMED_RECORD_IN_PARSING" en cuanto una fila no encaja con el tipo
#     deducido, y Criteo (10M filas) tiene filas irregulares. Con el esquema
#     declarado y mode=PERMISSIVE, esas celdas se leen como null (que el
#     bloque de imputacion de abajo ya convierte a 0.0) en vez de reventar.
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

schema = StructType(
    [StructField("label", IntegerType(), True)]
    + [StructField(c, DoubleType(), True) for c in NUMERICAL_COLS]
    + [StructField(c, StringType(), True) for c in CATEGORICAL_COLS]
)
raw = (spark.read
       .option("sep", "\t")
       .option("mode", "PERMISSIVE")
       .schema(schema)
       .csv("Files/raw/criteo_10m.tsv"))

# NOTA (deferred, hallazgo B9): los nulos se imputan a 0.0 sin una mascara
# de "is_null" separada, así que "ausente" y "vale cero" quedan fusionados.
# En Criteo el patron de ausencia es predictivo por si mismo; anadir 13
# columnas binarias I{j}_is_null es la mejora natural del siguiente pase,
# no de este (que se centra en que la normalizacion llegue al modelo).
for c in NUMERICAL_COLS:
    # Ya son DoubleType por el esquema declarado; no se hace cast a float32
    # (float32 perdia precision en el StandardScaler con outliers a ~690
    # desviaciones tipicas, ver verificacion mas abajo).
    raw = raw.withColumn(c, F.when(F.col(c).isNull(), 0.0)
                           .otherwise(F.abs(F.col(c))))

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

# Escritura como TABLA GESTIONADA del catalogo (saveAsTable), NO por ruta
# (.save("Tables/train")). En Fabric, .save() a una ruta escribe los ficheros
# Delta pero NO siempre actualiza la entrada del catalogo que consulta
# spark.read.table("train"): el resultado era que 01 escribia los datos
# normalizados en disco mientras la TABLA 'train' del catalogo seguia
# apuntando a una version anterior con datos crudos (verificado: el historial
# Delta mostraba escrituras de dias atras pese a reejecutar 01). saveAsTable
# con overwrite reemplaza la tabla del catalogo, asi que 04/05/diagnostico,
# que leen por nombre, ven de verdad la version nueva.
train_t.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("train")
val_t.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("val")
test_t.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("test")
print(f"TRAIN: {train_t.count():,} | VAL: {val_t.count():,} | TEST: {test_t.count():,}")

# ── Verificacion (guardar esta salida para el Anexo D) ─────────────────────
# CLAVE: se lee de vuelta la TABLA ESCRITA EN DISCO, no train_t en memoria.
# Antes se verificaba train_t (el DataFrame en memoria), asi que si la
# escritura Delta iba a otro sitio o fallaba en silencio, la verificacion
# no lo detectaba -- y el modelo acababa entrenando sobre una tabla vieja
# sin normalizar (rango I6 hasta ~230000 en vez de ~[-3,3]).
# Refrescar la cache del catalogo antes de releer, para que la verificacion
# lea la version que ACABAMOS de escribir y no una cacheada de esta sesion.
spark.catalog.refreshTable("train")
written = spark.read.table("train")

print("\nVerificacion — I1..I5 (log1p), leido de la tabla escrita:")
written.select(LOG_COLS).describe().show()
print("Verificacion — I6..I13 (StandardScaler, esperado: mean~0, stddev~1):")
written.select(STD_COLS).describe().show()

# Asercion dura: si I6..I13 NO estan normalizadas en la tabla escrita, parar
# aqui con un error claro en vez de dejar que 04/05 entrenen sobre datos
# crudos y produzcan curvas espuriamente lineales (hallazgo A2).
from pyspark.sql import functions as F
stats = written.select(
    *[F.stddev(c).alias(f"std_{c}") for c in STD_COLS]
).collect()[0]
bad = [c for c in STD_COLS if stats[f"std_{c}"] is not None and stats[f"std_{c}"] > 5.0]
if bad:
    raise RuntimeError(
        f"La tabla 'train' escrita NO esta normalizada: {bad} tienen stddev>5 "
        f"(deberia ser ~1.0). El StandardScaler no llego a estas columnas. "
        f"Revisa que apply_pipeline_and_unpack se aplico y que la escritura "
        f"Delta apunta a la tabla correcta. NO ejecutes 04/05 hasta arreglar esto."
    )
print("\n✓ Verificacion OK: la tabla 'train' escrita esta normalizada (I6..I13 stddev~1).")
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
