# Microsoft Fabric Notebook — 01_spark_ingest_mlllib
# Ejecutar en: Fabric Workspace → Data Engineering → Notebook
# Runtime: Spark (incluido en Fabric, no necesita configuración adicional)
#
# PURPOSE:
#   - Cargar Criteo desde Fabric Lakehouse
#   - Normalizar con Spark MLlib Pipeline
#   - Calcular feature importance (ChiSqSelector + Spearman)
#   - Escribir train/val/test en OneLake (Delta format)
#   - Registrar experimento en Fabric ML Experiments (MLflow)

import json
import mlflow
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer, ChiSqSelector
)
from pyspark.ml.stat import Correlation

# ── Configuración ────────────────────────────────────────────────────────────
LAKEHOUSE_NAME  = "kanrec_lakehouse"   # nombre del Lakehouse creado en Fabric
LAKEHOUSE_PATH  = f"abfss://kanrec@onelake.dfs.fabric.microsoft.com/{LAKEHOUSE_NAME}.Lakehouse/Files"
TABLES_PATH     = f"abfss://kanrec@onelake.dfs.fabric.microsoft.com/{LAKEHOUSE_NAME}.Lakehouse/Tables"

NUMERICAL_COLS  = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
LOG_COLS = [f"I{i}" for i in range(1, 6)]
STD_COLS = [f"I{i}" for i in range(6, 14)]

# ── 1. Iniciar MLflow (Fabric ML Experiments) ─────────────────────────────────
mlflow.set_experiment("kanrec-preprocessing")

with mlflow.start_run(run_name="criteo-mlllib-pipeline"):

    # ── 2. Cargar datos desde Lakehouse ──────────────────────────────────────
    print("Cargando Criteo desde Lakehouse...")
    raw = (spark.read
           .option("sep", "\t")
           .option("inferSchema", "true")
           .csv(f"{LAKEHOUSE_PATH}/raw/criteo_10m.tsv")
           .toDF("label", *NUMERICAL_COLS, *CATEGORICAL_COLS))

    for c in NUMERICAL_COLS:
        raw = raw.withColumn(c, F.when(F.col(c).isNull(), 0.0)
                               .otherwise(F.col(c).cast("float")))

    total = raw.count()
    mlflow.log_param("total_rows", total)
    print(f"Total filas: {total:,}")

    # ── 3. Split temporal (80/10/10) ──────────────────────────────────────────
    df = raw.withColumn("row_id", F.monotonically_increasing_id())
    train_raw = df.filter(F.col("row_id") < int(total * 0.80))
    val_raw   = df.filter((F.col("row_id") >= int(total * 0.80)) &
                           (F.col("row_id") <  int(total * 0.90)))
    test_raw  = df.filter(F.col("row_id") >= int(total * 0.90))

    # ── 4. log1p para columnas de cola larga ──────────────────────────────────
    def apply_log1p(df, cols):
        for c in cols:
            df = df.withColumn(c, F.log1p(F.col(c)))
        return df

    train_log = apply_log1p(train_raw, LOG_COLS)
    val_log   = apply_log1p(val_raw,   LOG_COLS)
    test_log  = apply_log1p(test_raw,  LOG_COLS)

    # ── 5. MLlib Pipeline (ajustado solo en TRAIN) ────────────────────────────
    assembler = VectorAssembler(inputCols=STD_COLS, outputCol="num_raw")
    scaler    = StandardScaler(inputCol="num_raw", outputCol="num_scaled",
                               withMean=True, withStd=True)
    indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                               handleInvalid="keep")
                 for c in CATEGORICAL_COLS]

    pipeline = Pipeline(stages=[assembler, scaler] + indexers)
    print("Ajustando Pipeline MLlib en TRAIN...")
    pipeline_model = pipeline.fit(train_log)

    # Guardar Pipeline en OneLake
    pipeline_model.save(f"{LAKEHOUSE_PATH}/models/mlllib_pipeline")
    print("Pipeline guardado en OneLake.")

    train_t = pipeline_model.transform(train_log)
    val_t   = pipeline_model.transform(val_log)
    test_t  = pipeline_model.transform(test_log)

    # ── 6. Feature importance ─────────────────────────────────────────────────
    print("Calculando feature importance...")
    vec_asm   = VectorAssembler(inputCols=NUMERICAL_COLS, outputCol="all_num")
    train_vec = vec_asm.transform(train_log)

    selector = ChiSqSelector(numTopFeatures=13, featuresCol="all_num",
                              outputCol="selected", labelCol="label")
    sel_model    = selector.fit(train_vec)
    chisq_ranking = sel_model.selectedFeatures

    corr_matrix = Correlation.corr(train_vec, "all_num", method="spearman").head()
    spearman    = {f"I{i+1}": abs(float(corr_matrix[0][i, i])) for i in range(13)}

    import numpy as np
    threshold_spear = float(np.percentile(list(spearman.values()), 25))
    low_chisq = set(f"I{chisq_ranking[i]+1}" for i in range(13-3, 13))

    excluded = [c for c in NUMERICAL_COLS
                if c in low_chisq and spearman[c] < threshold_spear]
    selected = [c for c in NUMERICAL_COLS if c not in excluded]

    print(f"Seleccionados ({len(selected)}): {selected}")
    print(f"Excluidos    ({len(excluded)}): {excluded}")

    selection = {
        "selected": selected,
        "excluded": excluded,
        "chisq_ranking": [f"I{i+1}" for i in chisq_ranking],
        "spearman_scores": {k: round(v, 6) for k, v in spearman.items()},
    }

    mlflow.log_param("selected_fields", selected)
    mlflow.log_param("excluded_fields", excluded)
    mlflow.log_metric("n_selected_fields", len(selected))

    # Guardar selección como JSON en OneLake
    sel_json = json.dumps(selection, indent=2)
    dbutils.fs.put(f"{LAKEHOUSE_PATH}/config/feature_selection.json",
                   sel_json, overwrite=True)

    # ── 7. Escribir Delta Tables en OneLake ───────────────────────────────────
    print("Escribiendo Delta Tables en OneLake...")
    train_t.write.format("delta").mode("overwrite").saveAsTable("kanrec_lakehouse.train")
    val_t.write.format("delta").mode("overwrite").saveAsTable("kanrec_lakehouse.val")
    test_t.write.format("delta").mode("overwrite").saveAsTable("kanrec_lakehouse.test")

    n_train = train_t.count()
    n_val   = val_t.count()
    n_test  = test_t.count()

    mlflow.log_metric("train_rows", n_train)
    mlflow.log_metric("val_rows",   n_val)
    mlflow.log_metric("test_rows",  n_test)

    print(f"TRAIN: {n_train:,} | VAL: {n_val:,} | TEST: {n_test:,}")
    print("Pipeline completado. Tablas Delta disponibles en OneLake.")
