# Microsoft Fabric Notebook — 06_stream_processing
# Processes streaming_kfk with the pre-fitted MLlib Pipeline from 01.
# Consume la tabla 'streaming_kfk' que alimenta Eventstream desde Confluent
# Cloud. Requiere que el Eventstream este activo y haya recibido eventos.
# Attach kanrec_lakehouse before running.
#
# Requires the kanrec package installed in this session:
#   %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"
#
# IMPORTANTE: kanrec<0.3.2 arrastraba mlflow como dependencia obligatoria,
# lo que rompia el mlflow propio de Fabric (ver 01_spark_ingest_mlllib.py
# para el detalle completo del fallo). kanrec>=0.3.2 ya no lo hace. Si tras
# actualizar sigue fallando, reinicia el kernel de PySpark.
#
# Fix applied (2026-09, auditoria de tribunal - hallazgo A2, consistencia
# batch/stream)
# --------------------------------------------------------------------------
# This notebook called `pipeline_model.transform(flat)` directly and then
# selected NUMERICAL_COLS by name. That gave the same bug as 01 before its
# fix: StandardScaler's output lives in a separate vector column
# ("num_scaled") that was never unpacked, so I6..I13 in
# Tables/streaming_processed were still in their raw scale — exactly the
# "Pipeline serialised guarantees consistency between batch and stream"
# claim the memoria makes, except it wasn't true for the stream side.
#
# Now both notebooks call the same kanrec.spark_utils.apply_pipeline_and_unpack
# helper, so batch and stream are provably consistent (same function, not
# just "the same Pipeline object" — that part was already true and was
# never the bug).

import json
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import pandas as pd

NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
LOG_COLS = [f"I{i}" for i in range(1, 6)]
STD_COLS = [f"I{i}" for i in range(6, 14)]

# Cargar los estadisticos del scaler y los mapas de indices que ESCRIBIO 01.
# 01 dejo de usar el PipelineModel de MLlib (su StringIndexer abortaba con
# valores categoricos que parecen registros CSV malformados). En su lugar
# guardo scaler_stats.json y la tabla cat_index_maps, que 06 reutiliza aqui
# para aplicar EXACTAMENTE la misma transformacion al stream que al batch.
print("Loading scaler stats and categorical index maps from 01...")
with open("/lakehouse/default/Files/config/scaler_stats.json") as f:
    scaler_stats = json.load(f)
index_maps_df = spark.read.table("cat_index_maps")  # columnas: column, value, idx

print("Reading streaming_kfk...")
raw = spark.read.table("streaming_kfk")
print(f"Rows: {raw.count():,}")

flat = raw.select(
    col("timestamp"), col("label").cast("integer"),
    *[col(f"numerical.{c}").alias(c) for c in NUMERICAL_COLS],
    *[col(f"categorical.{c}").alias(c) for c in CATEGORICAL_COLS],
)

for c in NUMERICAL_COLS:
    flat = flat.withColumn(c, F.when(F.col(c).isNull(), 0.0).otherwise(F.abs(F.col(c))))
for c in LOG_COLS:
    flat = flat.withColumn(c, F.log1p(F.greatest(F.col(c), F.lit(0.0))))

# StandardScaler nativo con los stats de TRAIN (misma media/desviacion que el batch)
for c in STD_COLS:
    m, s = scaler_stats[c]["mean"], scaler_stats[c]["std"]
    flat = flat.withColumn(c, (F.col(c) - F.lit(m)) / F.lit(s if s else 1.0))

# Indexado categorico con los mapas de TRAIN (join broadcast, no-vistas -> n_cats)
for c in CATEGORICAL_COLS:
    mp = (index_maps_df.filter(F.col("column") == c)
          .select(F.col("value").alias(c), F.col("idx").alias(f"{c}_idx")))
    n_cats = mp.count()
    flat = (flat.join(F.broadcast(mp), on=c, how="left")
                .withColumn(f"{c}_idx", F.coalesce(F.col(f"{c}_idx"), F.lit(n_cats)).cast("int")))

processed = flat
output_cols = (
    ["timestamp", "label"] + NUMERICAL_COLS
    + [f"C{i}_idx" for i in range(1, 27) if f"C{i}_idx" in processed.columns]
)
stream_processed = processed.select(output_cols)
stream_processed.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("streaming_processed")

# ── Verificacion (guardar para el Anexo D) ─────────────────────────────────
print("\nVerificacion — I6..I13 en el stream (esperado: mean~0, stddev~1, igual que en 01):")
stream_processed.select(STD_COLS).describe().show()

# Metrics
ctr = stream_processed.agg(F.avg("label").alias("ctr")).collect()[0]["ctr"]
clicks = stream_processed.filter(F.col("label") == 1).count()
total  = stream_processed.count()

from datetime import datetime
metrics = [{"timestamp": datetime.now().isoformat(), "impressions": total,
            "clicks": clicks, "ctr": float(ctr), "source": "confluent-kafka"}]
spark.createDataFrame(pd.DataFrame(metrics)).write.format("delta").mode("append").save("Tables/streaming_metrics")

print(f"\n✓ streaming_processed: {total:,} rows")
print(f"✓ CTR: {ctr:.4f} ({ctr*100:.2f}%) | Clicks: {clicks:,} | Impressions: {total:,}")
