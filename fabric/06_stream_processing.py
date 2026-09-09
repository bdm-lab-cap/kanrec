# Microsoft Fabric Notebook — 06_stream_processing
# Processes streaming_kfk with the pre-fitted MLlib Pipeline from 01.
# Run AFTER 02_streaming_kafka. Attach kanrec_lakehouse before running.
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

from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from kanrec.spark_utils import apply_pipeline_and_unpack
import pandas as pd

NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
LOG_COLS = [f"I{i}" for i in range(1, 6)]
STD_COLS = [f"I{i}" for i in range(6, 14)]

print("Loading MLlib Pipeline...")
pipeline_model = PipelineModel.load("Files/models/mlllib_pipeline")

print("Reading streaming_kfk...")
raw = spark.read.table("streaming_kfk")
print(f"Rows: {raw.count():,}")

# Flatten numerical/categorical structs
flat = raw.select(
    col("timestamp"), col("label").cast("integer"),
    *[col(f"numerical.{c}").alias(c) for c in NUMERICAL_COLS],
    *[col(f"categorical.{c}").alias(c) for c in CATEGORICAL_COLS],
)

for c in NUMERICAL_COLS:
    flat = flat.withColumn(c, F.when(F.col(c).isNull(), 0.0).otherwise(F.abs(F.col(c).cast("float"))))
for c in LOG_COLS:
    flat = flat.withColumn(c, F.log1p(F.greatest(F.col(c), F.lit(0.0))))

# Same helper as 01: applies the fitted Pipeline AND unpacks StandardScaler's
# vector output back into I6..I13, instead of leaving it in an unused column.
processed = apply_pipeline_and_unpack(flat, pipeline_model, STD_COLS)

output_cols = (
    ["timestamp", "label"] + NUMERICAL_COLS
    + [f"C{i}_idx" for i in range(1, 27) if f"C{i}_idx" in processed.columns]
)
stream_processed = processed.select(output_cols)
stream_processed.write.format("delta").mode("overwrite").save("Tables/streaming_processed")

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
