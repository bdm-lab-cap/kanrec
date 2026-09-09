# Microsoft Fabric Notebook — 06_stream_processing
# Processes streaming_kfk with pre-fitted MLlib Pipeline.
# Run AFTER 02_streaming_kafka. Attach kanrec_lakehouse before running.

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
import pandas as pd

NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
LOG_COLS = [f"I{i}" for i in range(1, 6)]

print("Loading MLlib Pipeline...")
pipeline_model = PipelineModel.load("Files/models/mlllib_pipeline")

print("Reading streaming_kfk...")
raw = spark.read.table("streaming_kfk")
print(f"Rows: {raw.count():,}")

# Flatten numerical struct
from pyspark.sql.functions import col
flat = raw.select(
    col("timestamp"), col("label").cast("integer"),
    *[col(f"numerical.{c}").alias(c) for c in NUMERICAL_COLS],
    *[col(f"categorical.{c}").alias(c) for c in CATEGORICAL_COLS],
)

for c in NUMERICAL_COLS:
    flat = flat.withColumn(c, F.when(F.col(c).isNull(), 0.0).otherwise(F.abs(F.col(c).cast("float"))))
for c in LOG_COLS:
    flat = flat.withColumn(c, F.log1p(F.greatest(F.col(c), F.lit(0.0))))

processed = pipeline_model.transform(flat)
output_cols = ["timestamp", "label"] + NUMERICAL_COLS + [f"C{i}_idx" for i in range(1, 27) if f"C{i}_idx" in processed.columns]
stream_processed = processed.select(output_cols)
stream_processed.write.format("delta").mode("overwrite").save("Tables/streaming_processed")

# Metrics
ctr = stream_processed.agg(F.avg("label").alias("ctr")).collect()[0]["ctr"]
clicks = stream_processed.filter(F.col("label")==1).count()
total  = stream_processed.count()

from datetime import datetime
metrics = [{"timestamp": datetime.now().isoformat(), "impressions": total,
            "clicks": clicks, "ctr": float(ctr), "source": "confluent-kafka"}]
spark.createDataFrame(pd.DataFrame(metrics)).write.format("delta").mode("append").save("Tables/streaming_metrics")

print(f"✓ streaming_processed: {total:,} rows")
print(f"✓ CTR: {ctr:.4f} ({ctr*100:.2f}%) | Clicks: {clicks:,} | Impressions: {total:,}")
