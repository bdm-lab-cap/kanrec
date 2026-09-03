# Databricks notebook — Spark Structured Streaming consumer for Kafka topic
#
# IMPORTANT: Databricks Community Edition cannot reach localhost:9092.
# Options (pick one):
#   A) Use Confluent Cloud free tier (public endpoint, easiest)
#   B) Use ngrok to expose local broker: ngrok tcp 9092
#   C) Write Kafka messages to JSONL on DBFS and read with readStream("json")
#
# This notebook uses Option C (DBFS JSONL) as default — no extra setup needed.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, LongType,
    FloatType, StringType, MapType
)
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("kanrec-streaming").getOrCreate()

NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
DELTA_BASE = "/dbfs/FileStore/kanrec/delta"

# Load the already-fitted MLlib Pipeline (no retraining needed)
print("Loading pre-fitted MLlib Pipeline...")
pipeline_model = PipelineModel.load(f"{DELTA_BASE}/mlllib_pipeline")

msg_schema = StructType([
    StructField("timestamp",   LongType(),   True),
    StructField("label",       IntegerType(), True),
    StructField("numerical",   MapType(StringType(), FloatType()), True),
    StructField("categorical", MapType(StringType(), StringType()), True),
])

# ── Option C: Read JSONL files written by the producer to DBFS ───────────────
# The producer writes one JSON per line to /dbfs/FileStore/kanrec/stream_input/
raw_stream = (spark.readStream
              .schema(msg_schema)
              .option("maxFilesPerTrigger", 5)
              .json("/dbfs/FileStore/kanrec/stream_input/"))

# Expand map columns to individual feature columns
parsed = raw_stream
for c in NUMERICAL_COLS:
    parsed = parsed.withColumn(c, F.col("numerical")[c].cast("float"))
for c in CATEGORICAL_COLS:
    parsed = parsed.withColumn(c, F.col("categorical")[c])
parsed = parsed.drop("numerical", "categorical")


def process_batch(batch_df, batch_id):
    """Apply the pre-fitted Pipeline to each micro-batch and append to Delta."""
    if batch_df.count() == 0:
        return
    enriched = pipeline_model.transform(batch_df)
    (enriched.write
     .format("delta")
     .mode("append")
     .save(f"{DELTA_BASE}/streaming_raw"))
    print(f"Batch {batch_id}: {batch_df.count()} rows written to streaming_raw")


query = (parsed.writeStream
         .foreachBatch(process_batch)
         .option("checkpointLocation", f"{DELTA_BASE}/streaming/_checkpoint")
         .trigger(processingTime="30 seconds")
         .start())

print("Streaming query started. Run query.stop() to halt.")
# query.awaitTermination()  # uncomment to block until stopped
