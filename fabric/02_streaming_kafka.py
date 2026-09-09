# Microsoft Fabric Notebook — 02_streaming_kafka
# Simulates Kafka Structured Streaming using Fabric pipeline.
# In production: Fabric Eventstream would connect to the external Kafka broker via ngrok.
# Attach kanrec_lakehouse before running.

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
LOG_COLS = [f"I{i}" for i in range(1, 6)]

print("Loading pre-fitted MLlib Pipeline...")
pipeline_model = PipelineModel.load("Files/models/mlllib_pipeline")

print("Loading raw data to simulate Kafka stream...")
raw = (spark.read.option("sep", "\t").option("inferSchema", "true")
       .csv("Files/raw/criteo_10m.tsv")
       .toDF("label", *NUMERICAL_COLS, *CATEGORICAL_COLS))

for c in NUMERICAL_COLS:
    raw = raw.withColumn(c, F.when(F.col(c).isNull(), 0.0)
                           .otherwise(F.abs(F.col(c).cast("float"))))

for c in LOG_COLS:
    raw = raw.withColumn(c, F.log1p(F.greatest(F.col(c), F.lit(0.0))))

print("Processing micro-batches (simulating Structured Streaming)...")
for batch_id in range(3):
    batch_df = (raw.limit(10000)
                   .withColumn("kafka_timestamp", F.current_timestamp())
                   .withColumn("batch_id", F.lit(batch_id)))
    enriched = pipeline_model.transform(batch_df)
    (enriched.write.format("delta").mode("append").save("Tables/streaming_raw"))
    print(f"  Batch {batch_id}: {batch_df.count():,} rows -> streaming_raw")

total = spark.read.format("delta").load("Tables/streaming_raw").count()
print(f"\nTotal in streaming_raw: {total:,}")
print("""
NOTE: In production, replace this simulation with:
  spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "YOUR-NGROK-URL:443")
    .option("subscribe", "ad_impressions")
    .load()
The pre-fitted MLlib Pipeline guarantees consistency between batch and stream.
""")
