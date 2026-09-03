# Databricks notebook — pulls item metadata from REST API and merges into Delta Lake
#
# Requires the mock API server running and accessible:
#   Local:   uvicorn data.mock_api_server:app --port 8000
#   Public:  ngrok http 8000  → update API_BASE below

import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType
)

spark = SparkSession.builder.appName("kanrec-api-ingest").getOrCreate()

API_BASE  = "http://localhost:8000"   # change to ngrok URL if running remotely
DELTA_BASE = "/dbfs/FileStore/kanrec/delta"

# ── 1. Collect unique item IDs from training set ──────────────────────────────
train = spark.read.format("delta").load(f"{DELTA_BASE}/train")
item_ids = (train.select("C1")
                  .filter(F.col("C1") != "<UNK>")
                  .distinct()
                  .limit(50_000)
                  .rdd.map(lambda r: r["C1"])
                  .collect())
print(f"Unique item IDs to enrich: {len(item_ids):,}")

# ── 2. Batch pull from API ────────────────────────────────────────────────────
BATCH_SIZE = 500
records = []
for i in range(0, len(item_ids), BATCH_SIZE):
    batch = item_ids[i:i + BATCH_SIZE]
    resp  = requests.post(f"{API_BASE}/items/batch", json=batch, timeout=30)
    resp.raise_for_status()
    records.extend(resp.json())
    if i % 5_000 == 0:
        print(f"  {i:,}/{len(item_ids):,} processed")

print(f"Metadata retrieved: {len(records):,} items")

# ── 3. Save metadata to Delta Lake ───────────────────────────────────────────
meta_schema = StructType([
    StructField("item_id",           StringType(),  True),
    StructField("price_usd",         FloatType(),   True),
    StructField("category",          StringType(),  True),
    StructField("popularity_score",  FloatType(),   True),
    StructField("days_since_listing", IntegerType(), True),
    StructField("avg_rating",        FloatType(),   True),
])
meta_df = spark.createDataFrame(records, schema=meta_schema)
meta_df.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/item_metadata")
print(f"Metadata saved: {DELTA_BASE}/item_metadata")

# ── 4. Enrich train split with metadata ───────────────────────────────────────
enriched = (train
            .join(meta_df.withColumnRenamed("item_id", "C1"), on="C1", how="left")
            .withColumn("price_usd",        F.coalesce(F.col("price_usd"),        F.lit(0.0)))
            .withColumn("popularity_score", F.coalesce(F.col("popularity_score"), F.lit(0.5)))
            .withColumn("days_since_listing", F.coalesce(F.col("days_since_listing"), F.lit(365))))

enriched.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/train_enriched")
print(f"Enriched train saved. New features: price_usd, category, popularity_score, days_since_listing, avg_rating")

# Apply same enrichment to val and test
for split in ["val", "test"]:
    df = spark.read.format("delta").load(f"{DELTA_BASE}/{split}")
    df_enr = (df.join(meta_df.withColumnRenamed("item_id", "C1"), on="C1", how="left")
                .withColumn("price_usd",          F.coalesce(F.col("price_usd"),          F.lit(0.0)))
                .withColumn("popularity_score",   F.coalesce(F.col("popularity_score"),   F.lit(0.5)))
                .withColumn("days_since_listing", F.coalesce(F.col("days_since_listing"), F.lit(365))))
    df_enr.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/{split}_enriched")
    print(f"{split.upper()} enriched and saved.")
