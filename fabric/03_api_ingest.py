# Microsoft Fabric Notebook — 03_api_ingest
# Pulls item metadata from REST API and enriches the train set.
# Requires: mock API server running locally + ngrok tunnel active.
# Attach kanrec_lakehouse before running.

import requests
import pandas as pd
from pyspark.sql import functions as F

# Update with your ngrok URL before running
API_BASE = "https://YOUR-NGROK-URL.ngrok-free.dev"

print("Reading unique item IDs from train set...")
train = spark.read.table("train")
item_ids = (train.select("C1")
                  .filter(F.col("C1").isNotNull())
                  .distinct()
                  .limit(10000)
                  .rdd.map(lambda r: r["C1"])
                  .collect())
print(f"Unique item IDs: {len(item_ids):,}")

print("Checking API health...")
resp = requests.get(f"{API_BASE}/health", timeout=10)
print(f"API status: {resp.json()}")

BATCH_SIZE = 500
records = []
for i in range(0, len(item_ids), BATCH_SIZE):
    batch = item_ids[i:i + BATCH_SIZE]
    try:
        resp = requests.post(f"{API_BASE}/items/batch", json=batch, timeout=30)
        resp.raise_for_status()
        records.extend(resp.json())
        if i % 2000 == 0:
            print(f"  {i:,}/{len(item_ids):,} processed")
    except Exception as e:
        print(f"  Error in batch {i}: {e}")

print(f"Metadata retrieved: {len(records):,} items")

if records:
    meta_df = spark.createDataFrame(pd.DataFrame(records))
    meta_df.write.format("delta").mode("overwrite").save("Tables/item_metadata")
    print(f"item_metadata table created: {len(records):,} items")
    meta_df.show(5)

    enriched = (train
                .join(meta_df.withColumnRenamed("item_id", "C1"), on="C1", how="left")
                .withColumn("price_usd",          F.coalesce(F.col("price_usd"),          F.lit(0.0)))
                .withColumn("popularity_score",   F.coalesce(F.col("popularity_score"),   F.lit(0.5)))
                .withColumn("days_since_listing", F.coalesce(F.col("days_since_listing"), F.lit(365)))
                .withColumn("avg_rating",         F.coalesce(F.col("avg_rating"),         F.lit(3.0))))

    cols_simple = ["label", "C1", "price_usd", "popularity_score", "days_since_listing", "avg_rating"]
    enriched.select(cols_simple).write.format("delta").mode("overwrite").save("Tables/train_enriched_metadata")
    print(f"train_enriched_metadata created: {enriched.count():,} rows")
