# Databricks notebook — consolidates enriched splits and exports Parquet for PyTorch

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("kanrec-feature-store").getOrCreate()
DELTA_BASE = "/dbfs/FileStore/kanrec/delta"

print("=" * 60)
print("KAN-REC FEATURE STORE — SUMMARY")
print("=" * 60)

for split in ["train_enriched", "val_enriched", "test_enriched"]:
    df = spark.read.format("delta").load(f"{DELTA_BASE}/{split}")
    n   = df.count()
    pos = df.filter(F.col("label") == 1).count()
    print(f"  {split:20s}: {n:>10,} rows | CTR={pos/n:.4f} | cols={len(df.columns)}")

print("=" * 60)

# Export to Parquet for direct use by the PyTorch DataLoader
for split, src in [("train", "train_enriched"),
                   ("val",   "val_enriched"),
                   ("test",  "test_enriched")]:
    df = spark.read.format("delta").load(f"{DELTA_BASE}/{src}")
    out = f"{DELTA_BASE}/parquet/{split}"
    df.write.mode("overwrite").parquet(out)
    print(f"Parquet exported: {out}")

print("\nFeature store ready. Download the parquet/ folder to your local machine")
print("or Colab before running experiments/train.py")
