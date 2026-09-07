# Microsoft Fabric Notebook — 03_api_ingest
# Ingesta de metadata de items desde REST API y merge en OneLake

import requests
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType
)

LAKEHOUSE_PATH = "abfss://kanrec@onelake.dfs.fabric.microsoft.com/kanrec_lakehouse.Lakehouse/Files"

# ── Configuración API ─────────────────────────────────────────────────────────
# Opción A (desarrollo): ngrok expone el mock server local
#   ngrok http 8000  →  copia la URL
# Opción B (producción): despliega mock_api_server.py en Azure Container Apps
API_BASE = "https://TU-URL-NGROK.ngrok-free.app"  # sustituir

# ── 1. Obtener item IDs únicos del train set ──────────────────────────────────
train = spark.read.table("kanrec_lakehouse.train")
item_ids = (train.select("C1")
                  .filter(F.col("C1") != "<UNK>")
                  .distinct()
                  .limit(50_000)
                  .rdd.map(lambda r: r["C1"])
                  .collect())
print(f"Item IDs únicos: {len(item_ids):,}")

# ── 2. Batch pull desde API ───────────────────────────────────────────────────
BATCH_SIZE = 500
records = []
for i in range(0, len(item_ids), BATCH_SIZE):
    batch = item_ids[i:i + BATCH_SIZE]
    resp  = requests.post(f"{API_BASE}/items/batch", json=batch, timeout=30)
    resp.raise_for_status()
    records.extend(resp.json())
    if i % 5_000 == 0:
        print(f"  {i:,}/{len(item_ids):,}")

print(f"Metadata obtenida: {len(records):,} items")

# ── 3. Guardar metadata como Delta Table en OneLake ───────────────────────────
meta_schema = StructType([
    StructField("item_id",            StringType(),  True),
    StructField("price_usd",          FloatType(),   True),
    StructField("category",           StringType(),  True),
    StructField("popularity_score",   FloatType(),   True),
    StructField("days_since_listing", IntegerType(), True),
    StructField("avg_rating",         FloatType(),   True),
])
meta_df = spark.createDataFrame(records, schema=meta_schema)
meta_df.write.format("delta").mode("overwrite").saveAsTable("kanrec_lakehouse.item_metadata")
print("Tabla item_metadata creada en OneLake.")

# ── 4. Enriquecer splits con metadata ─────────────────────────────────────────
for split in ["train", "val", "test"]:
    df = spark.read.table(f"kanrec_lakehouse.{split}")
    enriched = (df.join(meta_df.withColumnRenamed("item_id", "C1"), on="C1", how="left")
                  .withColumn("price_usd",          F.coalesce(F.col("price_usd"),          F.lit(0.0)))
                  .withColumn("popularity_score",   F.coalesce(F.col("popularity_score"),   F.lit(0.5)))
                  .withColumn("days_since_listing", F.coalesce(F.col("days_since_listing"), F.lit(365))))
    enriched.write.format("delta").mode("overwrite").saveAsTable(f"kanrec_lakehouse.{split}_enriched")
    print(f"{split.upper()} enriquecido: {enriched.count():,} filas")

print("Ingesta de metadata completada.")
