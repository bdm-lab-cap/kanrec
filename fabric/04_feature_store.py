# Microsoft Fabric Notebook — 04_feature_store
# Consolida los splits enriquecidos y exporta Parquet para entrenamiento en Colab

import json
import mlflow
from pyspark.sql import functions as F

LAKEHOUSE_PATH = "abfss://kanrec@onelake.dfs.fabric.microsoft.com/kanrec_lakehouse.Lakehouse/Files"

mlflow.set_experiment("kanrec-preprocessing")

with mlflow.start_run(run_name="feature-store-consolidation"):

    print("=" * 60)
    print("KAN-REC FEATURE STORE — OneLake")
    print("=" * 60)

    for split in ["train_enriched", "val_enriched", "test_enriched"]:
        df  = spark.read.table(f"kanrec_lakehouse.{split}")
        n   = df.count()
        pos = df.filter(F.col("label") == 1).count()
        print(f"  {split:22s}: {n:>10,} filas | CTR={pos/n:.4f} | cols={len(df.columns)}")
        mlflow.log_metric(f"{split}_rows", n)
        mlflow.log_metric(f"{split}_ctr",  round(pos/n, 4))

    print("=" * 60)

    # ── Exportar Parquet para consumo desde Colab ──────────────────────────────
    for split, src in [("train", "train_enriched"),
                       ("val",   "val_enriched"),
                       ("test",  "test_enriched")]:
        df = spark.read.table(f"kanrec_lakehouse.{src}")
        out = f"{LAKEHOUSE_PATH}/parquet/{split}"
        df.write.mode("overwrite").parquet(out)
        print(f"Parquet exportado: Files/parquet/{split}")

    print("\nFeature store listo en OneLake.")
    print("Descarga Files/parquet/ desde Fabric UI → OneLake → Files")
    print("o usa el SDK de Azure para descargarlo a Colab/Mac.")
