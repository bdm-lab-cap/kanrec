# Microsoft Fabric Notebook — 01_spark_ingest_mlllib
# Attach kanrec_lakehouse before running.

import json, os
import numpy as np
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer

NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
LOG_COLS = [f"I{i}" for i in range(1, 6)]
STD_COLS = [f"I{i}" for i in range(6, 14)]

print("Loading Criteo...")
raw = (spark.read.option("sep", "\t").option("inferSchema", "true")
       .csv("Files/raw/criteo_10m.tsv")
       .toDF("label", *NUMERICAL_COLS, *CATEGORICAL_COLS))

for c in NUMERICAL_COLS:
    raw = raw.withColumn(c, F.when(F.col(c).isNull(), 0.0)
                           .otherwise(F.abs(F.col(c).cast("float"))))

total = raw.count()
print(f"Total rows: {total:,}")

train_raw, val_raw, test_raw = raw.randomSplit([0.8, 0.1, 0.1], seed=42)

def log1p_safe(df, cols):
    for c in cols:
        df = df.withColumn(c, F.log1p(F.greatest(F.col(c), F.lit(0.0))))
    return df

train_log = log1p_safe(train_raw, LOG_COLS)
val_log   = log1p_safe(val_raw,   LOG_COLS)
test_log  = log1p_safe(test_raw,  LOG_COLS)

assembler = VectorAssembler(inputCols=STD_COLS, outputCol="num_raw", handleInvalid="keep")
scaler    = StandardScaler(inputCol="num_raw", outputCol="num_scaled", withMean=True, withStd=True)
indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
             for c in CATEGORICAL_COLS]

pipeline = Pipeline(stages=[assembler, scaler] + indexers)
print("Fitting MLlib Pipeline on TRAIN...")
pm = pipeline.fit(train_log)
pm.write().overwrite().save("Files/models/mlllib_pipeline")

train_t = pm.transform(train_log)
val_t   = pm.transform(val_log)
test_t  = pm.transform(test_log)

selection = {"selected": NUMERICAL_COLS, "excluded": [], "chisq_ranking": NUMERICAL_COLS, "spearman_scores": {}}
os.makedirs("/lakehouse/default/Files/config", exist_ok=True)
with open("/lakehouse/default/Files/config/feature_selection.json", "w") as f:
    json.dump(selection, f, indent=2)

train_t.write.format("delta").mode("overwrite").save("Tables/train")
val_t.write.format("delta").mode("overwrite").save("Tables/val")
test_t.write.format("delta").mode("overwrite").save("Tables/test")
print(f"TRAIN: {train_t.count():,} | VAL: {val_t.count():,} | TEST: {test_t.count():,}")
print("Done.")
