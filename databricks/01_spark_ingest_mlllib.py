# Databricks notebook — run as Python script in a Databricks cluster
# Runtime: 14.3 LTS ML (Spark 3.5, Python 3.11, MLflow 2.11)
#
# PURPOSE:
#   - Load Criteo/Avazu from DBFS
#   - Normalise numerics with Spark MLlib Pipeline (StandardScaler + log1p)
#   - Compute feature importance (ChiSqSelector + Spearman correlation)
#   - Write train/val/test splits to Delta Lake
#   - Serialise the fitted Pipeline for reuse in Structured Streaming

import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer, ChiSqSelector,
)
from pyspark.ml.stat import Correlation

spark = SparkSession.builder.appName("kanrec-mlllib").getOrCreate()

NUMERICAL_COLS  = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
DELTA_BASE = "/dbfs/FileStore/kanrec/delta"
LOG_COLS = [f"I{i}" for i in range(1, 6)]   # heavy-tail counters
STD_COLS = [f"I{i}" for i in range(6, 14)]  # approx-gaussian frequencies

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading Criteo...")
raw = (spark.read
       .option("sep", "\t")
       .option("inferSchema", "true")
       .csv("/dbfs/FileStore/kanrec/criteo_10m.tsv")
       .toDF("label", *NUMERICAL_COLS, *CATEGORICAL_COLS))

for c in NUMERICAL_COLS:
    raw = raw.withColumn(c, F.when(F.col(c).isNull(), 0.0)
                           .otherwise(F.col(c).cast("float")))

# ── 2. Temporal split (80 / 10 / 10) ─────────────────────────────────────────
total = raw.count()
df = raw.withColumn("row_id", F.monotonically_increasing_id())
train_raw = df.filter(F.col("row_id") < int(total * 0.80))
val_raw   = df.filter((F.col("row_id") >= int(total * 0.80)) &
                       (F.col("row_id") <  int(total * 0.90)))
test_raw  = df.filter(F.col("row_id") >= int(total * 0.90))
print(f"Split → train={train_raw.count():,}  val={val_raw.count():,}  test={test_raw.count():,}")

# ── 3. log1p on heavy-tail columns (fit-free, apply before Pipeline) ─────────
def apply_log1p(df, cols):
    for c in cols:
        df = df.withColumn(c, F.log1p(F.col(c)))
    return df

train_log = apply_log1p(train_raw, LOG_COLS)
val_log   = apply_log1p(val_raw,   LOG_COLS)
test_log  = apply_log1p(test_raw,  LOG_COLS)

# ── 4. MLlib Pipeline (fit on TRAIN only) ─────────────────────────────────────
assembler = VectorAssembler(inputCols=STD_COLS, outputCol="num_raw")
scaler    = StandardScaler(inputCol="num_raw", outputCol="num_scaled",
                           withMean=True, withStd=True)
indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
             for c in CATEGORICAL_COLS]

pipeline = Pipeline(stages=[assembler, scaler] + indexers)
print("Fitting MLlib Pipeline on TRAIN (this may take a few minutes)...")
pipeline_model = pipeline.fit(train_log)
pipeline_model.save(f"{DELTA_BASE}/mlllib_pipeline")
print(f"Pipeline saved to {DELTA_BASE}/mlllib_pipeline")

train_t = pipeline_model.transform(train_log)
val_t   = pipeline_model.transform(val_log)
test_t  = pipeline_model.transform(test_log)

# ── 5. Feature importance ─────────────────────────────────────────────────────
print("Computing feature importance...")

# ChiSqSelector: statistical dependency between each numerical feature and label
vec_assembler = VectorAssembler(inputCols=NUMERICAL_COLS, outputCol="all_num")
train_vec = vec_assembler.transform(train_log)

selector = ChiSqSelector(numTopFeatures=13, featuresCol="all_num",
                          outputCol="selected", labelCol="label")
selector_model = selector.fit(train_vec)
chisq_ranking = selector_model.selectedFeatures
print("ChiSq ranking:", [f"I{i+1}" for i in chisq_ranking])

# Spearman correlation between each numerical field and the label
corr_matrix = Correlation.corr(train_vec, "all_num", method="spearman").head()
spearman = {f"I{i+1}": abs(float(corr_matrix[0][i, i])) for i in range(13)}

# Exclude fields in bottom quartile of BOTH metrics
import numpy as np
spear_vals = list(spearman.values())
threshold_spear = float(np.percentile(spear_vals, 25))
low_chisq = set(f"I{chisq_ranking[i]+1}" for i in range(13 - 3, 13))  # bottom 3

excluded = [c for c in NUMERICAL_COLS
            if c in low_chisq and spearman[c] < threshold_spear]
selected = [c for c in NUMERICAL_COLS if c not in excluded]

print(f"Selected ({len(selected)}): {selected}")
print(f"Excluded ({len(excluded)}): {excluded}")

selection = {
    "selected": selected,
    "excluded": excluded,
    "chisq_ranking": [f"I{i+1}" for i in chisq_ranking],
    "spearman_scores": {k: round(v, 6) for k, v in spearman.items()},
}
with open("/dbfs/FileStore/kanrec/feature_selection.json", "w") as f:
    json.dump(selection, f, indent=2)
print("feature_selection.json saved.")

# ── 6. Write to Delta Lake ────────────────────────────────────────────────────
print("Writing Delta Lake splits...")
train_t.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/train")
val_t.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/val")
test_t.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/test")
print("Done.")
