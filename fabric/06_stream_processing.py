# Microsoft Fabric Notebook — 06_stream_processing
#
# Consume de forma incremental la tabla `streaming_kfk` que alimenta
# Eventstream desde Confluent Cloud, la normaliza con la misma funcion que el
# batch, la puntua con el modelo KAN-REC y vigila la deriva de las entradas
# respecto a train. Attach kanrec_lakehouse before running.
#
# Procesamiento incremental
#   Structured Streaming sobre la tabla Delta con trigger(availableNow=True):
#   procesa solo lo llegado desde la ejecucion anterior, con checkpoint de
#   offsets, y termina. Es un trabajo con principio y fin, de modo que encaja
#   en el pipeline de Data Factory sin dejar un job vivo.
#
# Consistencia batch/stream
#   La normalizacion es kanrec.spark_utils.normalise_like_train, la misma
#   funcion que llama 01, con los mismos scaler_stats.json y cat_index_maps.
#   Ademas, check_manifest verifica por hash que el checkpoint se entreno con
#   esos mismos estadisticos antes de puntuar una sola impresion.
#
# Salidas
#   streaming_predictions   P(click) por impresion
#   streaming_processed     impresiones normalizadas
#   streaming_metrics       por lote: CTR observado y predicho, log-loss, AUC
#   streaming_drift         por lote y campo: PSI y cobertura del rango calibrado
#
# Dependencias (las escriben 01 y 04):
#   Files/config/scaler_stats.json                                (01)
#   Files/config/feature_selection.json                           (01)
#   Tables/cat_index_maps                                         (01)
#   Files/checkpoints/best_kan-bspline_gs10_s42.pt                (04)
#   Files/checkpoints/best_kan-bspline_gs10_s42.pt.manifest.json  (04)
#   Files/config/drift_reference.json                             (se crea aqui)
#
# ---------------------------------------------------------------------------
# Instalacion de kanrec en ejecucion por PIPELINE
# %pip install esta deshabilitado cuando el notebook se ejecuta desde un Data
# Pipeline: el paquete debe estar en el entorno de Fabric. score_spark usa
# mapInPandas, asi que kanrec tiene que estar tambien en los EJECUTORES.
# ---------------------------------------------------------------------------
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col

from kanrec.drift import DriftReference, calibrated_ranges, drift_report, summarize
from kanrec.serving import check_manifest, load_model, score_spark
from kanrec.spark_utils import (
    CATEGORICAL_COLS_DEFAULT, LOG_COLS_DEFAULT, STD_COLS_DEFAULT,
    load_index_maps, normalise_like_train,
)

NUMERICAL_COLS   = LOG_COLS_DEFAULT + STD_COLS_DEFAULT          # I1..I13
CATEGORICAL_COLS = CATEGORICAL_COLS_DEFAULT                     # C1..C26
IDX_COLS         = [f"{c}_idx" for c in CATEGORICAL_COLS]

FILES            = "/lakehouse/default/Files"
STATS_PATH       = f"{FILES}/config/scaler_stats.json"
SELECTION_PATH   = f"{FILES}/config/feature_selection.json"
DRIFT_REF_PATH   = f"{FILES}/config/drift_reference.json"
CKPT_PATH        = f"{FILES}/checkpoints/best_kan-bspline_gs10_s42.pt"
STREAM_CKPT      = "Files/stream_checkpoints/06_stream_processing"   # offsets de Structured Streaming

# ── Dependencias ────────────────────────────────────────────────────────────
_faltan = []
for _p, _who in [(STATS_PATH, "01"), (SELECTION_PATH, "01"), (CKPT_PATH, "04"),
                 (f"{CKPT_PATH}.manifest.json", "04 (celda final: write_manifest)")]:
    if not os.path.exists(_p):
        _faltan.append(f"{_p} (lo escribe {_who})")
try:
    spark.read.table("cat_index_maps").limit(1).count()
except Exception:
    _faltan.append("tabla cat_index_maps (la escribe 01)")
if _faltan:
    raise FileNotFoundError("Faltan artefactos:\n  - " + "\n  - ".join(_faltan))

# ── Verificacion del modelo y de los estadisticos ───────────────────────────
with open(STATS_PATH) as f:
    scaler_stats = json.load(f)
with open(SELECTION_PATH) as f:
    _sel = json.load(f)
if _sel["selected"] != NUMERICAL_COLS:
    raise RuntimeError(f"feature_selection.json dice {_sel['selected']}, el stream usa {NUMERICAL_COLS}")

manifest = check_manifest(CKPT_PATH, scaler_stats_path=STATS_PATH, numerical_cols=NUMERICAL_COLS)
MODEL_TAG = f"{manifest['checkpoint']}@{manifest['checkpoint_sha256'][:12]}"
print(f"Modelo verificado: {MODEL_TAG} | kanrec {manifest['kanrec_version']} | "
      f"git {manifest.get('git_sha') or '?'} | encoder {manifest['encoder']} "
      f"grid={manifest.get('grid_size')} d={manifest['embedding_dim']}")

index_maps = load_index_maps(spark, CATEGORICAL_COLS)

# ── Rangos calibrados del encoder (para la cobertura) ───────────────────────
_model_driver = load_model(CKPT_PATH, device="cpu")
CAL_RANGES = calibrated_ranges(_model_driver.numerical_encoder)
del _model_driver
print("Rango calibrado por campo:",
      {c: (round(lo, 2), round(hi, 2)) for c, (lo, hi) in zip(NUMERICAL_COLS, CAL_RANGES)})

# ── Referencia de deriva: bins de cuantiles sobre TRAIN, una sola vez ───────
if os.path.exists(DRIFT_REF_PATH):
    drift_ref = DriftReference.load(DRIFT_REF_PATH)
    print(f"Referencia de deriva cargada ({drift_ref.n_ref:,} filas de train).")
else:
    print("Construyendo referencia de deriva sobre train (muestra de 500k filas)...")
    _x_ref = (spark.read.table("train").select(NUMERICAL_COLS)
              .sample(fraction=0.07, seed=42).limit(500_000).toPandas()
              .to_numpy(dtype="float64"))
    drift_ref = DriftReference.fit(_x_ref, NUMERICAL_COLS, n_bins=10, calibrated_ranges=CAL_RANGES)
    os.makedirs(os.path.dirname(DRIFT_REF_PATH), exist_ok=True)
    drift_ref.save(DRIFT_REF_PATH)
    print(f"Referencia guardada en {DRIFT_REF_PATH}.")

# ── Procesamiento de cada micro-batch ───────────────────────────────────────
def _logloss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")

def process_batch(raw_batch, batch_id: int):
    t0 = time.time()
    n_in = raw_batch.count()
    if n_in == 0:
        print(f"[batch {batch_id}] vacío, nada que hacer")
        return

    flat = raw_batch.select(
        col("timestamp"), col("label").cast("integer"),
        *[col(f"numerical.{c}").alias(c) for c in NUMERICAL_COLS],
        *[col(f"categorical.{c}").alias(c) for c in CATEGORICAL_COLS],
    )

    # 1. Misma transformacion que 01: misma funcion y mismos artefactos
    processed = normalise_like_train(flat, scaler_stats, index_maps,
                                     numerical_cols=NUMERICAL_COLS,
                                     log_cols=LOG_COLS_DEFAULT, std_cols=STD_COLS_DEFAULT,
                                     categorical_cols=CATEGORICAL_COLS)
    processed = processed.select("timestamp", "label", *NUMERICAL_COLS, *IDX_COLS)

    # 2. Scoring con el checkpoint verificado
    scored = (score_spark(processed, CKPT_PATH, NUMERICAL_COLS, IDX_COLS, output_col="p_click")
              .withColumn("batch_id", F.lit(batch_id))
              .withColumn("model", F.lit(MODEL_TAG))
              .withColumn("scored_at", F.current_timestamp()))
    scored.persist()

    (scored.select("timestamp", "label", "p_click", "batch_id", "model", "scored_at")
           .write.format("delta").mode("append").saveAsTable("streaming_predictions"))
    (scored.select("timestamp", "label", *NUMERICAL_COLS, *IDX_COLS, "batch_id")
           .write.format("delta").mode("append").option("mergeSchema", "true")
           .saveAsTable("streaming_processed"))

    # 3. Metricas del lote
    pdf = scored.select("label", "p_click", *NUMERICAL_COLS).toPandas()
    y, p = pdf["label"].to_numpy(dtype=float), pdf["p_click"].to_numpy(dtype=float)
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "batch_id": batch_id,
        "impressions": int(len(y)), "clicks": int(y.sum()),
        "ctr_observed": float(y.mean()), "ctr_predicted": float(p.mean()),
        "logloss": _logloss(y, p), "auc": _auc(y, p),
        "model": MODEL_TAG, "source": "confluent-kafka",
    }

    # 4. Deriva por campo frente a train
    report = drift_report(pdf[NUMERICAL_COLS].to_numpy(dtype="float64"), drift_ref)
    for r in report:
        r.update({"batch_id": batch_id, "timestamp": metrics["timestamp"], "model": MODEL_TAG})
    spark.createDataFrame(pd.DataFrame(report)).write.format("delta").mode("append") \
         .saveAsTable("streaming_drift")

    s = summarize(report)
    metrics.update({"max_psi": s["max_psi"], "worst_psi_field": s["worst_psi_field"],
                    "min_coverage": s["min_coverage"],
                    "n_fields_psi_alert": len(s["fields_psi_alert"]),
                    "n_fields_coverage_alert": len(s["fields_coverage_alert"]),
                    "processing_seconds": round(time.time() - t0, 2)})
    spark.createDataFrame(pd.DataFrame([metrics])).write.format("delta").mode("append") \
         .option("mergeSchema", "true").saveAsTable("streaming_metrics")
    scored.unpersist()

    print(f"[batch {batch_id}] {len(y):,} impresiones | CTR obs {metrics['ctr_observed']:.4f} "
          f"vs pred {metrics['ctr_predicted']:.4f} | logloss {metrics['logloss']:.4f} "
          f"| AUC {metrics['auc']:.4f} | PSI máx {s['max_psi']:.3f} ({s['worst_psi_field']}) "
          f"| cobertura mín {s['min_coverage']:.4f} | {metrics['processing_seconds']}s")
    if s["fields_psi_alert"] or s["fields_coverage_alert"]:
        print(f"   ⚠ deriva: PSI {s['fields_psi_alert']} · cobertura {s['fields_coverage_alert']}")

# ── Consumo incremental ─────────────────────────────────────────────────────
# availableNow consume lo disponible hasta ahora y termina; los offsets quedan
# en STREAM_CKPT, de modo que la siguiente ejecucion empieza donde acabo esta.
# Para reprocesar desde cero, borrar STREAM_CKPT.
query = (spark.readStream.format("delta").table("streaming_kfk")
         .writeStream
         .foreachBatch(process_batch)
         .option("checkpointLocation", STREAM_CKPT)
         .trigger(availableNow=True)
         .start())
query.awaitTermination()
print("\n✓ 06 completado. Tablas actualizadas: streaming_processed, streaming_predictions, "
      "streaming_metrics, streaming_drift")

# ── Verificacion ────────────────────────────────────────────────────────────
print("\nI6..I13 en el stream (esperado mean~0, stddev~1, igual que en 01):")
spark.read.table("streaming_processed").select(STD_COLS_DEFAULT).describe().show()
print("Últimos lotes:")
(spark.read.table("streaming_metrics")
      .select("batch_id", "impressions", "ctr_observed", "ctr_predicted", "logloss", "auc",
              "max_psi", "worst_psi_field", "min_coverage", "model")
      .orderBy(F.desc("batch_id")).show(5, truncate=False))
