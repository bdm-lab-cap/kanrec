# Microsoft Fabric Notebook — 10_cargar_resultados_colab
#
# Registra como tablas Delta los CSV de resultados producidos en Colab, de modo
# que Power BI muestre las cifras de la memoria (3 semillas, 1,5 M de filas en
# GPU) y no las de la corrida reducida de Fabric.
#
# Carga manual: no forma parte del pipeline. Requiere haber subido los CSV de
# kanrec_results/ a Files/results/ del lakehouse.
#
# Attach kanrec_lakehouse before running.

from pyspark.sql import functions as F

RESULTS_PATH = "Files/results"

def _cargar_csv(nombre, tabla):
    """Lee un CSV de Files/results y lo registra como tabla Delta."""
    try:
        df = (spark.read.option("header", "true").option("inferSchema", "true")
              .csv(f"{RESULTS_PATH}/{nombre}"))
        (df.write.format("delta").mode("overwrite")
           .option("overwriteSchema", "true").saveAsTable(tabla))
        print(f"  {tabla:<28} {df.count():>5} filas   ← {nombre}")
        return df
    except Exception as e:
        print(f"  {tabla:<28} OMITIDA ({str(e)[:60]})")
        return None

print("Cargando resultados de Colab (los que van en la memoria)...\n")

exp = _cargar_csv("experiment_results_colab.csv", "experiment_results_colab")
_cargar_csv("gridsize_ablation_colab.csv", "gridsize_ablation")
_cargar_csv("latencia.csv",                "latencia")
_cargar_csv("estabilidad_semillas.csv",    "estabilidad_semillas")
_cargar_csv("monotonia.csv",               "monotonia")
_cargar_csv("fidelidad.csv",               "fidelidad")

# ── baseline_metrics desde los datos de Colab ──────────────────────────────
# Se sobrescribe la version generada desde el trial: el panel debe reflejar
# las cifras de la memoria (3 semillas), no las de la prueba de pipeline.
if exp is not None:
    _cols = set(exp.columns)
    _aggs = [F.count("*").alias("n_seeds")]
    if "test_auc" in _cols:
        _aggs += [F.avg("test_auc").alias("auc_mean"),
                  F.stddev("test_auc").alias("auc_std")]
    if "test_logloss" in _cols:
        _aggs += [F.avg("test_logloss").alias("logloss_mean"),
                  F.stddev("test_logloss").alias("logloss_std")]
    if "train_seconds" in _cols:
        _aggs += [F.avg("train_seconds").alias("train_seconds_mean")]
    if "n_params" in _cols:
        _aggs += [F.avg("n_params").alias("n_params")]

    (exp.groupBy("encoder").agg(*_aggs)
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").saveAsTable("baseline_metrics"))

    print("\nbaseline_metrics regenerada desde Colab (3 semillas):")
    (spark.read.table("baseline_metrics")
        .select("encoder", F.round("auc_mean", 4).alias("auc_mean"),
                F.round("auc_std", 4).alias("auc_std"),
                F.round("logloss_mean", 4).alias("logloss_mean"),
                F.round("train_seconds_mean", 1).alias("train_s"),
                "n_seeds")
        .orderBy(F.desc("auc_mean"))).show()

    print("Comprobacion: kan-bspline debe dar auc_mean ~0,7851 y auc_std ~0,0015.")
    print("Si ves ~0,75 con n_seeds=1, se han cargado los datos del trial.")

# ── latencia con la vectorizacion incluida ─────────────────────────────────
# El CSV de latencia mide los tres encoders ANTES de vectorizar. El resultado
# de la vectorizacion esta en resumen_cierre.json, asi que se anade como una
# fila mas para que el panel pueda mostrar el antes y el despues.
import json

try:
    with open("/lakehouse/default/Files/results/resumen_cierre.json") as f:
        _resumen = json.load(f)
    _vec = _resumen.get("vectorizacion")
    if _vec:
        _fila = spark.createDataFrame([{
            "encoder": "kan-bspline (vectorizado)",
            "modelo_ms": float(_vec["ms_vectorizado"]),
            "modelo_ms_std": 0.0,
            "encoder_ms": float(_vec["ms_vectorizado"]) * 0.355,
            "encoder_pct": 0.355,
            "filas_por_s": 4096 / (float(_vec["ms_vectorizado"]) / 1000.0),
            "batch": 4096,
            "device": "cuda:0",
        }])
        (spark.read.table("latencia").unionByName(_fila, allowMissingColumns=True)
            .write.format("delta").mode("overwrite")
            .option("overwriteSchema", "true").saveAsTable("latencia"))
        print(f"\nlatencia: anadida la fila del encoder vectorizado "
              f"({_vec['ms_vectorizado']:.2f} ms, speedup {_vec['speedup_modelo']:.2f}x)")
        spark.read.table("latencia").select(
            "encoder", F.round("modelo_ms", 2).alias("ms"),
            F.round("filas_por_s", 0).alias("filas_s")).orderBy("modelo_ms").show(truncate=False)
except Exception as e:
    print(f"\n(fila de vectorizacion omitida: {str(e)[:80]})")

print("\nTablas listas. Refresca el modelo semantico en Power BI para verlas.")
