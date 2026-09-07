# Microsoft Fabric Notebook — 05_ml_experiments
# Registra los resultados del entrenamiento KAN (ejecutado en Colab)
# en Fabric ML Experiments para centralizar el tracking en el workspace.
#
# FLUJO:
#   1. Colab entrena el modelo y guarda métricas en metrics.json
#   2. Sube metrics.json + checkpoint a OneLake (via Azure SDK)
#   3. Este notebook lee los resultados y los registra en ML Experiments

import json
import mlflow
import mlflow.pytorch

LAKEHOUSE_PATH = "abfss://kanrec@onelake.dfs.fabric.microsoft.com/kanrec_lakehouse.Lakehouse/Files"

mlflow.set_experiment("kanrec-training")

# ── Leer resultados de Colab desde OneLake ────────────────────────────────────
results_path = f"{LAKEHOUSE_PATH}/results/training_results.json"
results_json = dbutils.fs.head(results_path)
results = json.loads(results_json)

print("Registrando resultados en Fabric ML Experiments...")

for run_info in results["runs"]:
    with mlflow.start_run(run_name=f"{run_info['encoder']}-{run_info['dataset']}-s{run_info['seed']}"):

        # Parámetros
        mlflow.log_params({
            "encoder":       run_info["encoder"],
            "dataset":       run_info["dataset"],
            "seed":          run_info["seed"],
            "embedding_dim": run_info.get("embedding_dim", 16),
            "grid_size":     run_info.get("grid_size", 10),
            "spline_order":  run_info.get("spline_order", 3),
            "n_fields":      run_info.get("n_fields"),
        })

        # Métricas
        mlflow.log_metrics({
            "test_auc":      run_info["test_auc"],
            "test_logloss":  run_info["test_logloss"],
            "val_auc":       run_info.get("best_val_auc", 0),
            "latency_ms":    run_info.get("latency_ms", 0),
        })

        print(f"  ✓ {run_info['encoder']} seed={run_info['seed']} "
              f"AUC={run_info['test_auc']:.4f}")

print("\nTodos los experimentos registrados en Fabric ML Experiments.")
print("Visualiza en: Fabric Workspace → Data Science → ML Experiments → kanrec-training")

# ── Tabla comparativa en Delta ────────────────────────────────────────────────
from pyspark.sql import SparkSession
import pandas as pd

results_df = spark.createDataFrame(pd.DataFrame(results["runs"]))
results_df.write.format("delta").mode("overwrite").saveAsTable("kanrec_lakehouse.experiment_results")
print("Tabla experiment_results creada — conecta con Power BI para visualización.")
