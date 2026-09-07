# Microsoft Fabric Notebook — 06_data_activator_setup
# Configura alertas automáticas via Data Activator (Reflex)
# cuando el AUC de un nuevo entrenamiento cae por debajo del baseline.
#
# SETUP PREVIO EN FABRIC UI (más sencillo que via código):
#   1. Fabric Workspace → Real-Time Intelligence → Data Activator
#   2. New Reflex item → name: kanrec-alerts
#   3. Get data from: Power BI report → kanrec_metrics
#   4. Trigger: When test_auc < baseline_auc - 0.005
#   5. Action: Send email to pedroantonio@ucm.es
#
# Este notebook genera la tabla de baseline para que Data Activator la use.

from pyspark.sql import functions as F
import json

# ── Leer resultados de experimentos ───────────────────────────────────────────
exp_df = spark.read.table("kanrec_lakehouse.experiment_results")

# ── Calcular baseline por encoder y dataset ───────────────────────────────────
baseline = (exp_df
            .groupBy("encoder", "dataset")
            .agg(
                F.avg("test_auc").alias("baseline_auc"),
                F.avg("test_logloss").alias("baseline_logloss"),
                F.count("*").alias("n_runs"),
            )
            .withColumn("alert_threshold_auc",
                        F.col("baseline_auc") - F.lit(0.005)))

baseline.show()
baseline.write.format("delta").mode("overwrite").saveAsTable("kanrec_lakehouse.baseline_metrics")
print("Tabla baseline_metrics creada.")

# ── Instrucciones para Data Activator ─────────────────────────────────────────
print("""
═══════════════════════════════════════════════════════════
CONFIGURAR DATA ACTIVATOR EN FABRIC UI:
═══════════════════════════════════════════════════════════

1. En el workspace KAN-REC → + New → Reflex

2. Nombre: kanrec-model-alerts

3. Fuente de datos: lakehouse → kanrec_lakehouse → experiment_results

4. Objeto: cada fila es un experimento (key = run_id)

5. Condición: test_auc < (SELECT alert_threshold_auc
                          FROM baseline_metrics
                          WHERE encoder = 'kan-bspline'
                          AND dataset = 'criteo')

6. Acción: Send email
   Para: pedroa09@ucm.es
   Asunto: [KAN-REC] Alerta: AUC por debajo del baseline
   Cuerpo: El experimento {run_id} ha obtenido AUC={test_auc},
           por debajo del threshold ({alert_threshold_auc}).

7. Guardar y activar.
═══════════════════════════════════════════════════════════
""")
