# Microsoft Fabric Notebook — 03_api_ingest
# Pulls item metadata from REST API and enriches the train set.
# Requires: mock API server running locally + ngrok tunnel active.
# Attach kanrec_lakehouse before running.

# ---------------------------------------------------------------------------
# IMPORTANTE — instalacion de kanrec en ejecucion por PIPELINE
#
# `%pip install` esta DESHABILITADO cuando un notebook se ejecuta desde un
# Data Pipeline: solo funciona en sesiones interactivas. Verificado en Fabric:
#   MagicUsageError: %pip magic command is disabled
#
# Por eso la primera celda de cada notebook NO instala nada. El paquete se
# resuelve por una de estas dos vias, ambas compatibles con pipeline:
#
#   A) Carpeta en Files (rapida, sin publicar entorno). Subir la carpeta
#      `kanrec/` a Files/libs/ y anadir al inicio del notebook:
#
#          import sys
#          sys.path.insert(0, "/lakehouse/default/Files/libs")
#
#   B) Entorno de Fabric (la via formal). Workspace -> Nuevo -> Entorno ->
#      Bibliotecas personalizadas -> subir kanrec-0.3.2-py3-none-any.whl ->
#      Publicar -> asignar el entorno al workspace o al notebook.
#
# Las dependencias (torch, scipy, scikit-learn, pandas, pyarrow) ya vienen en
# el runtime de Fabric, asi que ninguna de las dos vias necesita resolverlas.
# ---------------------------------------------------------------------------
import requests
import pandas as pd
from pyspark.sql import functions as F

# ── Resolucion de la URL de la API ─────────────────────────────────────────
# En ejecucion por PIPELINE no sirve `%env`: es un comando magico, igual que
# `%pip`, y aunque se ejecutase, el valor fijado en una sesion interactiva no
# viaja al pipeline. La URL se resuelve por tres vias, en este orden:
#
#   1. Parametro del pipeline `api_base` (recomendado). En la actividad de
#      notebook: Configuracion -> Parametros de base -> api_base = https://...
#   2. Variable de entorno ITEM_API_BASE (uso interactivo).
#   3. Fichero Files/config/api_config.json, que sobrevive entre sesiones:
#          {"api_base": "https://<subdominio>.ngrok-free.dev"}
import json
import os

def _resolver_api_base() -> str:
    if "api_base" in globals() and globals()["api_base"]:      # parametro del pipeline
        return str(globals()["api_base"]).rstrip("/")
    entorno = os.environ.get("ITEM_API_BASE", "").strip()
    if entorno:
        return entorno.rstrip("/")
    try:
        with open("/lakehouse/default/Files/config/api_config.json") as f:
            return str(json.load(f).get("api_base", "")).rstrip("/")
    except Exception:
        return ""

API_BASE = _resolver_api_base()


def _api_disponible(base: str) -> bool:
    """
    Comprueba que la API responde y devuelve JSON de verdad.

    No basta con que la peticion no lance: un tunel caido devuelve una pagina
    HTML de error con codigo 200, y `resp.json()` falla entonces con
    JSONDecodeError, que es el error que rompia este notebook en el pipeline.
    """
    if not base:
        print("  API_BASE sin configurar.")
        return False
    try:
        r = requests.get(f"{base}/health", timeout=10)
    except Exception as e:
        print(f"  API inalcanzable: {type(e).__name__}: {e}")
        return False
    if r.status_code != 200:
        print(f"  API responde {r.status_code}, no 200.")
        return False
    if "application/json" not in r.headers.get("content-type", ""):
        print(f"  API devuelve {r.headers.get('content-type')}, no JSON "
              f"(tunel caido o pagina de error).")
        return False
    try:
        print(f"  API OK: {r.json()}")
        return True
    except ValueError:
        print("  API devuelve un cuerpo que no es JSON valido.")
        return False


print("Comprobando disponibilidad de la API...")
API_OK = _api_disponible(API_BASE)

if not API_OK:
    # El enriquecimiento es OPCIONAL: los metadatos son sinteticos y ningun
    # notebook de entrenamiento los consume. Hacer caer todo el pipeline
    # porque un tunel de desarrollo esta apagado seria fragil y ademas
    # bloquearia las etapas que si son criticas. Se registra y se sale.
    print("\nEnriquecimiento OMITIDO: la API no esta disponible.")
    print("Para ejecutarlo: levanta la API y el tunel, y pasa la URL como")
    print("parametro 'api_base' de la actividad de notebook en el pipeline.")
    try:
        import notebookutils
        notebookutils.notebook.exit("API no disponible; enriquecimiento omitido")
    except ImportError:
        raise SystemExit(0)

print("Reading unique item IDs from train set...")
train = spark.read.table("train")
item_ids = (train.select("C1")
                  .filter(F.col("C1").isNotNull())
                  .distinct()
                  .limit(10000)
                  .rdd.map(lambda r: r["C1"])
                  .collect())
print(f"Unique item IDs: {len(item_ids):,}")

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
