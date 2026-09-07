# Microsoft Fabric Notebook — 02_eventstream_kafka
# Ejecutar en: Fabric Workspace → Real-Time Intelligence → Eventstream
#
# SETUP PREVIO EN FABRIC UI:
#   1. Real-Time Intelligence → Eventstream → New Eventstream
#   2. Name: kanrec-stream
#   3. Source: Custom App (Kafka compatible endpoint)
#      → Copia el Bootstrap server y el connection string
#   4. Destination: Lakehouse → kanrec_lakehouse → tabla: streaming_raw
#
# Este notebook configura el procesamiento del stream con Spark Structured
# Streaming como alternativa al Eventstream UI para mayor control.

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, LongType,
    FloatType, StringType, MapType
)
from pyspark.ml import PipelineModel

LAKEHOUSE_PATH = "abfss://kanrec@onelake.dfs.fabric.microsoft.com/kanrec_lakehouse.Lakehouse/Files"

# ── Cargar Pipeline MLlib pre-ajustado (mismos estadísticos que batch) ────────
print("Cargando Pipeline MLlib pre-ajustado...")
pipeline_model = PipelineModel.load(f"{LAKEHOUSE_PATH}/models/mlllib_pipeline")

# ── Schema del mensaje Kafka ──────────────────────────────────────────────────
msg_schema = StructType([
    StructField("timestamp",   LongType(),   True),
    StructField("label",       IntegerType(), True),
    StructField("numerical",   MapType(StringType(), FloatType()), True),
    StructField("categorical", MapType(StringType(), StringType()), True),
])

NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]

# ── Opción A: Fabric Eventstream endpoint (Kafka compatible) ──────────────────
# Sustituye los valores con los del Eventstream creado en Fabric UI
EVENTSTREAM_BOOTSTRAP = "YOUR-EVENTSTREAM.servicebus.windows.net:9093"
EVENTSTREAM_TOPIC     = "kanrec-stream"
EVENTSTREAM_CONN_STR  = "YOUR-CONNECTION-STRING"

stream_df = (spark.readStream
             .format("kafka")
             .option("kafka.bootstrap.servers", EVENTSTREAM_BOOTSTRAP)
             .option("subscribe", EVENTSTREAM_TOPIC)
             .option("kafka.security.protocol", "SASL_SSL")
             .option("kafka.sasl.mechanism", "PLAIN")
             .option("kafka.sasl.jaas.config",
                     f'org.apache.kafka.common.security.plain.PlainLoginModule required '
                     f'username="$ConnectionString" password="{EVENTSTREAM_CONN_STR}";')
             .option("startingOffsets", "earliest")
             .option("maxOffsetsPerTrigger", 10000)
             .load())

# ── Parsear mensajes JSON ─────────────────────────────────────────────────────
parsed = (stream_df
          .select(F.from_json(F.col("value").cast("string"), msg_schema).alias("d"))
          .select("d.*"))

for c in NUMERICAL_COLS:
    parsed = parsed.withColumn(c, F.col("numerical")[c].cast("float"))
for c in CATEGORICAL_COLS:
    parsed = parsed.withColumn(c, F.col("categorical")[c])
parsed = parsed.drop("numerical", "categorical")

# ── Aplicar Pipeline MLlib pre-ajustado en cada micro-batch ──────────────────
def process_batch(batch_df, batch_id):
    if batch_df.count() == 0:
        return
    enriched = pipeline_model.transform(batch_df)
    (enriched.write
     .format("delta")
     .mode("append")
     .saveAsTable("kanrec_lakehouse.streaming_raw"))
    print(f"Batch {batch_id}: {batch_df.count()} rows → streaming_raw")

query = (parsed.writeStream
         .foreachBatch(process_batch)
         .option("checkpointLocation",
                 f"{LAKEHOUSE_PATH}/checkpoints/streaming")
         .trigger(processingTime="30 seconds")
         .start())

print("Streaming activo. Consumiendo desde Fabric Eventstream.")
print("Datos escritos en: kanrec_lakehouse.streaming_raw")
