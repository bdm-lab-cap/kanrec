"""
Kafka producer: replays Criteo CSV as a real-time stream of ad impressions.

Usage:
    python infra/kafka_producer.py --csv data/criteo_10m.tsv --delay-ms 10
    python infra/kafka_producer.py --csv data/criteo_10m.tsv --delay-ms 0 --max-rows 100000
"""
import json
import time
import argparse
import pandas as pd
from confluent_kafka import Producer

TOPIC = "ad_impressions"
NUMERICAL_COLS = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]


def delivery_report(err, msg):
    if err:
        print(f"[ERROR] Delivery failed: {err}")


def stream_criteo(csv_path: str, delay_ms: int = 10, max_rows: int = None,
                  bootstrap: str = "localhost:9092"):
    producer = Producer({"bootstrap.servers": bootstrap})
    print(f"Streaming {csv_path} → topic '{TOPIC}' (delay={delay_ms}ms)")

    cols = ["label"] + NUMERICAL_COLS + CATEGORICAL_COLS
    reader = pd.read_csv(
        csv_path, sep="\t", header=None, names=cols,
        chunksize=1000, nrows=max_rows
    )

    total = 0
    for chunk in reader:
        for _, row in chunk.iterrows():
            record = {
                "timestamp": int(time.time() * 1000),
                "label": int(row["label"]),
                "numerical": {
                    col: float(row[col]) if pd.notna(row[col]) else 0.0
                    for col in NUMERICAL_COLS
                },
                "categorical": {
                    col: str(row[col]) if pd.notna(row[col]) else "<UNK>"
                    for col in CATEGORICAL_COLS
                },
            }
            producer.produce(
                TOPIC,
                key=str(total),
                value=json.dumps(record),
                callback=delivery_report,
            )
            total += 1
            if total % 10_000 == 0:
                producer.flush()
                print(f"  {total:,} records sent")
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)

    producer.flush()
    print(f"Done: {total:,} records sent to '{TOPIC}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="data/criteo_10m.tsv")
    parser.add_argument("--delay-ms",  type=int, default=10)
    parser.add_argument("--max-rows",  type=int, default=None)
    parser.add_argument("--bootstrap", default="localhost:9092")
    args = parser.parse_args()
    stream_criteo(args.csv, args.delay_ms, args.max_rows, args.bootstrap)
