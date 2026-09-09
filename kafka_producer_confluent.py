"""
Kafka producer that sends Criteo impressions to Confluent Cloud.

Usage:
    cp .env.example .env      # rellena CONFLUENT_API_KEY / CONFLUENT_API_SECRET
    python infra/kafka_producer_confluent.py --max-rows 500 --delay-ms 50

Credentials are never stored in this file: they are resolved by
``kanrec.config`` from Azure Key Vault, the environment or ``.env``.

Confluent Cloud cluster: lkc-5756j1z (GCP us-east1)
Topic: ad_impressions
"""
import argparse
import json
import time

from confluent_kafka import Producer

from kanrec.config import confluent_config, get_secret

NUM_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]


def delivery_report(err, msg):
    if err:
        print(f"ERROR: {err}")


def build_message(line: str, index: int) -> dict | None:
    """Parses one raw Criteo TSV line into the topic's JSON schema."""
    parts = line.strip().split("\t")
    if len(parts) < 40:
        return None
    return {
        "timestamp": int(time.time() * 1000),
        "label": int(parts[0]) if parts[0] else 0,
        "numerical": {
            NUM_COLS[j]: float(parts[j + 1]) if parts[j + 1] else 0.0
            for j in range(13)
        },
        "categorical": {
            CAT_COLS[j]: parts[j + 14] if parts[j + 14] else "<UNK>"
            for j in range(26)
        },
    }


def stream(csv_path: str, max_rows: int = 500, delay_ms: int = 50) -> int:
    config = confluent_config()
    topic = get_secret("CONFLUENT_TOPIC", default="ad_impressions", required=False)

    producer = Producer(config)
    print(f"Connecting to Confluent Cloud ({config['bootstrap.servers']})...")
    print(f"Sending up to {max_rows:,} records to topic '{topic}'...")

    sent = 0
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            msg = build_message(line, i)
            if msg is None:
                continue
            producer.produce(
                topic, key=str(i), value=json.dumps(msg), callback=delivery_report
            )
            sent += 1
            if sent % 500 == 0:
                producer.flush()
                print(f"  {sent:,} messages sent to Confluent Cloud")
            time.sleep(delay_ms / 1000)

    producer.flush()
    print(f"Completed: {sent:,} messages sent to topic '{topic}'.")
    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAN-REC Confluent Cloud producer")
    parser.add_argument("--csv", default="data/criteo_1m.tsv")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--delay-ms", type=int, default=50)
    args = parser.parse_args()
    stream(args.csv, args.max_rows, args.delay_ms)
