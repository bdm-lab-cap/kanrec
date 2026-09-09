"""
Kafka producer that sends Criteo impressions to Confluent Cloud.
Usage: python infra/kafka_producer_confluent.py --max-rows 500 --delay-ms 50

Confluent Cloud cluster: lkc-5756j1z (GCP us-east1)
Topic: ad_impressions
"""
import json, time, argparse
from confluent_kafka import Producer

BOOTSTRAP  = "pkc-619z3.us-east1.gcp.confluent.cloud:9092"
API_KEY    = "***CREDENTIAL-ROTATED***"
API_SECRET = "***CREDENTIAL-ROTATED***"
TOPIC      = "ad_impressions"

config = {
    "bootstrap.servers": BOOTSTRAP,
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms":   "PLAIN",
    "sasl.username":     API_KEY,
    "sasl.password":     API_SECRET,
}

NUM_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]

def delivery_report(err, msg):
    if err:
        print(f"ERROR: {err}")

def stream(csv_path: str, max_rows: int = 500, delay_ms: int = 50):
    p = Producer(config)
    print(f"Connecting to Confluent Cloud (cluster: lkc-5756j1z)...")
    print(f"Sending {max_rows} records to topic '{TOPIC}'...")

    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            parts = line.strip().split("\t")
            if len(parts) < 40:
                continue
            msg = {
                "timestamp":   int(time.time() * 1000),
                "label":       int(parts[0]) if parts[0] else 0,
                "numerical":   {NUM_COLS[j]: float(parts[j+1]) if parts[j+1] else 0.0 for j in range(13)},
                "categorical": {CAT_COLS[j]: parts[j+14] if parts[j+14] else "<UNK>" for j in range(26)},
            }
            p.produce(TOPIC, key=str(i), value=json.dumps(msg), callback=delivery_report)
            if i % 500 == 0:
                p.flush()
                print(f"  {i:,} messages sent to Confluent Cloud")
            time.sleep(delay_ms / 1000)

    p.flush()
    print(f"Completed: {max_rows:,} messages sent to Confluent Cloud.")
    print(f"Topic: {TOPIC} | Bootstrap: {BOOTSTRAP}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAN-REC Confluent Cloud producer")
    parser.add_argument("--csv",      default="data/criteo_1m.tsv")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--delay-ms", type=int, default=50)
    args = parser.parse_args()
    stream(args.csv, args.max_rows, args.delay_ms)
