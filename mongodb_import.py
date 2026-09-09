"""
Import symbolic results and model metrics to MongoDB Atlas.

Usage:
    cp .env.example .env      # rellena ATLAS_URI
    python experiments/mongodb_import.py

The connection string is never stored in this file: it is resolved by
``kanrec.config`` from Azure Key Vault, the environment or ``.env``.

MongoDB Atlas cluster: ClusterKanrec (M0 free)
Collections: symbolic_results, model_alerts, item_embeddings
"""
import json
from datetime import datetime, timezone

from pymongo.mongo_client import MongoClient

from kanrec.config import atlas_uri

def import_symbolic_results(client, results_path: str = "data/symbolic_results.json"):
    """Import symbolic extraction results to MongoDB Atlas."""
    col = client["kanrec"]["symbolic_results"]
    with open(results_path) as f:
        data = json.load(f)

    docs = []
    for seed_str, seed_results in data["results_by_seed"].items():
        seed = int(seed_str)
        for field_name, r in seed_results.items():
            if r["params"] is None:
                continue
            docs.append({
                "run_id":    f"fabric-s{seed}",
                "dataset":   "criteo",
                "seed":      seed,
                "encoder":   "kan-bspline",
                "field":     field_name,
                "operator":  r["operator"],
                "r2":        r["r2"],
                "formula":   r["formula"],
                "params":    r["params"],
                "accepted":  r["accepted"],
                "timestamp": datetime.now(timezone.utc),
            })

    if docs:
        col.insert_many(docs)
        print(f"✓ {len(docs)} symbolic results imported to MongoDB Atlas")

    # Stability aggregation
    pipeline = [
        {"$match": {"encoder": "kan-bspline", "field": {"$ne": None}}},
        {"$group": {"_id": "$field", "avg_r2": {"$avg": "$r2"},
                    "n_seeds": {"$sum": 1}, "avg_auc": {"$avg": "$test_auc"}}},
        {"$sort": {"avg_r2": -1}}
    ]
    print("\nStability report:")
    for r in col.aggregate(pipeline):
        print(f"  {r['_id']:<6}: avg_R²={r.get('avg_r2',0):.4f} | seeds={r['n_seeds']}")


def import_model_alerts(client, results: list):
    """Import model training results to model_alerts collection."""
    col = client["kanrec"]["model_alerts"]
    docs = [{"run_id": r.get("run_id", f"run-{i}"),
             "encoder": r["encoder"], "dataset": r["dataset"],
             "seed": r["seed"], "auc": r["test_auc"],
             "logloss": r["test_logloss"], "timestamp": datetime.now(timezone.utc)}
            for i, r in enumerate(results)]
    col.insert_many(docs)
    print(f"✓ {len(docs)} model alerts imported to MongoDB Atlas")


if __name__ == "__main__":
    client = MongoClient(atlas_uri())
    client.admin.command("ping")
    print("✓ Connected to MongoDB Atlas (ClusterKanrec)")

    # Import symbolic results if available
    import os
    if os.path.exists("data/symbolic_results.json"):
        import_symbolic_results(client)
    else:
        print("data/symbolic_results.json not found — download from Fabric Files/results/")

    print(f"\nTotal documents:")
    for col_name in ["symbolic_results", "model_alerts", "item_embeddings"]:
        count = client["kanrec"][col_name].count_documents({})
        print(f"  kanrec.{col_name}: {count:,}")

    client.close()
