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
from kanrec.mongo_store import MongoSymbolicStore
from kanrec.schema import SymbolicResult


def import_symbolic_results(client, results_path: str = "data/symbolic_results.json"):
    """
    Importa los resultados de extraccion simbolica a MongoDB Atlas.

    Usa MongoSymbolicStore (y por tanto el esquema unico de kanrec.schema y
    sus indices) en vez de escribir directamente sobre la coleccion: asi los
    documentos son legibles por el informe de estabilidad del propio store.
    """
    store = MongoSymbolicStore(uri=atlas_uri())
    with open(results_path) as f:
        data = json.load(f)

    # Los documentos se construyen con el esquema UNICO de kanrec.schema.
    # Antes este script insertaba con 'field'/'accepted' mientras
    # MongoSymbolicStore agregaba por 'field_name'/'is_accepted': los
    # documentos importados aqui eran invisibles para el informe de
    # estabilidad, que devolvia resultados parciales.
    results = []
    for seed_str, seed_results in data["results_by_seed"].items():
        seed = int(seed_str)
        for field_name, r in seed_results.items():
            if r.get("params") is None:
                continue
            results.append(SymbolicResult.from_fit(
                run_id=f"fabric-s{seed}",
                dataset="criteo",
                seed=seed,
                field_name=field_name,
                fit=r,
            ))

    if results:
        n = store.upsert_results(results)
        print(f"✓ {n} resultados simbolicos escritos en MongoDB Atlas "
              f"(upsert sobre la clave natural: reimportar no duplica)")

    # Informe de estabilidad: agrupa por el MISMO campo que escriben ambos
    # productores, y filtra por is_accepted.
    pipeline = [
        {"$match": {"encoder": "kan-bspline", "is_accepted": True}},
        {"$group": {"_id": "$field_name",
                    "operadores": {"$addToSet": "$operator"},
                    "avg_r2": {"$avg": "$r2"},
                    "n_seeds": {"$sum": 1}}},
        {"$project": {"avg_r2": 1, "n_seeds": 1, "operadores": 1,
                      "n_operadores": {"$size": "$operadores"}}},
        {"$sort": {"_id": 1}},
    ]
    print("\nInforme de estabilidad (desde el pipeline de agregacion):")
    for r in store.col.aggregate(pipeline):
        estable = "estable" if r["n_operadores"] == 1 else f"{r['n_operadores']} operadores"
        print(f"  {r['_id']:<6}: R²={r.get('avg_r2', 0):.4f} | "
              f"{r['n_seeds']} semillas | {estable}")


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
