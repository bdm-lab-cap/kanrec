# Microsoft Fabric Notebook — 09_mongodb_persistence
#
# Persiste en MongoDB Atlas las formulas simbolicas extraidas por 05, usando
# el mismo esquema y los mismos indices que el resto del proyecto
# (kanrec.schema y kanrec.mongo_store), de modo que el informe de estabilidad
# del store lee los documentos escritos aqui sin traduccion intermedia.
#
# Es la actividad que cierra la rama de interpretabilidad del pipeline:
# 04 entrena -> 05 extrae las formulas -> 09 las deja consultables fuera del
# lakehouse, que es lo que un auditor necesita.
#
# Attach kanrec_lakehouse before running. Requiere kanrec en el entorno del
# workspace (no %pip: esta deshabilitado en ejecucion por pipeline).
#
# Dependencias:
#   Tables/symbolic_results                          (05)
#   Files/results/symbolic_results.json              (05)
#   Secreto 'atlas-uri' en el Key Vault 'kanrec-kv'
import json
import os

from kanrec.config import atlas_uri
from kanrec.mongo_store import MongoSymbolicStore
from kanrec.schema import SymbolicResult

RESULTS_PATH = "/lakehouse/default/Files/results/symbolic_results.json"
if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(f"{RESULTS_PATH} no existe; ejecuta antes 05_symbolic_extraction.")

with open(RESULTS_PATH) as f:
    data = json.load(f)

# Un fichero de resultados vacio significa que 05 no extrajo nada: subirlo
# dejaria la coleccion en un estado que parece correcto y no lo es.
results = []
for seed_str, seed_results in data["results_by_seed"].items():
    seed = int(seed_str)
    for field_name, fit in seed_results.items():
        if fit.get("params") is None:
            continue
        results.append(SymbolicResult.from_fit(
            run_id=f"fabric-s{seed}", dataset="criteo",
            seed=seed, field_name=field_name, fit=fit,
        ))
if not results:
    raise RuntimeError("symbolic_results.json no contiene ningun ajuste valido.")

store = MongoSymbolicStore(uri=atlas_uri())
n = store.upsert_results(results)
print(f"{n} resultados escritos en MongoDB Atlas (upsert sobre la clave natural)")

# Informe de estabilidad resuelto en el servidor con un pipeline de agregacion:
# agrupa por campo y cuenta cuantos operadores distintos ganan entre semillas.
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
print("\nEstabilidad entre semillas:")
for r in store.col.aggregate(pipeline):
    estado = "estable" if r["n_operadores"] == 1 else f"{r['n_operadores']} operadores"
    print(f"  {r['_id']:<6}: R2={r.get('avg_r2', 0):.4f} | {r['n_seeds']} semillas | {estado}")

store.close()
print("\n09 completado.")
